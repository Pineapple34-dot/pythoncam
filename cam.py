"""
"""

import cv2
import time
import numpy as np
from collections import defaultdict
from datetime import datetime
import sys
import threading
import os

# =================== CONFIGURATION ===================
class Config:
    # ESP32-CAM Stream
    STREAM_URL = "http://192.168.4.1/stream"
    
    # Performance
    FRAME_SKIP = 2              # Process more frames (was 3)
    DISPLAY_WINDOW = True
    FRAME_WIDTH = 640           # Better resolution (320)
    FRAME_HEIGHT = 480          # Better resolution (was 240)
    
    # Detection (OpenCV DNN)
    CONF_THRESHOLD = 0.3        # Lower = more detections (was 0.5)
    NMS_THRESHOLD = 0.4
    MIN_DETECTIONS = 1          # Announce immediately (was 2)
    
    # Audio
    TTS_ENABLED = True         # DISABLED
    TTS_COOLDOWN = 3            # Reduced to 3 seconds
    
    # Model files (will be downloaded once)
    MODEL_DIR = "models"
    WEIGHTS_FILE = "yolov4-tiny.weights"
    CONFIG_FILE = "yolov4-tiny.cfg"
    NAMES_FILE = "coco.names"

# =================== MODEL DOWNLOADER ===================
class ModelDownloader:
    """Download YOLO model files once (if not present)"""
    
    URLS = {
        'weights': 'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights',
        'config': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
        'names': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names'
    }
    
    @staticmethod
    def download_file(url, filename):
        """Download a file with progress and better error handling"""
        try:
            import urllib.request
            import ssl
            
            # Create SSL context to handle certificate issues
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            def progress(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    sys.stdout.write(f"\r  Downloading: {percent}%")
                    sys.stdout.flush()
            
            # Add headers to avoid blocking
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            
            print(f"  Downloading from: {url}")
            with urllib.request.urlopen(req, context=ssl_context, timeout=30) as response:
                with open(filename, 'wb') as out_file:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    block_size = 8192
                    
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        
                        downloaded += len(buffer)
                        out_file.write(buffer)
                        
                        if total_size > 0:
                            percent = int(downloaded * 100 / total_size)
                            sys.stdout.write(f"\r  Progress: {percent}% ({downloaded}/{total_size} bytes)")
                            sys.stdout.flush()
            
            print("\n  ✓ Download complete!")
            return True
            
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            return False
    
    @staticmethod
    def setup_models():
        """Download model files if they don't exist"""
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        
        files = {
            'weights': os.path.join(Config.MODEL_DIR, Config.WEIGHTS_FILE),
            'config': os.path.join(Config.MODEL_DIR, Config.CONFIG_FILE),
            'names': os.path.join(Config.MODEL_DIR, Config.NAMES_FILE)
        }
        
        # Check if all files exist
        all_exist = all(os.path.exists(f) for f in files.values())
        
        if all_exist:
            return files
        
        print("Downloading model files (first time only)...")
        
        # Try to download missing files
        for key, filepath in files.items():
            if not os.path.exists(filepath):
                print(f"\n[{key.upper()}]")
                success = ModelDownloader.download_file(ModelDownloader.URLS[key], filepath)
                
                if not success:
                    print(f"Failed to download {key}")
                    print(f"Download manually from: {ModelDownloader.URLS[key]}")
                    print(f"Save to: {filepath}")
                    raise Exception(f"Failed to download {key}")
        
        print("\n" + "="*60)
        print("[MODEL] ✓ All files downloaded successfully!")
        print("="*60 + "\n")
        return files

# =================== CROSS-PLATFORM TTS ===================
class TTSEngine:
    """Simple cross-platform TTS"""
    
    def __init__(self):
        self.engine_type = None
        self.lock = threading.Lock()
        self._init_engine()
    
    def _init_engine(self):
        """Initialize TTS"""
        # Try espeak first (Linux)
        if sys.platform.startswith('linux'):
            try:
                import subprocess
                result = subprocess.run(['which', 'espeak'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    self.engine_type = "espeak"
                    return
            except:
                pass
        
        # Try pyttsx3 (Windows/Mac)
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 170)
            self.engine_type = "pyttsx3"
            return
        except:
            pass
        
        self.engine_type = None
    
    def speak(self, text):
        """Non-blocking speech"""
        if not Config.TTS_ENABLED or not self.engine_type:
            return
        
        thread = threading.Thread(target=self._speak_blocking, args=(text,))
        thread.daemon = True
        thread.start()
    
    def _speak_blocking(self, text):
        """Blocking speech in thread"""
        with self.lock:
            try:
                if self.engine_type == "espeak":
                    import subprocess
                    subprocess.run(['espeak', text], capture_output=True, timeout=5)
                elif self.engine_type == "pyttsx3":
                    self.engine.say(text)
                    self.engine.runAndWait()
            except:
                pass

# =================== DETECTION EVENT MANAGER ===================
class DetectionEventManager:
    """Manage detection events with cooldown"""
    
    def __init__(self):
        self.last_spoken = defaultdict(float)
        self.detection_count = defaultdict(int)
        self.last_seen = defaultdict(float)
    
    def should_announce(self, label):
        """Check if should announce"""
        now = time.time()
        
        # Cooldown check
        if (now - self.last_spoken[label]) < Config.TTS_COOLDOWN:
            return False
        
        # Count consecutive detections
        if (now - self.last_seen[label]) < 2.0:
            self.detection_count[label] += 1
        else:
            self.detection_count[label] = 1
        
        self.last_seen[label] = now
        
        # Announce if enough detections
        if self.detection_count[label] >= Config.MIN_DETECTIONS:
            self.last_spoken[label] = now
            self.detection_count[label] = 0
            return True
        
        return False

# =================== YOLO DETECTOR (OpenCV DNN) ===================
class YOLODetector:
    """Pure OpenCV YOLO detector - No TensorFlow!"""
    
    def __init__(self):
        # Download models if needed (silently)
        files = ModelDownloader.setup_models()
        
        # Load class names
        with open(files['names'], 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Load YOLO network
        self.net = cv2.dnn.readNetFromDarknet(files['config'], files['weights'])
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layers
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect(self, frame):
        """Detect objects in frame"""
        height, width = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), 
                                     swapRB=True, crop=False)
        
        # Run detection
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > Config.CONF_THRESHOLD:
                    # Get box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 
                                   Config.CONF_THRESHOLD, 
                                   Config.NMS_THRESHOLD)
        
        # Prepare results
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    'label': self.classes[class_ids[i]],
                    'confidence': confidences[i],
                    'box': boxes[i]
                })
        
        return results

# =================== MAIN DETECTION SYSTEM ===================
class ObjectDetectionSystem:
    """Main detection system"""
    
    def __init__(self):
        self.detector = YOLODetector()
        self.tts = TTSEngine()
        self.event_manager = DetectionEventManager()
        self.cap = None
        self.frame_count = 0
        self.running = False
    
    def connect_to_stream(self):
        """Connect to ESP32-CAM"""
        try:
            self.cap = cv2.VideoCapture(Config.STREAM_URL)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = self.cap.read()
            if ret:
                print("Connected to ESP32-CAM\n")
                return True
            else:
                print("Failed to connect - check WiFi connection")
                return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def log_detection(self, label, confidence):
        """Log detection"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] DETECTED: {label.upper()} (confidence: {confidence:.0%})")
    
    def process_frame(self, frame):
        """Process frame"""
        # Resize for better visibility
        frame_resized = cv2.resize(frame, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
        
        # Detect objects
        detections = self.detector.detect(frame_resized)
        
        # Process each detection
        detected_labels = []
        for det in detections:
            label = det['label']
            conf = det['confidence']
            x, y, w, h = det['box']
            
            detected_labels.append(label)
            
            # ALWAYS log every detection to terminal (no cooldown for terminal)
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {label.upper()} detected (confidence: {conf:.0%})")
            
            # Check if should announce with TTS (this has cooldown)
            if self.event_manager.should_announce(label):
                self.tts.speak(f"{label} detected")
            
            # ALWAYS draw bounding box (green rectangle + label)
            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Draw label with background
            label_text = f"{label} {conf:.0%}"
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_resized, (x, y - text_height - 10), (x + text_width, y), (0, 255, 0), -1)
            cv2.putText(frame_resized, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Show detection count on frame
        if detected_labels:
            count_text = f"Detected: {len(detected_labels)} objects"
            cv2.putText(frame_resized, count_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame_resized
    
    def run(self):
        """Main loop"""
        if not self.connect_to_stream():
            return
        
        self.running = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    time.sleep(1)
                    if not self.connect_to_stream():
                        break
                    continue
                
                self.frame_count += 1
                
                # Process every Nth frame
                if self.frame_count % Config.FRAME_SKIP != 0:
                    cv2.imshow("ESP32-CAM Detection", frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
                    continue
                
                # Process frame for detection
                processed = self.process_frame(frame)
                
                # Always show window
                cv2.imshow("ESP32-CAM Detection", processed)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord('q'):  # Q
                    break
        
        except KeyboardInterrupt:
            pass
        
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()

# =================== MAIN ===================
if __name__ == "__main__":
    try:
        system = ObjectDetectionSystem()
        system.run()
    except Exception as e:
        print(f"Error: {e}")
