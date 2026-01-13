#include "esp_camera.h"
#include "esp_http_server.h"
#include <WiFi.h>

// _____________ CAMERA MODEL: AI_THINKER _____________
// Camera pin definitions for AI THINKER ESP32CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ----------------AP CONFIG --------
const char* ap_ssid = "ESP32_CAM";
const char* ap_password = "12345678"; //  Use this pass to connect with edp32

// HTTP server
httpd_handle_t camera_httpd = NULL;
httpd_handle_t stream_httpd = NULL;

// ______------MJPEG STREAM HANDLER -------______
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char * part_buf[64];

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace; boundary=frame");
  if (res != ESP_OK) {
    return res;
  }

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    if (fb->format != PIXFORMAT_JPEG) {
      Serial.println("Non-JPEG frame received");
      esp_camera_fb_return(fb);
      res = ESP_FAIL;
      break;
    }

    _jpg_buf_len = fb->len;
    _jpg_buf = fb->buf;

    size_t hlen = snprintf((char *)part_buf, 64, 
                          "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", 
                          _jpg_buf_len);
    res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, "\r\n", 2);
    }

    esp_camera_fb_return(fb);
    
    if (res != ESP_OK) {
      break;
    }
  }
  return res;
}

// __________________START CAMERA SERVER ___________________
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  Serial.printf("Starting stream server on port: '%d'\n", config.server_port);
  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
    Serial.println("Stream server started successfully");
    Serial.println("Stream URL: http://192.168.4.1/stream");
  } else {
    Serial.println("Failed to start stream server");
  }
}

// __________________ SETUP _______________
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\n=== ESP32-CAM AP Mode Stream ===");

  // Camera configuration
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  
  // Frame size and quality settings
  config.frame_size   = FRAMESIZE_VGA;    // 640x480 (quality  controll)
  config.jpeg_quality = 10;               // 0-63, lower means higher quality
  config.fb_count     = 2;                // Use 2 frame buffers for smoother streaming
  config.grab_mode    = CAMERA_GRAB_LATEST; // get latest frame

  // Initialize camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    delay(1000);
    ESP.restart();
  }
  Serial.println("Camera initialized successfully");

  // Adjust camera sensor settings for better image quality
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 0);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_special_effect(s, 0); // 0 to 6 (0 - No Effect)
    s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
    s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
    s->set_wb_mode(s, 0);        // 0 to 4
    s->set_exposure_ctrl(s, 1);  // 0 = disable , 1 = enable
    s->set_aec2(s, 0);           // 0 = disable , 1 = enable
    s->set_gain_ctrl(s, 1);      // 0 = disable , 1 = enable
    s->set_agc_gain(s, 0);       // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)0);  // 0 to 6
    s->set_bpc(s, 0);            // 0 = disable , 1 = enable
    s->set_wpc(s, 1);            // 0 = disable , 1 = enable
    s->set_raw_gma(s, 1);        // 0 = disable , 1 = enable
    s->set_lenc(s, 1);           // 0 = disable , 1 = enable
    s->set_hmirror(s, 0);        // 0 = disable , 1 = enable
    s->set_vflip(s, 1);          // 0 = disable , 1 = enable
    s->set_dcw(s, 1);            // 0 = disable , 1 = enable
    s->set_colorbar(s, 0);       // 0 = disable , 1 = enable
  }

  // Start WiFi in AP mode
  WiFi.mode(WIFI_AP);
  bool ap_started = WiFi.softAP(ap_ssid, ap_password);
  
  if (ap_started) {
    Serial.println("\n✓ ESP32-CAM AP Started Successfully");
    Serial.printf("SSID: %s\n", ap_ssid);
    Serial.printf("Password: %s\n", ap_password);
    Serial.print("IP Address: ");
    Serial.println(WiFi.softAPIP());
    Serial.println("\nStream URL: http://192.168.4.1/stream");
    Serial.println("=====================================\n");
  } else {
    Serial.println(" x Failed to start AP");
  }

  // Start camera streaming server
  startCameraServer();
  
  Serial.println("Setup complete. Ready to stream!");
}

// _____________LOOP _______________
void loop() {
  delay(1000);
  
  // Print connected clients count
  static unsigned long last_print = 0;
  if (millis() - last_print > 10000) { // Every 10 seconds
    Serial.printf("Connected clients: %d\n", WiFi.softAPgetStationNum());
    last_print = millis();
  }
}
