#include <WiFi.h>
#include <WebSocketsClient.h>

const char* ssid = "V2065";
const char* password = "12345678";

const char* serverIP = "MAC_IP";

WebSocketsClient webSocket;


void webSocketEvent(
  WStype_t type,
  uint8_t * payload,
  size_t length
) {

  switch (type) {

    case WStype_DISCONNECTED:
      Serial.println("WebSocket disconnected");
      break;

    case WStype_CONNECTED:
      Serial.println("WebSocket connected!");

      webSocket.sendTXT("HELLO FROM ESP32");
      break;

    case WStype_TEXT:
      Serial.print("Server says: ");
      Serial.println((char*)payload);
      break;

    case WStype_ERROR:
      Serial.println("WebSocket error");
      break;

    default:
      break;
  }
}


void setup() {

  Serial.begin(115200);

  WiFi.begin(ssid, password);

  Serial.print("Connecting to Wi-Fi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected");

  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());

  webSocket.begin(serverIP, 8000, "/ws");

  webSocket.onEvent(webSocketEvent);

  webSocket.setReconnectInterval(5000);
}


void loop() {

  webSocket.loop();

}