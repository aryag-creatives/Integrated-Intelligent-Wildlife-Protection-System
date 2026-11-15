
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>
#include <WiFi.h>
#include <WebServer.h>
// Enter your credentials locally. Do not share them publicly.
const char* WIFI_SSID = "WIFIPASS";
const char* WIFI_PASSWORD = "LINGLANGGULIGULIWATCHA";
const char* API_KEY = "SUIIIIIIIIIIIIIIII";

WebServer server(80);
unsigned long lastWiFiAttempt = 0;
bool httpServerStarted = false;
const unsigned long WIFI_RETRY_INTERVAL = 10000;

#define PIR_PIN 4
#define LED_PIN 18
#define SDA_PIN 22
#define SCL_PIN 21
#define ENA_PIN 25
#define IN1_PIN 27
#define IN2_PIN 14
#define HOME_LIMIT_PIN 33
#define FAR_LIMIT_PIN 32

#define HOME_LIMIT_ACTIVE LOW
#define FAR_LIMIT_ACTIVE LOW
#define MOTOR_FORWARD_IN1 HIGH
#define MOTOR_FORWARD_IN2 LOW
#define MOTOR_REVERSE_IN1 LOW
#define MOTOR_REVERSE_IN2 HIGH
#define MOTOR_SPEED 200

const unsigned long MPU_PRINT_INTERVAL = 5000;
const unsigned long MPU_SAMPLE_INTERVAL = 10;
const unsigned long MOTOR_TIMEOUT = 30000;
const unsigned long DIRECTION_CHANGE_DELAY = 300;
const unsigned long GATE_OPEN_DURATION = 15000; // 15 seconds

Adafruit_MPU6050 mpu;
bool mpuAvailable = false;

enum GateState {
HOMING,
IDLE,
MOVING_FAR,
OPEN,
MOVING_HOME,
ERROR_STATE
};
GateState gateState = HOMING;

bool previousPirState = LOW;
bool fenceLedState = false;
unsigned long motorStartTime = 0;
unsigned long directionChangeTime = 0;
unsigned long gateOpenedTime = 0;
unsigned long lastMpuSampleTime = 0;
unsigned long lastMpuPrintTime = 0;
float pitch = 0.0;
bool pitchInitialized = false;
bool closeMotorStarted = false;
String lastError = "";

// ---------------- SENSOR / STATE HELPERS ----------------
bool homeLimitPressed() { return digitalRead(HOME_LIMIT_PIN) == HOME_LIMIT_ACTIVE; }
bool farLimitPressed() { return digitalRead(FAR_LIMIT_PIN) == FAR_LIMIT_ACTIVE; }
bool pirDetected() { return digitalRead(PIR_PIN) == HIGH; }

const char* gateStateName() {
switch (gateState) {
case HOMING: return "HOMING";
case IDLE: return "CLOSED_IDLE";
case MOVING_FAR: return "OPENING";
case OPEN: return "OPEN";
case MOVING_HOME: return "CLOSING";
case ERROR_STATE: return "FAULT";
default: return "UNKNOWN";
}
}

// ---------------- MOTOR CONTROL ----------------
void motorStop() {
digitalWrite(ENA_PIN, LOW);
digitalWrite(IN1_PIN, LOW);
digitalWrite(IN2_PIN, LOW);
}

void motorForward() {
digitalWrite(IN1_PIN, MOTOR_FORWARD_IN1);
digitalWrite(IN2_PIN, MOTOR_FORWARD_IN2);
analogWrite(ENA_PIN, MOTOR_SPEED);
}

void motorReverse() {
digitalWrite(IN1_PIN, MOTOR_REVERSE_IN1);
digitalWrite(IN2_PIN, MOTOR_REVERSE_IN2);
analogWrite(ENA_PIN, MOTOR_SPEED);
}

void setFenceLED(bool on) {
fenceLedState = on;
digitalWrite(LED_PIN, on ? HIGH : LOW);
}

void enterError(const char* message) {
motorStop();
closeMotorStarted = false;
gateState = ERROR_STATE;
lastError = message;
Serial.println();
Serial.println("========== GATE ERROR ==========");
Serial.println(message);
Serial.println("Motor stopped. Reset requires a fresh homing cycle.");
Serial.println("================================");
}

// ---------------- GATE MOVEMENT COMMANDS ----------------
void startHoming() {
motorStop();
closeMotorStarted = false;

if (homeLimitPressed() && farLimitPressed()) {
enterError("Both limit switches are active.");
return;
}

if (homeLimitPressed()) {
gateState = IDLE;
lastError = "";
Serial.println("Home position confirmed. Gate CLOSED_IDLE.");
return;
}

if (farLimitPressed()) {
// On startup at the far limit, do not move or start a timer unexpectedly.
gateState = OPEN;
lastError = "";
Serial.println("Far limit active. Gate state OPEN; waiting for CLOSE command.");
return;
}

Serial.println("Starting homing toward home limit...");
motorReverse();
motorStartTime = millis();
gateState = HOMING;
}

bool startOpening() {
if (gateState != IDLE) return false;

if (homeLimitPressed() && farLimitPressed()) {
enterError("Both limit switches are active.");
return false;
}

if (!homeLimitPressed() || farLimitPressed()) {
enterError("Gate position is inconsistent for OPEN command.");
return false;
}

Serial.println("OPEN command accepted. Opening gate...");
motorForward();
motorStartTime = millis();
gateState = MOVING_FAR;
return true;
}

bool startClosing() {
if (gateState != OPEN) return false;

closeMotorStarted = false;

if (homeLimitPressed() && farLimitPressed()) {
enterError("Both limit switches are active.");
return false;
}

if (!farLimitPressed() || homeLimitPressed()) {
enterError("Gate position is inconsistent for CLOSE command.");
return false;
}

Serial.println("CLOSE command accepted. Closing gate...");
directionChangeTime = millis();
gateState = MOVING_HOME;
motorStop();
motorStartTime = millis();
return true;
}

void stopGate() {
motorStop();
closeMotorStarted = false;

if (gateState == MOVING_FAR || gateState == MOVING_HOME || gateState == HOMING) {
gateState = ERROR_STATE;
lastError = "STOP command received during movement; position requires homing.";
}

Serial.println("STOP command processed. Motor output disabled.");
}

// ---------------- GATE STATE MACHINE ----------------
void gateControl() {
if (homeLimitPressed() && farLimitPressed()) {
enterError("Both limit switches are active simultaneously.");
return;
}

if (gateState == HOMING || gateState == MOVING_FAR || gateState == MOVING_HOME) {
if (millis() - motorStartTime >= MOTOR_TIMEOUT) {
enterError("Motor movement timeout. Check motor and limit switches.");
return;
}
}

switch (gateState) {
case HOMING:
if (homeLimitPressed()) {
motorStop();
gateState = IDLE;
lastError = "";
Serial.println("Homing complete. Gate CLOSED_IDLE.");
}
break;

case IDLE:
// Wait for authorized OPEN command.
break;

case MOVING_FAR:
if (farLimitPressed()) {
motorStop();
gateState = OPEN;
gateOpenedTime = millis();
Serial.println("Far limit reached. Gate is fully OPEN.");
Serial.println("Automatic closing countdown started: 15 seconds.");
}
break;

case OPEN:
if (millis() - gateOpenedTime >= GATE_OPEN_DURATION) {
Serial.println("15 seconds elapsed. Starting automatic closing...");
startClosing();
}
break;

case MOVING_HOME:
if (millis() - directionChangeTime < DIRECTION_CHANGE_DELAY) {
motorStop();
} else {
if (!closeMotorStarted) {
motorReverse();
closeMotorStarted = true;
}

if (homeLimitPressed()) {
motorStop();
closeMotorStarted = false;
gateState = IDLE;
lastError = "";
Serial.println("Home limit reached. Gate CLOSED_IDLE.");
}
}
break;

case ERROR_STATE:
motorStop();
break;
}
}

// ---------------- PIR REPORTING ----------------
void pirControl() {
bool currentPirState = pirDetected();
if (currentPirState && !previousPirState) {
Serial.println("PIR motion detected. Available to Raspberry Pi via /status.");
}
previousPirState = currentPirState;
}

// ---------------- MPU6050 PITCH CALCULATION ----------------
void mpuControl() {
if (!mpuAvailable) return;

unsigned long currentTime = millis();

if (currentTime - lastMpuSampleTime >= MPU_SAMPLE_INTERVAL) {
float dt = (currentTime - lastMpuSampleTime) / 1000.0f;
lastMpuSampleTime = currentTime;

sensors_event_t acceleration, gyro, temperature;
mpu.getEvent(&acceleration, &gyro, &temperature);

float accPitch = atan2(
-acceleration.acceleration.x,
sqrt(acceleration.acceleration.y * acceleration.acceleration.y +
acceleration.acceleration.z * acceleration.acceleration.z)
) * 180.0f / PI;

float gyroPitchRate = gyro.gyro.y * 180.0f / PI;
if (!pitchInitialized) {
pitch = accPitch;
pitchInitialized = true;
} else {
pitch = 0.96f * (pitch + gyroPitchRate * dt) + 0.04f * accPitch;
}
}

if (currentTime - lastMpuPrintTime >= MPU_PRINT_INTERVAL) {
lastMpuPrintTime = currentTime;
if (pitchInitialized) {
Serial.print("MPU6050 PITCH: ");
Serial.print(pitch, 2);
Serial.println(" degrees");
}
}
}

// ---------------- HTTP API ----------------
bool authorizedRequest() {
if (!server.hasArg("key") || server.arg("key") != String(API_KEY)) {
server.send(401, "application/json", "{\"ok\":false,\"error\":\"unauthorized\"}");
return false;
}
return true;
}

void handleStatus() {
String json = "{";
json += "\"device\":\"ESP32\",";
json += "\"status\":\"online\",";
json += "\"ip\":\"" + WiFi.localIP().toString() + "\",";
json += "\"rssi\":" + String(WiFi.RSSI()) + ",";
json += "\"gate_state\":\"" + String(gateStateName()) + "\",";
json += "\"pir\":" + String(pirDetected() ? "true" : "false") + ",";
json += "\"home_limit\":" + String(homeLimitPressed() ? "true" : "false") + ",";
json += "\"far_limit\":" + String(farLimitPressed() ? "true" : "false") + ",";
json += "\"mpu_available\":" + String(mpuAvailable ? "true" : "false") + ",";
json += "\"pitch\":" + String(pitch, 2) + ",";
json += "\"fence_led\":" + String(fenceLedState ? "true" : "false") + ",";
json += "\"error\":\"" + lastError + "\"";
json += "}";
server.send(200, "application/json", json);
}

void handleCommand() {
if (!authorizedRequest()) return;
if (!server.hasArg("cmd")) {
server.send(400, "application/json", "{\"ok\":false,\"error\":\"missing cmd\"}");
return;
}

String cmd = server.arg("cmd");
cmd.toUpperCase();
bool accepted = false;
String message = "command rejected for current gate state";

if (cmd == "OPEN") {
accepted = startOpening();
if (accepted) message = "opening started";
} else if (cmd == "CLOSE") {
accepted = startClosing();
if (accepted) message = "closing started";
} else if (cmd == "STOP") {
stopGate();
accepted = true;
message = "motor stopped; inspect position before reset";
} else if (cmd == "RESET") {
if (gateState == ERROR_STATE) {
startHoming();
accepted = (gateState != ERROR_STATE);
message = accepted ? "homing/recovery started" : "reset failed";
} else {
message = "reset only allowed in FAULT state";
}
} else if (cmd == "LED_ON") {
setFenceLED(true);
accepted = true;
message = "fence indicator LED on";
} else if (cmd == "LED_OFF") {
setFenceLED(false);
accepted = true;
message = "fence indicator LED off";
} else {
server.send(400, "application/json", "{\"ok\":false,\"error\":\"unknown command\"}");
return;
}

String response = "{\"ok\":" + String(accepted ? "true" : "false") +
",\"message\":\"" + message +
"\",\"gate_state\":\"" + gateStateName() + "\"}";
server.send(accepted ? 200 : 409, "application/json", response);
}

void handleNotFound() {
server.send(404, "application/json", "{\"ok\":false,\"error\":\"not found\"}");
}

void beginHttpServer() {
if (httpServerStarted) return;
Serial.println("Starting HTTP server...");
server.begin();
httpServerStarted = true;
Serial.println("HTTP API initialized on port 80.");
Serial.println("ESP32 HTTP server is ready.");
}

void startWiFiAndServer() {
server.on("/status", HTTP_GET, handleStatus);
server.on("/command", HTTP_POST, handleCommand);
server.onNotFound(handleNotFound);

WiFi.mode(WIFI_STA);
delay(100);
WiFi.setSleep(false);
WiFi.setAutoReconnect(true);
WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

Serial.print("Connecting to hotspot...");
unsigned long started = millis();
while (WiFi.status() != WL_CONNECTED && millis() - started < 20000) {
delay(300);
Serial.print(".");
}
Serial.println();

if (WiFi.status() == WL_CONNECTED) {
Serial.println("Wi-Fi connected!");
Serial.print("ESP32 IP: ");
Serial.println(WiFi.localIP());
beginHttpServer();
} else {
Serial.println("Wi-Fi connection timeout. Gate safety logic remains local; Wi-Fi will retry.");
}
lastWiFiAttempt = millis();
}

void maintainWiFi() {
if (WiFi.status() == WL_CONNECTED) {
if (!httpServerStarted) {
Serial.print("Wi-Fi connected. ESP32 IP: ");
Serial.println(WiFi.localIP());
beginHttpServer();
}
if (httpServerStarted) server.handleClient();
return;
}

if (millis() - lastWiFiAttempt >= WIFI_RETRY_INTERVAL) {
lastWiFiAttempt = millis();
Serial.println("Wi-Fi disconnected. Attempting reconnect...");
WiFi.reconnect();
}
}

// ---------------- SERIAL BENCH-TEST COMMANDS ----------------
// g=open, c=close, s=stop, r=reset, l=LED on, o=LED off
void serialControl() {
if (!Serial.available()) return;
char command = Serial.read();
if (command == '\n' || command == '\r') return;
if (command >= 'A' && command <= 'Z') command += 32;

switch (command) {
case 'g': startOpening(); break;
case 'c': startClosing(); break;
case 's': stopGate(); break;
case 'r':
if (gateState == ERROR_STATE) startHoming();
else Serial.println("Reset ignored: gate is not in FAULT state.");
break;
case 'l': setFenceLED(true); Serial.println("Fence LED ON"); break;
case 'o': setFenceLED(false); Serial.println("Fence LED OFF"); break;
default:
Serial.println("Use g=open, c=close, s=stop, r=reset, l=LED on, o=LED off");
break;
}
}

// ---------------- SETUP ----------------
void setup() {
Serial.begin(115200);
delay(800);

Serial.println();
Serial.println("========================================");
Serial.println("WILDLIFE PROTECTION - ESP32 CONTROLLER");
Serial.println("Wi-Fi HTTP integration build");
Serial.println("Automatic gate closing: 15 seconds");
Serial.println("========================================");

pinMode(PIR_PIN, INPUT);
pinMode(LED_PIN, OUTPUT);
setFenceLED(false);

pinMode(ENA_PIN, OUTPUT);
pinMode(IN1_PIN, OUTPUT);
pinMode(IN2_PIN, OUTPUT);
pinMode(HOME_LIMIT_PIN, INPUT_PULLUP);
pinMode(FAR_LIMIT_PIN, INPUT_PULLUP);
motorStop();

Wire.begin(SDA_PIN, SCL_PIN);
Wire.setClock(100000);
delay(50);

Serial.println("Initializing MPU6050...");
if (mpu.begin()) {
mpuAvailable = true;
Serial.println("MPU6050 connected successfully.");
mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
mpu.setGyroRange(MPU6050_RANGE_500_DEG);
mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
lastMpuSampleTime = millis();
lastMpuPrintTime = millis();
} else {
mpuAvailable = false;
Serial.println("WARNING: MPU6050 not detected.");
}

Serial.print("Home switch: ");
Serial.println(homeLimitPressed() ? "ACTIVE" : "INACTIVE");
Serial.print("Far switch: ");
Serial.println(farLimitPressed() ? "ACTIVE" : "INACTIVE");

if (homeLimitPressed() && farLimitPressed()) {
enterError("Both limit switches are active during startup.");
} else {
startHoming();
}

Serial.println("Hardware initialization complete.");
startWiFiAndServer();
Serial.println("Serial commands: g=open, c=close, s=stop, r=reset, l=LED on, o=LED off");
}

// ---------------- MAIN LOOP ----------------
void loop() {
maintainWiFi();
pirControl();
mpuControl();
serialControl();
gateControl();
}

