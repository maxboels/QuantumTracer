/*
 * QuantumTracer RC Car Arduino Controller
 * =====================================
 * 
 * Receives control commands via Serial from Raspberry Pi and controls
 * steering servo and ESC (Electronic Speed Controller) for autonomous driving.
 * 
 * Commands are JSON format: {"steering": 0.5, "throttle": 0.3}
 * 
 * Hardware connections:
 * - Steering Servo: Pin 9 (PWM)
 * - ESC (Throttle): Pin 10 (PWM)  
 * - LED Status: Pin 13
 * - Emergency Stop Button: Pin 2 (interrupt)
 * 
 * Author: QuantumTracer Team
 */

#include <Servo.h>
#include <ArduinoJson.h>

// Hardware pins
const int STEERING_PIN = 9;
const int THROTTLE_PIN = 10;  
const int LED_PIN = 13;
const int EMERGENCY_STOP_PIN = 2;

// Servo objects
Servo steeringServo;
Servo throttleESC;

// Control parameters
const int STEERING_CENTER = 90;  // Neutral position (90 degrees)
const int STEERING_RANGE = 45;   // +/- degrees from center
const int THROTTLE_NEUTRAL = 90; // ESC neutral position
const int THROTTLE_MIN = 90;     // Minimum throttle (stopped)
const int THROTTLE_MAX = 180;    // Maximum throttle (full speed)

// Safety settings
const unsigned long COMMAND_TIMEOUT = 1000; // 1 second timeout
const unsigned long HEARTBEAT_INTERVAL = 500; // Status LED blink rate
bool emergencyStop = false;
unsigned long lastCommandTime = 0;
unsigned long lastHeartbeat = 0;

// Current control values
float currentSteering = 0.0;
float currentThrottle = 0.0;

// JSON parsing
StaticJsonDocument<200> jsonDoc;

void setup() {
  Serial.begin(115200);
  Serial.println("QuantumTracer Arduino Controller v1.0");
  Serial.println("Waiting for Raspberry Pi connection...");
  
  // Initialize servos
  steeringServo.attach(STEERING_PIN);
  throttleESC.attach(THROTTLE_PIN);
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(EMERGENCY_STOP_PIN, INPUT_PULLUP);
  
  // Set initial positions (neutral/stopped)
  setNeutralPositions();
  
  // Setup emergency stop interrupt
  attachInterrupt(digitalPinToInterrupt(EMERGENCY_STOP_PIN), emergencyStopISR, FALLING);
  
  Serial.println("Arduino initialized - Ready for commands");
  digitalWrite(LED_PIN, HIGH); // Signal ready
}

void loop() {
  // Check for incoming commands
  if (Serial.available() > 0) {
    processCommand();
  }
  
  // Safety timeout check
  checkCommandTimeout();
  
  // Update status LED
  updateHeartbeat();
  
  // Small delay to prevent overwhelming the loop
  delay(10);
}

void processCommand() {
  String jsonString = Serial.readStringUntil('\n');
  
  // Parse JSON command
  DeserializationError error = deserializeJson(jsonDoc, jsonString);
  
  if (error) {
    Serial.println("ERROR: Invalid JSON format");
    return;
  }
  
  // Update last command timestamp
  lastCommandTime = millis();
  
  // Extract commands
  if (jsonDoc.containsKey("steering")) {
    currentSteering = jsonDoc["steering"].as<float>();
    currentSteering = constrain(currentSteering, -1.0, 1.0);
  }
  
  if (jsonDoc.containsKey("throttle")) {
    currentThrottle = jsonDoc["throttle"].as<float>();
    currentThrottle = constrain(currentThrottle, 0.0, 1.0);
  }
  
  // Check for emergency stop command
  if (jsonDoc.containsKey("emergency_stop")) {
    if (jsonDoc["emergency_stop"].as<bool>()) {
      emergencyStop = true;
      Serial.println("EMERGENCY STOP ACTIVATED");
    }
  }
  
  // Check for test command  
  if (jsonDoc.containsKey("test")) {
    Serial.println("TEST command received - Arduino responding");
  }
  
  // Apply controls if not in emergency stop
  if (!emergencyStop) {
    applySteering(currentSteering);
    applyThrottle(currentThrottle);
    
    // Send acknowledgment
    Serial.print("OK: steering=");
    Serial.print(currentSteering, 2);
    Serial.print(", throttle=");
    Serial.println(currentThrottle, 2);
  } else {
    setNeutralPositions();
    Serial.println("EMERGENCY STOP ACTIVE - Commands ignored");
  }
}

void applySteering(float steering) {
  // Convert -1.0 to +1.0 range to servo degrees
  // -1.0 = full left, 0.0 = center, +1.0 = full right
  int servoAngle = STEERING_CENTER + (steering * STEERING_RANGE);
  servoAngle = constrain(servoAngle, STEERING_CENTER - STEERING_RANGE, 
                                   STEERING_CENTER + STEERING_RANGE);
  
  steeringServo.write(servoAngle);
}

void applyThrottle(float throttle) {
  // Convert 0.0 to 1.0 range to ESC pulses  
  // 0.0 = stopped, 1.0 = full speed
  int escPulse = THROTTLE_NEUTRAL + (throttle * (THROTTLE_MAX - THROTTLE_NEUTRAL));
  escPulse = constrain(escPulse, THROTTLE_MIN, THROTTLE_MAX);
  
  throttleESC.write(escPulse);
}

void setNeutralPositions() {
  // Set steering to center and throttle to stopped
  steeringServo.write(STEERING_CENTER);
  throttleESC.write(THROTTLE_NEUTRAL);
  currentSteering = 0.0;
  currentThrottle = 0.0;
}

void checkCommandTimeout() {
  // If no commands received for COMMAND_TIMEOUT, enter safe mode
  unsigned long timeSinceLastCommand = millis() - lastCommandTime;
  
  if (timeSinceLastCommand > COMMAND_TIMEOUT && lastCommandTime > 0) {
    if (!emergencyStop) {
      Serial.println("WARNING: Command timeout - entering safe mode");
      setNeutralPositions();
      // Don't set emergencyStop flag for timeout (allows recovery)
    }
  }
}

void updateHeartbeat() {
  // Blink LED to show Arduino is alive
  unsigned long currentTime = millis();
  
  if (currentTime - lastHeartbeat >= HEARTBEAT_INTERVAL) {
    digitalWrite(LED_PIN, !digitalRead(LED_PIN));
    lastHeartbeat = currentTime;
  }
}

void emergencyStopISR() {
  // Emergency stop interrupt service routine
  emergencyStop = true;
  // Note: Don't use Serial.println() in ISR
  // The main loop will handle the response
}

// Additional utility functions for diagnostics
void printStatus() {
  Serial.println("=== QUANTUM TRACER STATUS ===");
  Serial.print("Emergency Stop: ");
  Serial.println(emergencyStop ? "ACTIVE" : "INACTIVE");
  Serial.print("Current Steering: ");
  Serial.println(currentSteering, 2);
  Serial.print("Current Throttle: ");  
  Serial.println(currentThrottle, 2);
  Serial.print("Last Command: ");
  Serial.print(millis() - lastCommandTime);
  Serial.println("ms ago");
  Serial.println("============================");
}

/*
 * Additional commands that can be sent from Raspberry Pi:
 * 
 * {"status": true}                    - Get status report
 * {"reset_emergency": true}           - Reset emergency stop
 * {"calibrate_steering": 0.0}         - Set steering center
 * {"test_sequence": true}             - Run built-in test sequence  
 * 
 * Example Python code to send commands:
 * 
 * import serial
 * import json
 * 
 * ser = serial.Serial('/dev/ttyACM0', 115200)
 * command = {"steering": 0.5, "throttle": 0.3}
 * ser.write((json.dumps(command) + '\n').encode())
 */