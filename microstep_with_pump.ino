#include <ESP32Servo.h>

#define STAND_PIN     32
#define SHOULDER_PIN  33
#define ELBOW_PIN     25
#define PUMP_PIN      2

#define MIN_US  500
#define MAX_US 2500

Servo sServo, aServo, eServo;

int s_us = 1500;
int a_us = 1500;
int e_us = 1500;
bool pump_on = false;

void setup() {
  Serial.begin(115200);

  sServo.attach(STAND_PIN, MIN_US, MAX_US);
  aServo.attach(SHOULDER_PIN, MIN_US, MAX_US);
  eServo.attach(ELBOW_PIN, MIN_US, MAX_US);

  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, HIGH);

  sServo.writeMicroseconds(s_us);
  aServo.writeMicroseconds(a_us);
  eServo.writeMicroseconds(e_us);

  Serial.println("OK. Commands:");
  Serial.println(" s1500   → set Stand to 1500 µs");
  Serial.println(" a+10    → Shoulder += 10 µs");
  Serial.println(" e-5     → Elbow -= 5 µs");
  Serial.println(" p1 / p0 → pump ON / OFF");
}

int clamp(int val, int min_val, int max_val) {
  if (val < min_val) return min_val;
  if (val > max_val) return max_val;
  return val;
}

void loop() {
  static uint32_t lastOrg = 0;
  if (millis() - lastOrg > 200) {
    lastOrg = millis();
    Serial.printf("org: s%d a%d e%d p%d\n", s_us, a_us, e_us, pump_on ? 1 : 0);
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();                                
    
    if (cmd.length() < 2) {
      return;
    }

    char type = cmd[0];
    bool is_relative = (cmd.length() > 1) && (cmd[1] == '+' || cmd[1] == '-');
    bool ok = false;

    if (type == 'p' && cmd.length() == 2) {
      if (cmd[1] == '1') {
        pump_on = true;
        digitalWrite(PUMP_PIN, LOW);
        Serial.println("pump ON");
        ok = true;
      } else if (cmd[1] == '0') {
        pump_on = false;
        digitalWrite(PUMP_PIN, HIGH);
        Serial.println("pump OFF");
        ok = true;
      }
    }
    else if ((type == 's' || type == 'a' || type == 'e') && cmd.length() >= 2) {
      int delta_or_abs = 0;

      if (is_relative) {
        delta_or_abs = cmd.substring(1).toInt();
        if (type == 's') s_us = clamp(s_us + delta_or_abs, MIN_US, MAX_US);
        else if (type == 'a') a_us = clamp(a_us + delta_or_abs, MIN_US, MAX_US);
        else if (type == 'e') e_us = clamp(e_us + delta_or_abs, MIN_US, MAX_US);
      } else {
        delta_or_abs = cmd.substring(1).toInt();
        if (delta_or_abs < 100) {
          Serial.println("ERR: value too small");
          return;
        }
        if (type == 's') s_us = clamp(delta_or_abs, MIN_US, MAX_US);
        else if (type == 'a') a_us = clamp(delta_or_abs, MIN_US, MAX_US);
        else if (type == 'e') e_us = clamp(delta_or_abs, MIN_US, MAX_US);
      }

      if (type == 's') sServo.writeMicroseconds(s_us);
      else if (type == 'a') aServo.writeMicroseconds(a_us);
      else if (type == 'e') eServo.writeMicroseconds(e_us);

      Serial.printf("%c → %d µs\n", type,
        (type == 's') ? s_us :
        (type == 'a') ? a_us : e_us);
      ok = true;
    }

    if (!ok) {
      Serial.println("ERR: invalid cmd");
    }
  }
}
