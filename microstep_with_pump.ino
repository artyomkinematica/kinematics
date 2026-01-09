// #include <ESP32Servo.h>

// // Пины сервоприводов
// #define STAND_PIN     32      
// #define SHOULDER_PIN  33      
// #define ELBOW_PIN     25      

// // Пин помпы
// #define PUMP_PIN      2

// // Границы углов
// #define STAND_MIN_ANGLE   0
// #define STAND_MAX_ANGLE   180

// #define SHOULDER_MIN_ANGLE 0
// #define SHOULDER_MAX_ANGLE 180

// #define ELBOW_MIN_ANGLE    0
// #define ELBOW_MAX_ANGLE    180

// // Границы ШИМ сигнала
// #define MIN_PULSE_WIDTH 500
// #define MAX_PULSE_WIDTH 2400

// Servo servoStand;
// Servo servoArm;
// Servo servoElbow;

// // Текущие значения
// int standValue = 90;
// int shoulderValue = 90;
// int elbowValue = 90;

// bool isAngleMode = true; // true - углы, false - микросекунды

// // Состояние помпы
// bool pumpState = HIGH; // HIGH = выключена (помпа активна при LOW)

// void setup() {
//   Serial.begin(115200);
  
//   // Инициализация сервоприводов
//   servoStand.attach(STAND_PIN, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
//   servoArm.attach(SHOULDER_PIN, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
//   servoElbow.attach(ELBOW_PIN, MIN_PULSE_WIDTH, MAX_PULSE_WIDTH);
  
//   // Инициализация помпы
//   pinMode(PUMP_PIN, OUTPUT);
//   digitalWrite(PUMP_PIN, pumpState);
  
//   // Установка в начальное положение
//   servoStand.write(standValue);
//   servoArm.write(shoulderValue);
//   servoElbow.write(elbowValue);
  
//   Serial.println("=== Servo Control Ready ===");
//   Serial.println("Format: move [stand|shoulder|elbow] [value]");
//   Serial.println("Values: 0-180 (angle) or 500-2400 (pulse width)");
//   Serial.println("Pump commands: pump up / pump down / pump out");
//   Serial.println("Current mode: ANGLE (0-180)");
//   Serial.println();
// }

// void loop() {
//   if (Serial.available()) {
//     String command = Serial.readStringUntil('\n');
//     command.trim();
//     processCommand(command);
//   }
// }

// void processCommand(String cmd) {
//   cmd.toLowerCase();
  
//   // Обработка команд помпы
//   if (cmd == "pump up") {
//     pumpState = LOW;
//     digitalWrite(PUMP_PIN, pumpState);
//     Serial.println("Pump turned ON (LOW)");
//     return;
//   }
//   else if (cmd == "pump down" || cmd == "pump out") {
//     pumpState = HIGH;
//     digitalWrite(PUMP_PIN, pumpState);
//     Serial.println("Pump turned OFF (HIGH)");
//     return;
//   }
  
//   // Обработка команд серво
//   if (cmd.startsWith("move ")) {
//     int firstSpace = cmd.indexOf(' ');
//     int secondSpace = cmd.indexOf(' ', firstSpace + 1);
    
//     if (firstSpace == -1 || secondSpace == -1) {
//       Serial.println("ERROR: Invalid format. Use: move [servo] [value]");
//       return;
//     }
    
//     String servoName = cmd.substring(firstSpace + 1, secondSpace);
//     String valueStr = cmd.substring(secondSpace + 1);
//     int value = valueStr.toInt();
    
//     // Автоматическое определение режима по значению
//     bool newMode = (value >= 0 && value <= 180);
    
//     if (isAngleMode != newMode) {
//       isAngleMode = newMode;
//       Serial.print("Mode changed to: ");
//       Serial.println(isAngleMode ? "ANGLE (0-180)" : "PULSE WIDTH (500-2400)");
//     }
    
//     // Обработка команды
//     if (servoName == "stand") {
//       moveServo(servoStand, standValue, value, "Stand");
//     } 
//     else if (servoName == "shoulder") {
//       moveServo(servoArm, shoulderValue, value, "Shoulder");
//     } 
//     else if (servoName == "elbow") {
//       moveServo(servoElbow, elbowValue, value, "Elbow");
//     }
//     else {
//       Serial.println("ERROR: Unknown servo. Use: stand, shoulder, or elbow");
//     }
//   }
//   else if (cmd == "status") {
//     printStatus();
//   }
//   else if (cmd == "help") {
//     printHelp();
//   }
//   else {
//     Serial.println("ERROR: Unknown command. Type 'help' for instructions.");
//   }
// }

// void moveServo(Servo &servo, int &currentValue, int targetValue, const char* servoName) {
//   // Проверка границ в зависимости от режима
//   if (isAngleMode) {
//     if (targetValue < 0 || targetValue > 180) {
//       Serial.print("ERROR: Angle must be 0-180. Got: ");
//       Serial.println(targetValue);
//       return;
//     }
//   } else {
//     if (targetValue < MIN_PULSE_WIDTH || targetValue > MAX_PULSE_WIDTH) {
//       Serial.print("ERROR: Pulse width must be ");
//       Serial.print(MIN_PULSE_WIDTH);
//       Serial.print("-");
//       Serial.print(MAX_PULSE_WIDTH);
//       Serial.print(". Got: ");
//       Serial.println(targetValue);
//       return;
//     }
//   }
  
//   // Выполнение движения
//   if (isAngleMode) {
//     servo.write(targetValue);
//     Serial.print(servoName);
//     Serial.print(" moved to angle: ");
//     Serial.println(targetValue);
//   } else {
//     servo.writeMicroseconds(targetValue);
//     Serial.print(servoName);
//     Serial.print(" set to pulse width: ");
//     Serial.print(targetValue);
//     Serial.println(" μs");
//   }
  
//   currentValue = targetValue;
// }

// void printStatus() {
//   Serial.println("=== Current Status ===");
//   Serial.print("Mode: ");
//   Serial.println(isAngleMode ? "ANGLE (0-180)" : "PULSE WIDTH (500-2400)");
  
//   if (isAngleMode) {
//     Serial.print("Stand: ");
//     Serial.print(standValue);
//     Serial.println("°");
    
//     Serial.print("Shoulder: ");
//     Serial.print(shoulderValue);
//     Serial.println("°");
    
//     Serial.print("Elbow: ");
//     Serial.print(elbowValue);
//     Serial.println("°");
//   } else {
//     Serial.print("Stand: ");
//     Serial.print(standValue);
//     Serial.println(" μs");
    
//     Serial.print("Shoulder: ");
//     Serial.print(shoulderValue);
//     Serial.println(" μs");
    
//     Serial.print("Elbow: ");
//     Serial.print(elbowValue);
//     Serial.println(" μs");
//   }
  
//   // Вывод состояния помпы
//   Serial.print("Pump: ");
//   Serial.println(pumpState ? "OFF (HIGH)" : "ON (LOW)");
// }

// void printHelp() {
//   Serial.println("=== Servo Control Commands ===");
//   Serial.println("move [servo] [value] - Move servo to position");
//   Serial.println("  servos: stand, shoulder, elbow");
//   Serial.println("  values: 0-180 (angle) or 500-2400 (pulse width)");
//   Serial.println();
//   Serial.println("Pump commands:");
//   Serial.println("  pump up        - Turn pump ON");
//   Serial.println("  pump out       - Turn pump OFF");
//   Serial.println();
//   Serial.println("status - Show current positions and pump state");
//   Serial.println("help - Show this help");
//   Serial.println();
//   Serial.println("Examples:");
//   Serial.println("  move stand 90     - Stand to 90 degrees");
//   Serial.println("  move shoulder 45  - Shoulder to 45 degrees");
//   Serial.println("  move elbow 1500   - Elbow to 1500 μs pulse");
//   Serial.println("  pump up           - Activate pump");
// }

#include <ESP32Servo.h>

// Пины
#define STAND_PIN     32
#define SHOULDER_PIN  33
#define ELBOW_PIN     25
#define PUMP_PIN      2

// Безопасные пределы для PDI-HV2006MG
#define MIN_US  500
#define MAX_US 2500

Servo sServo, aServo, eServo;

// Текущие позиции в микросекундах
int s_us = 1500;  // начальное — нейтраль
int a_us = 1500;
int e_us = 1500;
bool pump_on = false;

void setup() {
  Serial.begin(115200);

  // Инициализация серв с вашим диапазоном
  sServo.attach(STAND_PIN, MIN_US, MAX_US);
  aServo.attach(SHOULDER_PIN, MIN_US, MAX_US);
  eServo.attach(ELBOW_PIN, MIN_US, MAX_US);

  pinMode(PUMP_PIN, OUTPUT);
  digitalWrite(PUMP_PIN, HIGH); // помпа выключена

  // Установка начального положения
  sServo.writeMicroseconds(s_us);
  aServo.writeMicroseconds(a_us);
  eServo.writeMicroseconds(e_us);

  Serial.println("OK. Commands:");
  Serial.println(" s1500   → set Stand to 1500 µs");
  Serial.println(" a+10    → Shoulder += 10 µs");
  Serial.println(" e-5     → Elbow -= 5 µs");
  Serial.println(" p1 / p0 → pump ON / OFF");
}

// Ограничение значения в диапазоне
int clamp(int val, int min_val, int max_val) {
  if (val < min_val) return min_val;
  if (val > max_val) return max_val;
  return val;
}

void loop() {
  // Постоянный вывод org
  static uint32_t lastOrg = 0;
  if (millis() - lastOrg > 200) {
    lastOrg = millis();
    Serial.printf("org: s%d a%d e%d p%d\n", s_us, a_us, e_us, pump_on ? 1 : 0);
  }

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');  // ← чтение
    cmd.trim();                                 // ← отдельный trim()
    
    if (cmd.length() < 2) {
      return;
    }

    char type = cmd[0];
    // Проверяем, что строка достаточно длинная, прежде чем смотреть cmd[1]
    bool is_relative = (cmd.length() > 1) && (cmd[1] == '+' || cmd[1] == '-');
    bool ok = false;

    // Команды помпы
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
    // Команды серво
    else if ((type == 's' || type == 'a' || type == 'e') && cmd.length() >= 2) {
      int delta_or_abs = 0;

      if (is_relative) {
        // Относительное смещение: a+15, e-10
        delta_or_abs = cmd.substring(1).toInt();
        if (type == 's') s_us = clamp(s_us + delta_or_abs, MIN_US, MAX_US);
        else if (type == 'a') a_us = clamp(a_us + delta_or_abs, MIN_US, MAX_US);
        else if (type == 'e') e_us = clamp(e_us + delta_or_abs, MIN_US, MAX_US);
      } else {
        // Абсолютное значение: s1500
        delta_or_abs = cmd.substring(1).toInt();
        if (delta_or_abs < 100) {
          Serial.println("ERR: value too small");
          return;
        }
        if (type == 's') s_us = clamp(delta_or_abs, MIN_US, MAX_US);
        else if (type == 'a') a_us = clamp(delta_or_abs, MIN_US, MAX_US);
        else if (type == 'e') e_us = clamp(delta_or_abs, MIN_US, MAX_US);
      }

      // Применяем
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