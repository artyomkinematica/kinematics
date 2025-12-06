import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time

# === Твои настройки (без изменений) ===
BAUD_RATE = 115200
STEP_US = 10
STEP_FORWARD = 20  # для S и ↑
STEP_BACK = 18     # для W и ↓

current_us = {"s": 1500, "a": 1500, "e": 1500}
MIN_US = 500
MAX_US = 2400

ser = None

def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "CP210" in port.description or "ESP32" in port.description or "USB" in port.description:
            return port.device
    return None

def connect_serial():
    global ser
    try:
        port = find_esp32_port()
        if not port:
            raise Exception("ESP32 не найден")
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        print(f"Подключено к {port}")
        return True
    except Exception as e:
        print("Serial error:", e)
        return False

def send_command(cmd: str):
    if ser and ser.is_open:
        try:
            ser.write((cmd + "\n").encode('utf-8'))
            print(f"→ {cmd}")
        except Exception as e:
            print("Ошибка отправки:", e)

def merge_rectangles(rects, max_distance=20):
    if not rects:
        return []
    merged = []
    rects = sorted(rects, key=lambda r: r[0])
    while len(rects) > 0:
        current = rects.pop(0)
        x1, y1, w1, h1 = current
        merged_current = [x1, y1, w1, h1]
        i = 0
        while i < len(rects):
            x2, y2, w2, h2 = rects[i]
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            if overlap_x > 0 or overlap_y > 0 or \
               (abs(x1 - x2) < max_distance and abs(y1 - y2) < max_distance):
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                merged_current = [new_x, new_y, new_w, new_h]
                rects.pop(i)
                i = 0
                x1, y1, w1, h1 = merged_current
            else:
                i += 1
        merged.append(tuple(merged_current))
    return merged

# === Параметры ===
X_TARGET = 150
X_TOL = 8
Y_TOL = 20
MOVE_DELAY = 0.5
MIN_AREA = 800
MAX_DIST = 30

# === Подключение ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Камера не подключена")
    exit(1)

if not connect_serial():
    print("ESP32 не подключён")
    exit(1)

print("Управление запущено. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    Y_TARGET = h // 2

    # Линии цели
    cv2.line(frame, (X_TARGET, 0), (X_TARGET, h), (0, 0, 255), 2)
    cv2.line(frame, (0, Y_TARGET), (w, Y_TARGET), (255, 0, 0), 2)

    roi = frame[:, X_TARGET + 1:] if X_TARGET + 1 < w else frame
    if roi.size == 0:
        continue

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > MIN_AREA]
    merged = merge_rectangles(rects, MAX_DIST)

    if not merged:
        cv2.imshow("Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    x, y, w_obj, h_obj = merged[0]
    center_x = x + X_TARGET + 1 + w_obj // 2
    center_y = y + h_obj // 2

    cv2.rectangle(frame, (x + X_TARGET + 1, y), (x + X_TARGET + 1 + w_obj, y + h_obj), (0, 255, 0), 2)
    cv2.putText(frame, f"X:{center_x} Y:{center_y}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Вычисляем ошибки
    err_x = center_x - X_TARGET      # >0 → справа
    err_y = center_y - Y_TARGET      # >0 → ниже центра

    in_x_zone = abs(err_x) <= X_TOL
    in_y_zone = abs(err_y) <= Y_TOL

    if in_x_zone and in_y_zone:
        print("✅ Цель достигнута!")
        # Можно остановиться или продолжать
        cv2.imshow("Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # === Выбор команды по квадранту ===
    command_sent = False

    if err_x > X_TOL and err_y > Y_TOL:
        # Справа и ниже → нужно ↖ (влево+вверх) → ↓
        current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)  # ↓: плечо вниз → влево+вверх
        send_command(f"a{current_us['a']}")
        print("→ Команда: ↓ (влево+вверх)")
        command_sent = True

    elif err_x > X_TOL and err_y < -Y_TOL:
        # Справа и выше → нужно ↙ (влево+вниз) → W
        current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)  # W: локоть вверх → влево+вниз
        send_command(f"e{current_us['e']}")
        print("→ Команда: W (влево+вниз)")
        command_sent = True

    elif err_x < -X_TOL and err_y > Y_TOL:
        # Слева и ниже → нужно ↗ (вправо+вверх) → S
        current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)  # S: локоть вниз → вправо+вверх
        send_command(f"e{current_us['e']}")
        print("→ Команда: S (вправо+вверх)")
        command_sent = True

    elif err_x < -X_TOL and err_y < -Y_TOL:
        # Слева и выше → нужно ↘ (вправо+вниз) → ↑
        current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)  # ↑: плечо вверх → вправо+вниз
        send_command(f"a{current_us['a']}")
        print("→ Команда: ↑ (вправо+вниз)")
        command_sent = True

    # === Если только по одной оси ошибка ===
    elif err_x > X_TOL:
        # Только справа → выбираем между W и ↓
        # Предпочтение — менее вредная по Y
        if err_y >= 0:
            # Y ниже или в зоне → лучше ↓ (влево+вверх)
            current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
            send_command(f"a{current_us['a']}")
            print("→ Команда: ↓ (только X, Y ниже)")
        else:
            # Y выше → лучше W (влево+вниз)
            current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
            send_command(f"e{current_us['e']}")
            print("→ Команда: W (только X, Y выше)")
        command_sent = True

    elif err_x < -X_TOL:
        # Только слева → выбираем между S и ↑
        if err_y >= 0:
            # Y ниже → ↑ (вправо+вниз)
            current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
            send_command(f"a{current_us['a']}")
            print("→ Команда: ↑ (только X, Y ниже)")
        else:
            # Y выше → S (вправо+вверх)
            current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
            send_command(f"e{current_us['e']}")
            print("→ Команда: S (только X, Y выше)")
        command_sent = True

    elif err_y > Y_TOL:
        # Только ниже → нужно вверх → ↓ или S
        # ↓ даёт влево+вверх, S — вправо+вверх
        # Выбираем то, что меньше нарушает X
        if err_x >= 0:
            # X справа → ↓ (влево+вверх) — приблизит к X=100
            current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
            send_command(f"a{current_us['a']}")
            print("→ Команда: ↓ (только Y, X справа)")
        else:
            # X слева → S (вправо+вверх)
            current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
            send_command(f"e{current_us['e']}")
            print("→ Команда: S (только Y, X слева)")
        command_sent = True

    elif err_y < -Y_TOL:
        # Только выше → нужно вниз → W или ↑
        if err_x >= 0:
            # X справа → W (влево+вниз)
            current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
            send_command(f"e{current_us['e']}")
            print("→ Команда: W (только Y, X справа)")
        else:
            # X слева → ↑ (вправо+вниз)
            current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
            send_command(f"a{current_us['a']}")
            print("→ Команда: ↑ (только Y, X слева)")
        command_sent = True

    if command_sent:
        time.sleep(MOVE_DELAY)

    cv2.imshow("Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()