import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time

# === Настройки ===
BAUD_RATE = 115200
STEP_FORWARD = 20  # для S и ↑
STEP_BACK = 18     # для W и ↓
STEP_S = 20        # шаг для оси "s" (вторая камера)

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

def process_frame_with_lines(frame, x_thresh, y_target_horiz=None, y_top=None, y_bottom=None):
    """
    Рисует линии и возвращает frame + ROI-маску.
    Для второй камеры будем использовать только x_thresh и y_target_horiz.
    """
    h, w = frame.shape[:2]

    # Вертикальная линия (граница ROI)
    x_line = min(x_thresh, w - 1)
    cv2.line(frame, (x_line, 0), (x_line, h), (0, 0, 255), 2)

    # Горизонтальная целевая линия (на всю ширину)
    if y_target_horiz is not None:
        y_clamped = min(max(y_target_horiz, 0), h - 1)
        cv2.line(frame, (0, y_clamped), (w, y_clamped), (255, 0, 0), 2)  # синяя

    # ROI справа от вертикальной линии
    if x_thresh + 1 >= w:
        return frame, np.zeros((h, w), dtype=np.uint8)

    roi = frame[:, x_thresh + 1:]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[:, x_thresh + 1:] = mask
    return frame, mask

# === Подключение камер ===
cap1 = cv2.VideoCapture(0)  # Камера 1 — позиционирование
cap2 = cv2.VideoCapture(2)  # Камера 2 — выравнивание по Y

if not cap1.isOpened():
    print("Камера 1 не подключена")
    exit()
if not cap2.isOpened():
    print("Камера 2 не подключена")
    exit()

# === Подключение ESP32 ===
if not connect_serial():
    print("ESP32 не подключён")
    exit()

# === Параметры ===
# Камера 1
X_TARGET_1 = 150
Y_TOL_1 = 20
X_TOL_1 = 8
MIN_AREA_1 = 800
MAX_DIST_1 = 30

# Камера 2
X_THRESH_2 = 260
Y_TARGET_2 = 260
Y_TOL_2 = 15
MIN_AREA_2 = 800
MAX_DIST_2 = 30

MOVE_DELAY = 0.5

print("Управление запущено. Нажмите 'q' для выхода.")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        print("Ошибка чтения с камер")
        break

    h1, w1 = frame1.shape[:2]
    Y_TARGET_1 = h1 // 2

    # === Обработка камеры 1 (позиционирование) ===
    frame1_vis = frame1.copy()
    # Рисуем линии
    cv2.line(frame1_vis, (X_TARGET_1, 0), (X_TARGET_1, h1), (0, 0, 255), 2)
    cv2.line(frame1_vis, (0, Y_TARGET_1), (w1, Y_TARGET_1), (255, 0, 0), 2)

    roi1 = frame1[:, X_TARGET_1 + 1:] if X_TARGET_1 + 1 < w1 else frame1
    if roi1.size > 0:
        gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        _, mask1 = cv2.threshold(gray1, 30, 255, cv2.THRESH_BINARY_INV)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((15, 15)))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((7, 7)))
        contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects1 = [cv2.boundingRect(c) for c in contours1 if cv2.contourArea(c) > MIN_AREA_1]
        merged1 = merge_rectangles(rects1, MAX_DIST_1)

        if merged1:
            x, y, w_obj, h_obj = merged1[0]
            center_x = x + X_TARGET_1 + 1 + w_obj // 2
            center_y = y + h_obj // 2
            cv2.rectangle(frame1_vis, (x + X_TARGET_1 + 1, y),
                          (x + X_TARGET_1 + 1 + w_obj, y + h_obj), (0, 255, 0), 2)
            cv2.putText(frame1_vis, f"X:{center_x} Y:{center_y}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # === Управление "a" и "e" (полная логика из твоего второго скрипта) ===
            err_x = center_x - X_TARGET_1
            err_y = center_y - Y_TARGET_1
            in_x_zone = abs(err_x) <= X_TOL_1
            in_y_zone = abs(err_y) <= Y_TOL_1

            if not (in_x_zone and in_y_zone):
                command_sent = False

                if err_x > X_TOL_1 and err_y > Y_TOL_1:
                    current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
                    send_command(f"a{current_us['a']}")
                    print("→ Команда: ↓ (влево+вверх)")
                    command_sent = True

                elif err_x > X_TOL_1 and err_y < -Y_TOL_1:
                    current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
                    send_command(f"e{current_us['e']}")
                    print("→ Команда: W (влево+вниз)")
                    command_sent = True

                elif err_x < -X_TOL_1 and err_y > Y_TOL_1:
                    current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
                    send_command(f"e{current_us['e']}")
                    print("→ Команда: S (вправо+вверх)")
                    command_sent = True

                elif err_x < -X_TOL_1 and err_y < -Y_TOL_1:
                    current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
                    send_command(f"a{current_us['a']}")
                    print("→ Команда: ↑ (вправо+вниз)")
                    command_sent = True

                elif err_x > X_TOL_1:
                    if err_y >= 0:
                        current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
                        send_command(f"a{current_us['a']}")
                        print("→ Команда: ↓ (только X, Y ниже)")
                    else:
                        current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
                        send_command(f"e{current_us['e']}")
                        print("→ Команда: W (только X, Y выше)")
                    command_sent = True

                elif err_x < -X_TOL_1:
                    if err_y >= 0:
                        current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
                        send_command(f"a{current_us['a']}")
                        print("→ Команда: ↑ (только X, Y ниже)")
                    else:
                        current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
                        send_command(f"e{current_us['e']}")
                        print("→ Команда: S (только X, Y выше)")
                    command_sent = True

                elif err_y > Y_TOL_1:
                    if err_x >= 0:
                        current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
                        send_command(f"a{current_us['a']}")
                        print("→ Команда: ↓ (только Y, X справа)")
                    else:
                        current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
                        send_command(f"e{current_us['e']}")
                        print("→ Команда: S (только Y, X слева)")
                    command_sent = True

                elif err_y < -Y_TOL_1:
                    if err_x >= 0:
                        current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
                        send_command(f"e{current_us['e']}")
                        print("→ Команда: W (только Y, X справа)")
                    else:
                        current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
                        send_command(f"a{current_us['a']}")
                        print("→ Команда: ↑ (только Y, X слева)")
                    command_sent = True

                if command_sent:
                    time.sleep(MOVE_DELAY)
        # else: объект не найден — ничего не делаем
    # else: ROI пуст

    # === Обработка камеры 2 (выравнивание по Y=260) ===
    frame2_vis = frame2.copy()
    h2, w2 = frame2.shape[:2]

    # Рисуем линии на камере 2
    x_line2 = min(X_THRESH_2, w2 - 1)
    cv2.line(frame2_vis, (x_line2, 0), (x_line2, h2), (0, 0, 255), 2)  # красная вертикаль
    y_line2 = min(max(Y_TARGET_2, 0), h2 - 1)
    cv2.line(frame2_vis, (0, y_line2), (w2, y_line2), (255, 0, 0), 2)   # синяя горизонталь

    # ROI справа от X_THRESH_2
    if X_THRESH_2 + 1 < w2:
        roi2 = frame2[:, X_THRESH_2 + 1:]
        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        _, mask2 = cv2.threshold(gray2, 30, 255, cv2.THRESH_BINARY_INV)
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((15, 15)))
        mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, np.ones((7, 7)))
        contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects2 = [cv2.boundingRect(c) for c in contours2 if cv2.contourArea(c) > MIN_AREA_2]
        merged2 = merge_rectangles(rects2, MAX_DIST_2)

        if merged2:
            x, y, w_obj, h_obj = merged2[0]
            center_y2 = y + h_obj // 2
            x_full = x + X_THRESH_2 + 1
            cv2.rectangle(frame2_vis, (x_full, y), (x_full + w_obj, y + h_obj), (0, 255, 0), 2)
            cv2.putText(frame2_vis, f"Y:{center_y2}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Управление по оси "s"
            err_y2 = center_y2 - Y_TARGET_2
            if err_y2 > Y_TOL_2:
                # Объект ниже линии → нужно ПОДНЯТЬ → уменьшаем "s"
                current_us["s"] = max(MIN_US, current_us["s"] - STEP_S)
                send_command(f"s{current_us['s']}")
                print(f"→ Команда s: УМЕНЬШИТЬ (объект ниже Y={Y_TARGET_2})")
                time.sleep(MOVE_DELAY)
            elif err_y2 < -Y_TOL_2:
                # Объект выше → нужно ОПУСТИТЬ → увеличиваем "s"
                current_us["s"] = min(MAX_US, current_us["s"] + STEP_S)
                send_command(f"s{current_us['s']}")
                print(f"→ Команда s: УВЕЛИЧИТЬ (объект выше Y={Y_TARGET_2})")
                time.sleep(MOVE_DELAY)

    # Отображение
    cv2.imshow("Camera 1 - Positioning", frame1_vis)
    cv2.imshow("Camera 2 - Alignment (Y=260)", frame2_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Очистка
cap1.release()
cap2.release()
cv2.destroyAllWindows()
if ser:
    ser.close()