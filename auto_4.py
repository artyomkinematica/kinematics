import cv2
import numpy as np
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import messagebox
import threading
import time
from PIL import Image, ImageTk

# параметры для подключения к ESP32 + настройка шагов для автоматического управления по камерам
BAUD_RATE = 115200
STEP_FORWARD = 20
STEP_BACK = 18
STEP_US = 10
STEP_S = 20

# словарь для хранения данных о положении сервомоторов в данный момент + минимальное и максимальное значение для серво
current_us = {"s": None, "a": None, "e": None}
MIN_US = 500
MAX_US = 2400

ser = None
auto_mode_active = False
cap1 = None
cap2 = None

# переменные флаги
success_enter_time = None
pump_triggered = False
post_success_sequence_done = True
success_mode_enabled = False
stop_requested = False
success_zone1_entered = False
success_zone2_entered = False
success_partial_time = None
emergency_lift_in_progress = False
emergency_lift_done = False
success_full_start_time = None
success_full_confirmed = False
SUCCESS_HOLD_TIME = 2

# параметры 1-ой камеры (разлиновка для центрирования)
X_TARGET_1 = 110
X_TARGET_MIN = 60
X_TOL_1 = 8
Y_TOL_1 = 20
MIN_AREA_1 = 800
MAX_DIST_1 = 30

Y_TOP_1 = 200
Y_BOTTOM_1 = 240

NUM_UP_ACTIONS = 6

# параметры для 2-ой камеры (разлиновка для центрирования)

Y_TARGET_2 = 240
Y_TOL_2 = 50
X_THRESH_2 = 120
MIN_AREA_2 = 800
MAX_DIST_2 = 30

X_TARGET_MIN_2 = 120
X_TARGET_2 = 190
Y_SUCCESS_TOP_2 = 200
Y_SUCCESS_BOTTOM_2 = 280


# функции для подключения к ESP32
def find_esp32_port():
    for port in serial.tools.list_ports.comports():
        if "CP210" in port.description or "ESP32" in port.description or "USB" in port.description:
            return port.device
    return None


def connect_serial():
    global ser, current_us
    try:
        port = find_esp32_port()
        if not port:
            raise Exception("ESP32 не найден. Подключите устройство по USB.")
        ser = serial.Serial(port, BAUD_RATE, timeout=1)

        line = ser.readline().decode('utf-8').strip()
        if line.startswith('org:'):
            sae = line[5:].split()
            current_us['s'] = int(sae[0][1:])
            current_us['a'] = int(sae[1][1:])
            current_us['e'] = int(sae[2][1:])

        status_label.config(text=f"Подключено к {port}", fg="green")
        print(f"Подключено к {port}")
    except Exception as e:
        status_label.config(text=f"Ошибка: {e}", fg="red")
        print("Serial error:", e)

# функция для отправки команд на микроконтроллер
def send_command(cmd: str):
    if ser and ser.is_open and not stop_requested:
        try:
            ser.write((cmd + "\n").encode('utf-8'))
            print(f"→ {cmd}")
        except Exception as e:
            status_label.config(text=f"Ошибка отправки: {e}", fg="red")

# функция для ручного управления (стрелки + ws + вкл/откл помпу + вовзрат в "домашнее положение")
def on_key(event):
    if auto_mode_active or stop_requested:
        return

    key = event.keysym
    changed = False
    if key == "Up":
        current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
        changed = True
    elif key == "Down":
        current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
        changed = True
    elif key == "Left":
        current_us["s"] = max(MIN_US, current_us["s"] - STEP_US)
        changed = True
    elif key == "Right":
        current_us["s"] = min(MAX_US, current_us["s"] + STEP_US)
        changed = True
    elif key == "w":
        current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
        changed = True
    elif key == "s":
        current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
        changed = True
    elif key == "1":
        send_command("p1")
        return
    elif key == "0":
        send_command("p0")
        return
    elif key == "h":
        for k in current_us:
            current_us[k] = 1500
        send_command("s1500")
        send_command("a1500")
        send_command("e1500")
        update_labels()
        return

    if changed:
        if key in ("Up", "Down"):
            send_command(f"a{current_us['a']}")
        elif key in ("Left", "Right"):
            send_command(f"s{current_us['s']}")
        elif key in ("w", "s"):
            send_command(f"e{current_us['e']}")
        update_labels()

# обновление текста в GUI
def update_labels():
    stand_label.config(text=f"Stand: {current_us['s']} µs")
    shoulder_label.config(text=f"Shoulder: {current_us['a']} µs")
    elbow_label.config(text=f"Elbow: {current_us['e']} µs")

# функция для объединения
def merge_rectangles(rects, max_distance=20):
    """Функция объединения близких прямоугольников"""
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


def perform_emergency_lift():
    """Выполнить аварийное поднятие (как в камере 1)"""
    global stop_requested, emergency_lift_in_progress, emergency_lift_done

    if emergency_lift_in_progress:
        print("⚠️ Аварийное поднятие уже выполняется, пропускаю...")
        return

    emergency_lift_in_progress = True
    emergency_lift_done = True  # Помечаем что поднятие выполнено

    print("🚨 Выполняю аварийное поднятие (деталь в success_zone1, но не в success_zone2)")

    # Аварийный режим - поднятие
    for i in range(NUM_UP_ACTIONS):
        if stop_requested:
            emergency_lift_in_progress = False
            return
        current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
        send_command(f"a{current_us['a']}")
        for _ in range(10):
            if stop_requested:
                break
            time.sleep(0.01)

    for i in range(6):
        if stop_requested:
            emergency_lift_in_progress = False
            return
        current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
        send_command(f"e{current_us['e']}")
        for _ in range(10):
            if stop_requested:
                break
            time.sleep(0.01)

    root.after(0, update_labels)

    # Короткая пауза
    for _ in range(30):
        if stop_requested:
            break
        time.sleep(0.01)

    print("✅ Аварийное поднятие завершено")
    emergency_lift_in_progress = False


# --- Основной цикл: обновление видео + управление ---
def auto_control_and_display():
    global auto_mode_active, cap1, cap2, success_enter_time, pump_triggered, post_success_sequence_done, stop_requested
    global success_zone1_entered, success_zone2_entered, success_partial_time, emergency_lift_done, emergency_lift_in_progress
    global success_full_start_time, success_full_confirmed  # добавить эту строку

    if cap1 is None or not cap1.isOpened():
        cap1 = cv2.VideoCapture(1)
    if cap2 is None or not cap2.isOpened():
        cap2 = cv2.VideoCapture(0)

    if not cap1.isOpened() or not cap2.isOpened():
        root.after(0, lambda: status_label.config(text="Ошибка: камеры не найдены", fg="red"))
        return

    MOVE_DELAY = 0.5

    while True:
        if not (cap1.isOpened() and cap2.isOpened()):
            break

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        # === Если нажат "Стоп" — только показываем видео, без логики ===
        if stop_requested:
            def update_gui_only():
                if not root.winfo_exists():
                    return
                try:
                    img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                    img1 = cv2.resize(img1, (320, 240))
                    img2 = cv2.resize(img2, (320, 240))
                    pil1 = Image.fromarray(img1)
                    pil2 = Image.fromarray(img2)
                    tk_img1 = ImageTk.PhotoImage(pil1)
                    tk_img2 = ImageTk.PhotoImage(pil2)
                    label_cam1.config(image=tk_img1)
                    label_cam1.image = tk_img1
                    label_cam2.config(image=tk_img2)
                    label_cam2.image = tk_img2
                except Exception as e:
                    print("Ошибка обновления GUI:", e)

            root.after(0, update_gui_only)
            time.sleep(0.03)
            continue

        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        # Если аварийное поднятие в процессе, сбрасываем все success флаги
        if emergency_lift_in_progress:
            success_zone1_entered = False
            success_zone2_entered = False
            success_partial_time = None

        # Сбрасываем флаги для нового кадра (но сохраняем если уже был success)
        if success_enter_time is None and not emergency_lift_done:
            # Только сбрасываем если не было аварийного поднятия для этой детали
            if not (success_zone1_entered or success_zone2_entered):
                success_zone1_entered = False
                success_zone2_entered = False

        # === Камера 1: линии и обработка ===
        Y_TARGET_1_ACTUAL = Y_TOP_1 + (Y_BOTTOM_1 - Y_TOP_1) // 2

        cv2.line(frame1, (X_TARGET_1, Y_TOP_1), (X_TARGET_1, Y_BOTTOM_1), (0, 0, 255), 2)
        cv2.line(frame1, (0, Y_TARGET_1_ACTUAL), (w1, Y_TARGET_1_ACTUAL), (255, 0, 0), 2)
        cv2.line(frame1, (X_TARGET_MIN, Y_TOP_1), (X_TARGET_MIN, Y_BOTTOM_1), (0, 0, 255), 2)
        cv2.line(frame1, (0, Y_TOP_1), (X_TARGET_1, Y_TOP_1), (0, 0, 255), 2)
        cv2.line(frame1, (0, Y_BOTTOM_1), (X_TARGET_1, Y_BOTTOM_1), (0, 0, 255), 2)

        roi1 = frame1[:, X_TARGET_MIN:] if X_TARGET_MIN < w1 else np.zeros((h1, 1, 3), dtype=np.uint8)

        if roi1.size > 0:
            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            _, mask1 = cv2.threshold(gray1, 30, 255, cv2.THRESH_BINARY_INV)
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
            contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rects1 = [cv2.boundingRect(c) for c in contours1 if cv2.contourArea(c) > MIN_AREA_1]
            merged1 = merge_rectangles(rects1, MAX_DIST_1)

            if merged1:
                x, y, w_obj, h_obj = merged1[0]
                center_x = x + X_TARGET_MIN + w_obj // 2
                center_y = y + h_obj // 2

                cv2.circle(frame1, (center_x, center_y), radius=5, color=(0, 255, 255), thickness=-1)

                in_forbidden_zone = (0 <= center_x <= X_TARGET_MIN) and (Y_TOP_1 <= center_y <= Y_BOTTOM_1)
                color = (0, 0, 255) if in_forbidden_zone else (0, 255, 0)
                cv2.rectangle(frame1, (x + X_TARGET_MIN, y), (x + X_TARGET_MIN + w_obj, y + h_obj), color, 2)
                cv2.putText(frame1, f"X:{center_x} Y:{center_y}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if not in_forbidden_zone:
                    in_success_zone1 = (X_TARGET_MIN <= center_x <= X_TARGET_1) and (180 <= center_y <= 270)

                    # Отмечаем на изображении если в зоне success
                    if in_success_zone1:
                        cv2.putText(frame1, "In Success Zone 1", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        success_zone1_entered = True

                        # Если деталь только что попала в зону 1 и поднятие не выполнено, запоминаем время
                        if success_partial_time is None and not emergency_lift_done:
                            success_partial_time = time.time()
                    else:
                        # Если вышли из зоны success и не в процессе поднятия
                        if success_partial_time is not None and not success_zone2_entered and not emergency_lift_in_progress:
                            success_partial_time = None
                            emergency_lift_done = False  # Сбрасываем флаг поднятия

                    if auto_mode_active and not emergency_lift_in_progress:
                        err_x = center_x - X_TARGET_1
                        err_y = center_y - Y_TARGET_1_ACTUAL
                        in_x_zone = abs(err_x) <= X_TOL_1
                        in_y_zone = abs(err_y) <= Y_TOL_1

                        if not (in_x_zone and in_y_zone):
                            command_sent = False
                            if err_x > X_TOL_1 and err_y > Y_TOL_1:
                                current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
                                send_command(f"a{current_us['a']}")
                                command_sent = True
                            elif err_x > X_TOL_1 and err_y < -Y_TOL_1:
                                current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
                                send_command(f"e{current_us['e']}")
                                command_sent = True
                            elif err_x < -X_TOL_1 and err_y > Y_TOL_1:
                                current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
                                send_command(f"e{current_us['e']}")
                                command_sent = True
                            elif err_x < -X_TOL_1 and err_y < -Y_TOL_1:
                                current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
                                send_command(f"a{current_us['a']}")
                                command_sent = True
                            elif err_x > X_TOL_1:
                                if err_y >= 0:
                                    current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
                                    send_command(f"a{current_us['a']}")
                                else:
                                    current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
                                    send_command(f"e{current_us['e']}")
                                command_sent = True
                            elif err_x < -X_TOL_1:
                                if err_y >= 0:
                                    current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
                                    send_command(f"a{current_us['a']}")
                                else:
                                    current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
                                    send_command(f"e{current_us['e']}")
                                command_sent = True
                            elif err_y > Y_TOL_1:
                                if err_x >= 0:
                                    current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
                                    send_command(f"a{current_us['a']}")
                                else:
                                    current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
                                    send_command(f"e{current_us['e']}")
                                command_sent = True
                            elif err_y < -Y_TOL_1:
                                if err_x >= 0:
                                    current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
                                    send_command(f"e{current_us['e']}")
                                else:
                                    current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
                                    send_command(f"a{current_us['a']}")
                                command_sent = True

                            if command_sent:
                                time.sleep(MOVE_DELAY)
                                root.after(0, update_labels)

                        # Аварийный режим (как в первом коде)
                        if (center_y < Y_TOP_1 or center_y > Y_BOTTOM_1) and (center_x < X_TARGET_1):
                            if stop_requested or emergency_lift_in_progress:
                                continue
                            print(f"🚨 Авария: Y={center_y} вне [{Y_TOP_1},{Y_BOTTOM_1}], X={center_x} < {X_TARGET_1}")
                            cv2.putText(frame1, "EMERGENCY!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            perform_emergency_lift()

        # === Камера 2: выравнивание по Y ===
        cv2.line(frame2, (X_THRESH_2, 0), (X_THRESH_2, h2), (0, 0, 255), 2)
        cv2.line(frame2, (0, Y_TARGET_2), (w2, Y_TARGET_2), (255, 0, 0), 2)

        # Рисуем зону Success для камеры 2
        cv2.rectangle(frame2,
                      (X_TARGET_MIN_2, Y_SUCCESS_TOP_2),
                      (X_TARGET_2, Y_SUCCESS_BOTTOM_2),
                      (0, 0, 255), 2)

        if X_THRESH_2 + 1 < w2:
            roi2 = frame2[:, X_THRESH_2 + 1:]
            if roi2.size > 0:
                gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
                _, mask2 = cv2.threshold(gray2, 30, 255, cv2.THRESH_BINARY_INV)
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

                contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rects2 = [cv2.boundingRect(c) for c in contours2 if cv2.contourArea(c) > MIN_AREA_2]
                merged2 = merge_rectangles(rects2, MAX_DIST_2)

                if merged2:
                    x, y, w_obj, h_obj = merged2[0]
                    center_y2 = y + h_obj // 2
                    x_full = x + X_THRESH_2 + 1
                    center_x2 = x_full + w_obj // 2

                    cv2.circle(frame2, (center_x2, center_y2), radius=5, color=(0, 255, 255), thickness=-1)

                    # Success-зона для второй камеры
                    in_success_zone2 = (
                            X_TARGET_MIN_2 <= center_x2 <= X_TARGET_2 and
                            Y_SUCCESS_TOP_2 <= center_y2 <= Y_SUCCESS_BOTTOM_2
                    )

                    if in_success_zone2:
                        cv2.putText(frame2, "In Success Zone 2", (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.rectangle(frame2, (x_full, y), (x_full + w_obj, y + h_obj), (0, 255, 255), 3)
                        success_zone2_entered = True
                    else:
                        # Если вышли из зоны 2 и не в процессе поднятия
                        if success_partial_time is not None and not success_zone1_entered and not emergency_lift_in_progress:
                            success_partial_time = None
                            emergency_lift_done = False  # Сбрасываем флаг поднятия

                    # Базовый прямоугольник
                    cv2.rectangle(frame2, (x_full, y), (x_full + w_obj, y + h_obj), (0, 255, 0), 2)
                    cv2.putText(frame2, f"Y:{center_y2} X:{center_x2}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    # Управление выравниванием по Y
                    if auto_mode_active and not emergency_lift_in_progress:
                        err_y2 = center_y2 - Y_TARGET_2
                        if err_y2 > Y_TOL_2:
                            current_us["s"] = max(MIN_US, current_us["s"] - STEP_S)
                            send_command(f"s{current_us['s']}")
                            time.sleep(MOVE_DELAY)
                            root.after(0, update_labels)
                        elif err_y2 < -Y_TOL_2:
                            current_us["s"] = min(MAX_US, current_us["s"] + STEP_S)
                            send_command(f"s{current_us['s']}")
                            time.sleep(MOVE_DELAY)
                            root.after(0, update_labels)

        # === ЛОГИКА ОБРАБОТКИ SUCCESS СОСТОЯНИЙ ===
        if success_mode_enabled and success_enter_time is None and not emergency_lift_in_progress:
            # Проверяем условия для partial success (только одна зона)
            if success_zone1_entered and not success_zone2_entered and not emergency_lift_done:
                # Если деталь в зоне 1 больше 1 секунды и нет в зоне 2
                if success_partial_time is not None and (time.time() - success_partial_time) > 1.0:
                    print("⚠️ Деталь в success_zone1, но не в success_zone2. Выполняю аварийное поднятие.")
                    cv2.putText(frame1, "PARTIAL SUCCESS - EMERGENCY LIFT", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    # Выполняем аварийное поднятие
                    perform_emergency_lift()
                    # Сбрасываем состояние
                    success_partial_time = None

            # --- Проверка полного success с таймером удержания ---
            if success_zone1_entered and success_zone2_entered:
                # Если только что вошли в обе зоны – запоминаем время
                if success_full_start_time is None:
                    success_full_start_time = time.time()
                    success_full_confirmed = False
                else:
                    # Уже были в зонах – проверяем длительность удержания
                    if (time.time() - success_full_start_time) >= SUCCESS_HOLD_TIME:
                        if not success_full_confirmed:
                            success_full_confirmed = True
                            print("✅ Success! Деталь продержалась в обеих зонах достаточно долго.")
                            cv2.putText(frame1, "Success", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                            cv2.putText(frame2, "Success", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                            # Отключаем авторежим и запускаем послеуспешную последовательность
                            auto_mode_active = False
                            auto_btn.config(text="Автомат", bg="lightblue")
                            status_label.config(text="Авто режим: отключён (успех)", fg="green")
                            success_enter_time = time.time()
                            pump_triggered = False
                            success_partial_time = None
                            emergency_lift_done = False
            else:
                # Если деталь вышла из зон до подтверждения – сбрасываем таймер
                if success_full_start_time is not None and not success_full_confirmed:
                    success_full_start_time = None
                    # (опционально) print("Таймер полного success сброшен – выход из зон")

        # === Выполнение последовательности после полного Success ===
        if success_enter_time is not None and not pump_triggered and not emergency_lift_in_progress:
            if (time.time() - success_enter_time) >= 2.0:
                if stop_requested:
                    continue
                print("→ Включаю помпу (p1)")
                send_command("p1")
                pump_triggered = True

                # Выполняем 45 раз: ↑ (плечо вверх) + s (локоть вниз)
                for i in range(45):
                    if stop_requested:
                        print("→ Последовательность прервана (Стоп)")
                        break
                    current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
                    send_command(f"a{current_us['a']}")
                    for _ in range(15):
                        if stop_requested:
                            break
                        time.sleep(0.01)

                    if stop_requested:
                        break

                    current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
                    send_command(f"e{current_us['e']}")
                    for _ in range(15):
                        if stop_requested:
                            break
                        time.sleep(0.01)

                    root.after(0, update_labels)

                if stop_requested:
                    continue

                for _ in range(100):
                    if stop_requested:
                        print("→ Пауза 1 сек прервана (Стоп)")
                        break
                    time.sleep(0.01)

                if not stop_requested:
                    # Плавное перемещение плеча к 2010
                    target_a = 2010
                    step = 25
                    while current_us["a"] < target_a and not stop_requested:
                        current_us["a"] = min(target_a, current_us["a"] + step)
                        send_command(f"a{current_us['a']}")
                        for _ in range(20):
                            if stop_requested:
                                break
                            time.sleep(0.01)
                        root.after(0, update_labels)

                    # Плавное перемещение локтя к 2042
                    target_e = 2042
                    while current_us["e"] < target_e and not stop_requested:
                        current_us["e"] = min(target_e, current_us["e"] + step)
                        send_command(f"e{current_us['e']}")
                        for _ in range(20):
                            if stop_requested:
                                break
                            time.sleep(0.01)
                        root.after(0, update_labels)

                    # Плавное перемещение основания к 2390
                    print("→ Устанавливаю stand = 2390")
                    target_s = 2390
                    while current_us["s"] < target_s and not stop_requested:
                        current_us["s"] = min(target_s, current_us["s"] + step)
                        send_command(f"s{current_us['s']}")
                        for _ in range(20):
                            if stop_requested:
                                break
                            time.sleep(0.01)
                        root.after(0, update_labels)

                    if not stop_requested:
                        print("→ Выключаю помпу (p0)")
                        send_command("p0")

                        post_success_sequence_done = False
                        status_label.config(
                            text="Последовательность завершена. Нажмите 'Сброс Success'.",
                            fg="orange"
                        )

        # === Обновление изображений ===
        def update_gui():
            if not root.winfo_exists():
                return
            try:
                img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
                img1 = cv2.resize(img1, (320, 240))
                img2 = cv2.resize(img2, (320, 240))
                pil1 = Image.fromarray(img1)
                pil2 = Image.fromarray(img2)
                tk_img1 = ImageTk.PhotoImage(pil1)
                tk_img2 = ImageTk.PhotoImage(pil2)
                label_cam1.config(image=tk_img1)
                label_cam1.image = tk_img1
                label_cam2.config(image=tk_img2)
                label_cam2.image = tk_img2
            except Exception as e:
                print("Ошибка обновления GUI:", e)

        root.after(0, update_gui)
        time.sleep(0.03)


def emergency_pump_off():
    """Отправляет команду выключения помпы даже если stop_requested=True"""
    if ser and ser.is_open:
        try:
            ser.write(b"p0\n")
            print("→ Экстренное выключение помпы (p0)")
        except Exception as e:
            status_label.config(text=f"Ошибка отправки p0: {e}", fg="red")


def toggle_stop_continue():
    global stop_requested, auto_mode_active, emergency_lift_in_progress, emergency_lift_done
    if stop_requested:
        # Возобновляем управление
        stop_requested = False
        control_btn.config(text="Стоп", bg="red")
        status_label.config(text="Управление возобновлено", fg="green")
    else:
        # Останавливаем всё
        emergency_pump_off()
        stop_requested = True
        auto_mode_active = False
        emergency_lift_in_progress = False
        emergency_lift_done = False
        auto_btn.config(text="Автомат", bg="lightblue")
        control_btn.config(text="Продолжить", bg="lightgreen")
        status_label.config(text="Все команды остановлены (Стоп)", fg="orange")


# --- Управление автоматическим режимом ---
def toggle_auto_mode():
    global auto_mode_active, post_success_sequence_done
    if not post_success_sequence_done:
        messagebox.showinfo("Информация", "Сначала завершите или сбросьте последовательность после Success!")
        return

    if auto_mode_active:
        auto_mode_active = False
        auto_btn.config(text="Автомат", bg="lightblue")
        status_label.config(text="Авто режим: остановлен", fg="gray")
    else:
        if ser is None or not ser.is_open:
            messagebox.showwarning("Ошибка", "Сначала подключитесь к ESP32!")
            return
        auto_mode_active = True
        auto_btn.config(text="Автомат (РАБОТАЕТ)", bg="lightgreen")
        status_label.config(text="Авто режим: активен", fg="blue")


def toggle_success_mode():
    global success_mode_enabled
    success_mode_enabled = not success_mode_enabled
    if success_mode_enabled:
        success_toggle_btn.config(text="Выключить Success-режим", bg="red")
        status_label.config(text="Success-режим: ВКЛЮЧЁН", fg="green")
    else:
        success_toggle_btn.config(text="Включить Success-режим", bg="lightgreen")
        status_label.config(text="Success-режим: ВЫКЛЮЧЕН", fg="gray")


def reset_success_state():
    global success_enter_time, pump_triggered, post_success_sequence_done, success_mode_enabled, stop_requested
    global success_zone1_entered, success_zone2_entered, success_partial_time, emergency_lift_done, emergency_lift_in_progress
    global success_full_start_time, success_full_confirmed

    success_full_start_time = None
    success_full_confirmed = False

    success_enter_time = None
    pump_triggered = False
    post_success_sequence_done = True
    success_mode_enabled = False
    stop_requested = False
    success_zone1_entered = False
    success_zone2_entered = False
    success_partial_time = None
    emergency_lift_done = False
    emergency_lift_in_progress = False

    success_toggle_btn.config(text="Включить Success-режим", bg="lightgreen")
    status_label.config(text="Состояние сброшено. Включите Success-режим для новой детали.", fg="green")
    auto_btn.config(state="normal")
    control_btn.config(text="Стоп", bg="red")


def on_closing():
    global cap1, cap2, ser, auto_mode_active, stop_requested
    auto_mode_active = False
    stop_requested = True
    time.sleep(0.1)
    if cap1:
        cap1.release()
    if cap2:
        cap2.release()
    if ser:
        ser.close()
    root.destroy()


# --- Создание GUI ---
root = tk.Tk()
root.title("Управление робо-рукой с камерами")
root.geometry("700x900")
root.resizable(False, False)
root.protocol("WM_DELETE_WINDOW", on_closing)

tk.Label(root, text="Управление:", font=("Arial", 12, "bold")).pack(pady=5)
tk.Label(root, text="← → : основание\n↑ ↓ : плечо\nW/S : локоть\n1/0 : помпа\nH : домой",
         justify="left", font=("Arial", 10)).pack(pady=5)

stand_label = tk.Label(root, text="", font=("Arial", 11))
shoulder_label = tk.Label(root, text="", font=("Arial", 11))
elbow_label = tk.Label(root, text="", font=("Arial", 11))
stand_label.pack()
shoulder_label.pack()
elbow_label.pack()

connect_btn = tk.Button(root, text="Подключиться к ESP32", command=connect_serial, bg="lightgreen")
connect_btn.pack(pady=5)

auto_btn = tk.Button(root, text="Автомат", command=toggle_auto_mode, bg="lightblue", font=("Arial", 10, "bold"))
auto_btn.pack(pady=5)

control_btn = tk.Button(root, text="Стоп", command=toggle_stop_continue, bg="red", font=("Arial", 10, "bold"))
control_btn.pack(pady=5)

reset_btn = tk.Button(root, text="Сброс Success", command=reset_success_state, bg="lightcoral")
reset_btn.pack(pady=5)

success_toggle_btn = tk.Button(root, text="Включить Success-режим", command=toggle_success_mode, bg="lightgreen")
success_toggle_btn.pack(pady=5)

status_label = tk.Label(root, text="Нажмите 'Подключиться'", fg="gray", font=("Arial", 10))
status_label.pack(pady=5)

tk.Label(root, text="Камера 1 (Позиционирование)", font=("Arial", 10, "bold")).pack(pady=(10, 0))
label_cam1 = tk.Label(root)
label_cam1.pack(pady=2)

tk.Label(root, text="Камера 2 (Выравнивание по Y=240)", font=("Arial", 10, "bold")).pack(pady=(10, 0))
label_cam2 = tk.Label(root)
label_cam2.pack(pady=2)

blank = ImageTk.PhotoImage(Image.new('RGB', (320, 240), (0, 0, 0)))
label_cam1.config(image=blank)
label_cam1.image = blank
label_cam2.config(image=blank)
label_cam2.image = blank

root.bind("<Key>", on_key)
update_labels()

threading.Thread(target=auto_control_and_display, daemon=True).start()

root.mainloop()