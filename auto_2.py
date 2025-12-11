import cv2
import numpy as np
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import messagebox
import threading
import time
from PIL import Image, ImageTk

# === Настройки ===
BAUD_RATE = 115200
STEP_FORWARD = 20
STEP_BACK = 18
STEP_US = 10
STEP_S = 20

current_us = {"s": 1500, "a": 1500, "e": 1500}
MIN_US = 500
MAX_US = 2400

ser = None
auto_mode_active = False
cap1 = None
cap2 = None

last_success_time = None
pump_activated = False

# === ПАРАМЕТРЫ ДЛЯ КАМЕР ===
# Камера 1 (позиционирование)
X_TARGET_1 = 120
X_TOL_1 = 8
Y_TOL_1 = 20
MIN_AREA_1 = 800
MAX_DIST_1 = 30

Y_TOP_1 = 170
Y_BOTTOM_1 = 270

NUM_UP_ACTIONS = 6

# Камера 2 (выравнивание) — ОСНОВНОЕ МЕСТО НАСТРОЙКИ:
Y_TARGET_2 = 200      # ← Горизонтальная линия
Y_TOL_2 = 50          # ←←← ДИАПАЗОН ДОПУСКА (± пикселей). МЕНЯТЬ ЭТОТ ПАРАМЕТР!
X_THRESH_2 = 280
MIN_AREA_2 = 800
MAX_DIST_2 = 30

X_ZONE_LEFT   = 30
X_ZONE_RIGHT  = 200
Y_ZONE_TOP    = 160
Y_ZONE_BOTTOM = 240

TOLERANCE = 35  # среднее между 30 и 40

# --- Вспомогательные функции ---
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
            raise Exception("ESP32 не найден. Подключите устройство по USB.")
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        status_label.config(text=f"Подключено к {port}", fg="green")
        print(f"Подключено к {port}")
        start_status_reader()
    except Exception as e:
        status_label.config(text=f"Ошибка: {e}", fg="red")
        print("Serial error:", e)

def send_command(cmd: str):
    if ser and ser.is_open:
        try:
            ser.write((cmd + "\n").encode('utf-8'))
            print(f"→ {cmd}")
        except Exception as e:
            status_label.config(text=f"Ошибка отправки: {e}", fg="red")

def on_key(event):
    if auto_mode_active:
        return  # игнорируем клавиши в автоматическом режиме

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

def update_labels():
    stand_label.config(text=f"Stand: {current_us['s']} µs")
    shoulder_label.config(text=f"Shoulder: {current_us['a']} µs")
    elbow_label.config(text=f"Elbow: {current_us['e']} µs")

def status_reader():
    while ser and ser.is_open:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("org:"):
                print(f"← {line}")
        except:
            break

def start_status_reader():
    thread = threading.Thread(target=status_reader, daemon=True)
    thread.start()

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

def auto_control_and_display():
    global auto_mode_active, cap1, cap2, success_enter_time, pump_triggered

    if cap1 is None or not cap1.isOpened():
        cap1 = cv2.VideoCapture(2)
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

        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        # === Камера 1: обработка ===
        Y_TARGET_1_ACTUAL = Y_TOP_1 + (Y_BOTTOM_1 - Y_TOP_1) // 2
        cv2.line(frame1, (X_TARGET_1, Y_TOP_1), (X_TARGET_1, Y_BOTTOM_1), (0, 0, 255), 2)
        cv2.line(frame1, (0, Y_TARGET_1_ACTUAL), (w1, Y_TARGET_1_ACTUAL), (255, 0, 0), 2)
        cv2.line(frame1, (30, Y_TOP_1), (30, Y_BOTTOM_1), (0, 0, 255), 2)
        cv2.line(frame1, (0, Y_TOP_1), (X_TARGET_1, Y_TOP_1), (0, 0, 255), 2)
        cv2.line(frame1, (0, Y_BOTTOM_1), (X_TARGET_1, Y_BOTTOM_1), (0, 0, 255), 2)

        roi1 = frame1[:, 31:] if 31 < w1 else np.zeros((h1, 1, 3), dtype=np.uint8)
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
                center_x = x + 31 + w_obj // 2
                center_y = y + h_obj // 2

                # Рисуем bounding box и точку
                in_forbidden_zone = (0 <= center_x <= 30) and (Y_TOP_1 <= center_y <= Y_BOTTOM_1)
                color = (0, 0, 255) if in_forbidden_zone else (0, 255, 0)
                cv2.rectangle(frame1, (x + 31, y), (x + 31 + w_obj, y + h_obj), color, 2)
                cv2.circle(frame1, (center_x, center_y), radius=10, color=(0, 255, 255), thickness=-1)
                cv2.putText(frame1, f"X:{center_x} Y:{center_y}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                if in_forbidden_zone:
                    cv2.putText(frame1, "IGNORED", (x + 31, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    # === Основная логика: Success-режим и управление ===
                    in_success_zone = (30 <= center_x <= 120) and (160 <= center_y <= 260)

                    if in_success_zone:
                        cv2.putText(frame1, "Success", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                        # Запуск таймера для помпы
                        if success_enter_time is None:
                            success_enter_time = time.time()
                            pump_triggered = False

                        # Включить помпу через 2 сек
                        if not pump_triggered and (time.time() - success_enter_time) >= 2.0:
                            print("→ Автоматически включаю помпу (p1)")
                            send_command("p1")
                            pump_triggered = True

                        # 🔴 ДВИЖЕНИЯ ОСТАНОВЛЕНЫ — ничего не делаем
                        pass

                    else:
                        # Объект вне зоны — сбрасываем и управляем
                        success_enter_time = None
                        pump_triggered = False

                        if auto_mode_active:
                            err_x = center_x - X_TARGET_1
                            err_y = center_y - Y_TARGET_1_ACTUAL
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

                    # === Аварийный режим (работает только вне Success-зоны) ===
                    if auto_mode_active and not in_success_zone:
                        if (center_y < Y_TOP_1 or center_y > Y_BOTTOM_1) and (center_x < 200):
                            print(f"🚨 Авария: Y={center_y} вне [{Y_TOP_1},{Y_BOTTOM_1}], X={center_x} < 200")
                            cv2.putText(frame1, "EMERGENCY!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                            for i in range(NUM_UP_ACTIONS):
                                current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
                                send_command(f"a{current_us['a']}")
                                time.sleep(0.1)

                            for i in range(6):
                                current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
                                send_command(f"e{current_us['e']}")
                                time.sleep(0.1)

                            root.after(0, update_labels)
                            time.sleep(0.3)

        # === Камера 2: выравнивание по Y ===
        cv2.line(frame2, (X_THRESH_2, 0), (X_THRESH_2, h2), (0, 0, 255), 2)
        cv2.line(frame2, (0, Y_TARGET_2), (w2, Y_TARGET_2), (255, 0, 0), 2)

        if X_THRESH_2 + 1 < w2:
            roi2 = frame2[:, X_THRESH_2 + 1:]
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
                cv2.rectangle(frame2, (x_full, y), (x_full + w_obj, y + h_obj), (0, 255, 0), 2)
                cv2.putText(frame2, f"Y:{center_y2}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Управление КАМЕРОЙ 2 — только если в автомате И НЕ в Success-режиме (опционально)
                if auto_mode_active and (success_enter_time is None or not (30 <= center_x <= 120 and 160 <= center_y <= 260)):
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

        # === Обновление GUI ===
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
# --- Управление автоматическим режимом ---
def toggle_auto_mode():
    global auto_mode_active
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

def on_closing():
    global cap1, cap2, ser, auto_mode_active
    auto_mode_active = False
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
root.geometry("700x820")
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

status_label = tk.Label(root, text="Нажмите 'Подключиться'", fg="gray", font=("Arial", 10))
status_label.pack(pady=5)

tk.Label(root, text="Камера 1 (Позиционирование)", font=("Arial", 10, "bold")).pack(pady=(10, 0))
label_cam1 = tk.Label(root)
label_cam1.pack(pady=2)

tk.Label(root, text="Камера 2 (Выравнивание по Y=260)", font=("Arial", 10, "bold")).pack(pady=(10, 0))
label_cam2 = tk.Label(root)
label_cam2.pack(pady=2)

# Пустые изображения
blank = ImageTk.PhotoImage(Image.new('RGB', (320, 240), (0, 0, 0)))
label_cam1.config(image=blank)
label_cam1.image = blank
label_cam2.config(image=blank)
label_cam2.image = blank

root.bind("<Key>", on_key)
update_labels()

# Запуск потока обновления камер сразу при старте
threading.Thread(target=auto_control_and_display, daemon=True).start()

root.mainloop()