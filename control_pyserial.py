import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import messagebox
import threading

# === Настройки ===
BAUD_RATE = 115200
STEP_US = 10  # шаг в микросекундах (1 мкс ≈ 0.1° → 1–2 мм)
STEP_FORWARD = 20  # шаг в микросекундах (1 мкс ≈ 0.1° → 1–2 мм)
STEP_BACK = 18  # шаг в микросекундах (1 мкс ≈ 0.1° → 1–2 мм)

# Текущие позиции (в микросекундах)
current_us = {
    "s": None,  # Stand
    "a": None,  # Shoulder
    "e": None  # Elbow
}

# Пределы (должны совпадать с ESP32)
MIN_US = 500
MAX_US = 2400

# Глобальный Serial
ser = None

def find_esp32_port():
    """Автоматически ищет порт ESP32"""
    for port in serial.tools.list_ports.comports():
        if "CP210" in port.description or "ESP32" in port.description or "USB" in port.description:
            return port.device
    return None


def connect_serial():
    global ser
    global current_us
    try:
        port = find_esp32_port()
        if not port:
            raise Exception("ESP32 не найден. Подключите устройство по USB.")
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
        # получение значений импульсов серво в течении одной сессии микроконтроллера
        line = ser.readline().decode('utf-8').strip()
        if line.startswith('org:'):
            sae = line[5:].split()
            current_us['s'] = int(sae[0][1:])
            current_us['a'] = int(sae[1][1:])
            current_us['e'] = int(sae[2][1:])

        status_label.config(text=f"Подключено к {port}", fg="green")
        print(f"Подключено к {port}")
        start_status_reader()
    except Exception as e:
        status_label.config(text=f"Ошибка: {e}", fg="red")
        print("Serial error:", e)


def send_command(cmd: str):
    """Отправка команды в ESP32"""
    if ser and ser.is_open:
        try:
            ser.write((cmd + "\n").encode('utf-8'))
            print(f"→ {cmd}")
        except Exception as e:
            status_label.config(text=f"Ошибка отправки: {e}", fg="red")


# Обработка стрелок и клавиш
def on_key(event):
    key = event.keysym
    if key == "Up":
        current_us["a"] = max(MIN_US, current_us["a"] - STEP_FORWARD)
        send_command(f"a{current_us['a']}")
    elif key == "Down":
        current_us["a"] = min(MAX_US, current_us["a"] + STEP_BACK)
        send_command(f"a{current_us['a']}")
    elif key == "Left":
        current_us["s"] = max(MIN_US, current_us["s"] - STEP_US)
        send_command(f"s{current_us['s']}")
    elif key == "Right":
        current_us["s"] = min(MAX_US, current_us["s"] + STEP_US)
        send_command(f"s{current_us['s']}")
    elif key == "w":  # локоть вверх
        current_us["e"] = max(MIN_US, current_us["e"] - STEP_BACK)
        send_command(f"e{current_us['e']}")
    elif key == "s":  # локоть вниз
        current_us["e"] = min(MAX_US, current_us["e"] + STEP_FORWARD)
        send_command(f"e{current_us['e']}")
    elif key == "1":
        send_command("p1")  # помпа вкл
    elif key == "0":
        send_command("p0")  # помпа выкл
    elif key == "h":
        # Домашняя позиция (1500 мкс для всех)
        for k in current_us:
            current_us[k] = 1500
        send_command("s1500")
        send_command("a1500")
        send_command("e1500")

    update_labels()


def update_labels():
    stand_label.config(text=f"Stand: {current_us['s']} µs")
    shoulder_label.config(text=f"Shoulder: {current_us['a']} µs")
    elbow_label.config(text=f"Elbow: {current_us['e']} µs")


# Чтение org-строк из Serial (в фоне)
def status_reader():
    while ser and ser.is_open:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line.startswith("org:"):
                print(f"← {line}")
                # Можно парсить и обновлять GUI, но опционально
        except:
            break


def start_status_reader():
    thread = threading.Thread(target=status_reader, daemon=True)
    thread.start()


root = tk.Tk()
root.title("Управление робо-рукой (Serial + Стрелки)")
root.geometry("500x500")
root.resizable(False, False)

tk.Label(root, text="Управление клавиатурой:", font=("Arial", 12, "bold")).pack(pady=5)
tk.Label(root, text="← → : поворот основания\n↑ ↓ : плечо\nW/S : локоть\n1/0 : помпа вкл/выкл\nH : домой",
         justify="left", font=("Arial", 10)).pack(pady=5)

# Текущие значения
stand_label = tk.Label(root, text="", font=("Arial", 11))
shoulder_label = tk.Label(root, text="", font=("Arial", 11))
elbow_label = tk.Label(root, text="", font=("Arial", 11))
stand_label.pack()
shoulder_label.pack()
elbow_label.pack()

# Кнопка подключения
connect_btn = tk.Button(root, text="Подключиться к ESP32", command=connect_serial, bg="lightgreen")
connect_btn.pack(pady=10)

# Статус
status_label = tk.Label(root, text="Нажмите 'Подключиться'", fg="gray", font=("Arial", 10))
status_label.pack(side="bottom", pady=5)

# Привязка клавиш
root.bind("<Key>", on_key)
update_labels()

root.mainloop()