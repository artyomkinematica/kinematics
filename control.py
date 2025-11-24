import socket
import tkinter as tk
from tkinter import messagebox

# Настройки подключения
HOST = "10.65.72.28"  # ← Заменить на IP вашего ESP32
PORT = 8080

# Домашние позиции (совпадают с теми, что в ESP32)
STAND_MID_POS = 85
SHOULDER_MID_POS = 60
ELBOW_MID_POS = 75

# Шаг изменения угла (начальное значение)
STEP = 10

# Диапазоны углов
ANGLE_RANGES = {
    "stand":   (-50, 180),
    "shoulder": (0, 180),
    "elbow":   (0, 180)
}

# Текущие значения углов (изначально домашние)
current_angles = {
    "stand": STAND_MID_POS,
    "shoulder": SHOULDER_MID_POS,
    "elbow": ELBOW_MID_POS
}

# Функция отправки команды
def send_command(cmd):
    global current_angles
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            s.connect((HOST, PORT))
            s.sendall(cmd.encode('utf-8') + b'\n')
            print(f"Отправлено: {cmd}")

            if cmd == "home":
                # Обновляем текущие углы до домашних
                current_angles["stand"] = STAND_MID_POS
                current_angles["shoulder"] = SHOULDER_MID_POS
                current_angles["elbow"] = ELBOW_MID_POS
                update_labels()
                status_label.config(text="Рука перемещена в домашнюю позицию")

    except Exception as e:
        print("Ошибка подключения:", e)
        status_label.config(text="Ошибка подключения")

def move_servo(name, delta):
    min_angle, max_angle = ANGLE_RANGES[name]
    new_angle = current_angles[name] + delta
    if min_angle <= new_angle <= max_angle:
        current_angles[name] = new_angle
        send_command(f"move {name} {new_angle}")
        update_labels()
        status_label.config(text=f"{name.capitalize()} перемещён на {new_angle}")

def claw_control(position):
    send_command(f"claw {position}")
    status_label.config(text=f"Захват: {position.upper()}")

def update_labels():
    stand_label.config(text=f"Поворот: {current_angles['stand']}°")
    shoulder_label.config(text=f"Плечо: {current_angles['shoulder']}°")
    elbow_label.config(text=f"Локоть: {current_angles['elbow']}°")

# Обновление шага
def update_step():
    global STEP
    try:
        new_step = int(step_entry.get())
        if new_step < 1 or new_step > 90:
            raise ValueError("Шаг должен быть от 1 до 90")
        STEP = new_step
        status_label.config(text=f"Шаг изменён на {STEP}")
    except Exception as e:
        messagebox.showerror("Ошибка", "Введите корректное число для шага (1–90).")
        status_label.config(text="Ошибка: Неверный шаг")

# Создание окна
root = tk.Tk()
root.title("Управление Робо-рукой")
root.geometry("600x700")
root.resizable(False, False)

# Стиль
font_main = ("Arial", 12)

# === Поле ввода шага ===
step_frame = tk.Frame(root)
step_frame.pack(pady=10)

tk.Label(step_frame, text="Шаг перемещения (1–90):", font=font_main).pack(side=tk.LEFT)
step_entry = tk.Entry(step_frame, width=5, font=font_main)
step_entry.insert(0, str(STEP))
step_entry.pack(side=tk.LEFT, padx=5)

tk.Button(step_frame, text="Применить", command=update_step, width=10).pack(side=tk.LEFT)

# Управление поворотом
tk.Label(root, text="Поворот основания", font=font_main).pack(pady=5)
stand_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
stand_label.pack()
tk.Button(root, text="← Влево", width=10, command=lambda: move_servo("stand", STEP)).pack(pady=2)
tk.Button(root, text="Вправо →", width=10, command=lambda: move_servo("stand", -STEP)).pack(pady=2)

# Управление плечом
tk.Label(root, text="Управление плечом", font=font_main).pack(pady=5)
shoulder_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
shoulder_label.pack()
tk.Button(root, text="↑ Назад", width=10, command=lambda: move_servo("shoulder", -STEP)).pack(pady=2)
tk.Button(root, text="↓ Вперед", width=10, command=lambda: move_servo("shoulder", STEP)).pack(pady=2)

# Управление локтем
tk.Label(root, text="Управление локтем", font=font_main).pack(pady=5)
elbow_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
elbow_label.pack()
tk.Button(root, text="↑ Вверх", width=10, command=lambda: move_servo("elbow", -STEP)).pack(pady=2)
tk.Button(root, text="↓ Вниз", width=10, command=lambda: move_servo("elbow", STEP)).pack(pady=2)

# Управление захватом
tk.Label(root, text="Захват", font=font_main).pack(pady=5)
tk.Button(root, text="Открыть", width=10, command=lambda: claw_control("open")).pack(pady=2)
tk.Button(root, text="Закрыть", width=10, command=lambda: claw_control("close")).pack(pady=2)

# Кнопка Home
tk.Button(root, text="Домой", width=10, bg="lightblue", command=lambda: send_command("home")).pack(pady=10)

# Статус
status_label = tk.Label(root, text="Ожидание команд...", fg="green", font=("Arial", 10))
status_label.pack(side="bottom", pady=10)

update_labels()

# Запуск интерфейса
root.mainloop()