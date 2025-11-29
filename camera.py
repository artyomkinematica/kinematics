import cv2
import numpy as np

def merge_rectangles(rects, max_distance=20):
    """
    Объединяет прямоугольники, которые пересекаются или находятся близко друг к другу.
    rects: список [(x, y, w, h), ...]
    max_distance: максимальное расстояние между прямоугольниками для объединения
    """
    if not rects:
        return []

    merged = []
    rects = sorted(rects, key=lambda r: r[0])  # сортируем по X для оптимизации

    while len(rects) > 0:
        current = rects.pop(0)
        x1, y1, w1, h1 = current
        merged_current = [x1, y1, w1, h1]

        i = 0
        while i < len(rects):
            x2, y2, w2, h2 = rects[i]
            # Проверяем, пересекаются ли прямоугольники или находятся близко
            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))

            # Если есть пересечение или расстояние мало — объединяем
            if overlap_x > 0 or overlap_y > 0 or \
               (abs(x1 - x2) < max_distance and abs(y1 - y2) < max_distance):
                # Объединяем границы
                new_x = min(x1, x2)
                new_y = min(y1, y2)
                new_w = max(x1 + w1, x2 + w2) - new_x
                new_h = max(y1 + h1, y2 + h2) - new_y
                merged_current = [new_x, new_y, new_w, new_h]
                rects.pop(i)
                # Перезапускаем проверку с новым объединённым прямоугольником
                i = 0
                x1, y1, w1, h1 = merged_current
            else:
                i += 1

        merged.append(tuple(merged_current))

    return merged

# --- Основной код ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Камера не подключена")
    exit()

X_THRESHOLD = 100
MIN_CONTOUR_AREA = 800
MAX_DISTANCE_BETWEEN_RECTS = 30  # максимальное расстояние для объединения

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Рисуем линию
    cv2.line(frame, (X_THRESHOLD, 0), (X_THRESHOLD, frame.shape[0]), (0, 0, 255), 2)
    y_line1 = 220
    y_line2 = y_line1 + 80
    cv2.line(frame, (0, y_line1), (99, y_line1), (0, 0, 255), 2)
    cv2.line(frame, (0, y_line2), (99, y_line2), (0, 0, 255), 2)

    # ROI: справа от x=52
    roi = frame[:, X_THRESHOLD + 1:]
    if roi.size == 0:
        continue

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Морфология
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    # Находим контуры
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Собираем boundingRect'ы
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, w, h))

    # Объединяем близкие прямоугольники
    merged_rects = merge_rectangles(rects, MAX_DISTANCE_BETWEEN_RECTS)

    # Рисуем объединённые прямоугольники
    for x, y, w, h in merged_rects:
        x_full = x + X_THRESHOLD + 1
        cv2.rectangle(frame, (x_full, y), (x_full + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()