import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('Ошибка камера не подключена!')
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Ошибка чтения кадра')
            break
        cv2.imshow('MES Vision 2', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()