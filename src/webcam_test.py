import cv2

cap = cv2.VideoCapture(0)  # 0 = default webcam
if not cap.isOpened():
    print("Webcam not opened. Check camera index or permissions.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame captured")
        break

    cv2.imshow("Webcam Test - press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
