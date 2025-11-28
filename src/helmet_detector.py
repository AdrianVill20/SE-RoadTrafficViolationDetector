import cv2
import math
import cvzone
from ultralytics import YOLO

# Load model
model = YOLO("Weights/best.pt")

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

classNames = ['With Helmet', 'Without Helmet']
 
while True:
    success, img = cap.read()
    if not success:
        print("Camera not available.")
        break

    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1

            # Draw box
            cvzone.cornerRect(img, (x1, y1, w, h))

            # Confidence & class
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}',
                               (x1, max(30, y1)))

    cv2.imshow("Helmet Detector", img)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  
cv2.destroyAllWindows()
