# helmet_detector.py
import cv2
import cvzone
from ultralytics import YOLO

class HelmetDetector:
    def __init__(self):
        self.model = YOLO("Weights/best.pt")
        self.classNames = ['With Helmet', 'Without Helmet']
        self.cap = cv2.VideoCapture(0)
        self.running = False

    def get_frame(self):
        """Return processed frame from webcam."""
        if not self.running:
            return None

        success, img = self.cap.read()
        if not success:
            return None

        results = self.model(img, stream=True)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                cvzone.putTextRect(img,
                                   f"{self.classNames[cls]} {conf:.2f}",
                                   (x1, max(30, y1)))

        return img

    def release(self):
        self.cap.release()
