# helmet_detector.py
import cv2
import cvzone
import time
import os
from ultralytics import YOLO

class HelmetDetector:
    def __init__(self):
        self.model = YOLO("Weights/best.pt")
        self.classNames = ['With Helmet', 'Without Helmet']
        self.cap = cv2.VideoCapture(0)
        self.running = False

        # Create folder for captured images
        self.save_path = "Captured_No_Helmet"
        os.makedirs(self.save_path, exist_ok=True)

        # To prevent saving too many pictures at once
        self.last_capture_time = 0  
        self.capture_delay = 2  # seconds

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

                label = self.classNames[cls]
                cvzone.putTextRect(img,
                                   f"{label} {conf:.2f}",
                                   (x1, max(30, y1)))

                # ----------------------------------------------
                # SAVE IMAGE IF WITHOUT HELMET
                # ----------------------------------------------
                if label == "Without Helmet":
                    current_time = time.time()
                    if current_time - self.last_capture_time >= self.capture_delay:

                        filename = f"{self.save_path}/no_helmet_{int(time.time())}.jpg"
                        cv2.imwrite(filename, img)
                        print(f"❗ Saved image: {filename}")

                        # Update last capture time
                        self.last_capture_time = current_time

        return img

    def release(self):
        self.cap.release()
