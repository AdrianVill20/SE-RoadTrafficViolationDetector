import cv2
import os
from ultralytics import YOLO
import easyocr
from datetime import datetime

class HelmetDetector:
    def __init__(self, camera_index=0, video_path=None):
        self.running = False
        self.save_path = "captures"
        os.makedirs(self.save_path, exist_ok=True)

        # YOLO models
        self.helmet_model = YOLO("Weights/best.pt")   # helmet detection
        self.plate_model = YOLO("Weights/plate.pt")   # plate detection

        # OCR for plates
        self.ocr = easyocr.Reader(['en'], gpu=False)

        # Video capture
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(camera_index)

        self.recent_plates = set()  # avoid duplicate logging

    # ----------------- For video/webcam frames -----------------
    def get_frame(self):
        if not self.running:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # Resize for faster detection
        frame = cv2.resize(frame, (640, 480))
        return self._detect(frame)

    # ----------------- For single image -----------------
    def get_frame_from_image(self, img):
        img = cv2.resize(img, (640, 480))
        return self._detect(img.copy())

    # ----------------- Core detection -----------------
    def _detect(self, frame):
        results = self.helmet_model(frame)
        without_helmet_boxes = []

        # 1️⃣ Helmet detection
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 1:  # Without Helmet
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                    cv2.putText(frame, "Helmet: NO", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    without_helmet_boxes.append((x1, y1, x2, y2))
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, "Helmet: YES", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # 2️⃣ Plate detection on full frame
        try:
            plate_results = self.plate_model(frame)
            for p in plate_results:
                for pb in p.boxes:
                    px1, py1, px2, py2 = map(int, pb.xyxy[0])
                    plate_crop = frame[py1:py2, px1:px2]

                    if plate_crop.shape[0] < 20 or plate_crop.shape[1] < 20:
                        continue

                    # OCR
                    ocr_result = self.ocr.readtext(plate_crop)
                    if len(ocr_result) > 0:
                        text = ocr_result[0][1]

                        # Match with "without helmet" boxes
                        plate_center = ((px1+px2)//2, (py1+py2)//2)
                        match = any(
                            x1 <= plate_center[0] <= x2 and y1 <= plate_center[1] <= y2
                            for x1, y1, x2, y2 in without_helmet_boxes
                        )
                        if not match:
                            continue

                        # Avoid duplicates
                        if text not in self.recent_plates:
                            self.recent_plates.add(text)
                            if len(self.recent_plates) > 50:
                                self.recent_plates.pop()
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_file = os.path.join(self.save_path, f"{text}_{timestamp}.jpg")
                            cv2.imwrite(save_file, plate_crop)
                            print(f"[LOG] Saved plate: {text}")

                        # Draw plate on frame
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (255,0,0), 2)
                        cv2.putText(frame, f"Plate: {text}", (px1, py2+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        except Exception as e:
            print("Plate detection skipped:", e)

        return frame

    # ----------------- Release camera -----------------
    def release(self):
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
