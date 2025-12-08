import cv2
import os
from ultralytics import YOLO
import easyocr
from datetime import datetime

class HelmetDetector:
    def __init__(self, camera_index=0):
        self.running = False
        self.save_path = "captures"
        os.makedirs(self.save_path, exist_ok=True)

        # Load models
        self.helmet_model = YOLO("Weights/best.pt")
        self.plate_model = YOLO("Weights/plate.pt")
        self.ocr = easyocr.Reader(['en'], gpu=False)

        # Camera
        self.cap = cv2.VideoCapture(camera_index)

    def get_frame(self):
        if not self.running:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        # --- HELMET DETECTION ---
        results = self.helmet_model(frame)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if cls == 1:  # Without Helmet
                    roi = frame[max(0, y1-20):y2+50, max(0, x1-50):x2+50]

                    # --- PLATE DETECTION ---
                    plate_results = self.plate_model(roi)
                    for p in plate_results:
                        for pb in p.boxes:
                            px1, py1, px2, py2 = map(int, pb.xyxy[0])
                            plate_crop = roi[py1:py2, px1:px2]

                            # --- OCR ---
                            ocr_result = self.ocr.readtext(plate_crop)
                            if len(ocr_result) > 0:
                                text = ocr_result[0][1]
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_file = os.path.join(self.save_path, f"{text}_{timestamp}.jpg")
                                cv2.imwrite(save_file, plate_crop)

                                # Draw rectangles + label on frame
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                                cv2.putText(frame, f"Helmet: NO", (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                                cv2.putText(frame, f"Plate: {text}", (x1, y2+20),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        return frame

    def release(self):
        self.cap.release()
