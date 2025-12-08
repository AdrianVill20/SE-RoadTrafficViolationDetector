import cv2
import os
from ultralytics import YOLO
import easyocr
from datetime import datetime

# Models
helmet_model = YOLO("Weights/best.pt")      # your helmet detection
plate_model = YOLO("Weights/plate.pt")      # downloaded license plate detector
ocr = easyocr.Reader(['en'], gpu=False)

# Output folder
os.makedirs("captures", exist_ok=True)

def detect_from_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        # --- HELMET DETECTION ---
        helmet_results = helmet_model(frame)

        for r in helmet_results:
            for box in r.boxes:
                cls = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # class 1 = Without Helmet
                if cls == 1:
                    roi = frame[max(0, y1-20):y2+50, max(0, x1-50):x2+50]

                    # --- LICENSE PLATE DETECTION ---
                    plate_results = plate_model(roi)

                    for p in plate_results:
                        for pb in p.boxes:
                            px1, py1, px2, py2 = map(int, pb.xyxy[0])
                            plate_crop = roi[py1:py2, px1:px2]

                            # --- OCR ---
                            ocr_result = ocr.readtext(plate_crop)
                            if len(ocr_result) > 0:
                                text = ocr_result[0][1]
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                save_path = f"captures/{text}_{timestamp}.jpg"
                                cv2.imwrite(save_path, plate_crop)
                                print("Captured →", text, "→", save_path)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_from_camera()
