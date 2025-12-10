import cv2
import os
import time
import json
from ultralytics import YOLO
import easyocr
import cvzone

class HelmetDetector:
    def __init__(self, camera_index=0, cooldown_seconds=3):
        # Resolve paths relative to this file for robustness
        base = os.path.dirname(os.path.abspath(__file__))  # src/
        self.weights_dir = os.path.join(base, "Weights")

        # Models - ensure files exist
        self.model = YOLO(os.path.join(self.weights_dir, "best.pt"))
        self.plate_model = YOLO(os.path.join(self.weights_dir, "plate.pt"))
        self.classNames = ['With Helmet', 'Without Helmet']

        # Camera
        self.cap = cv2.VideoCapture(camera_index)
        self.running = False

        # OCR
        self.reader = easyocr.Reader(['en'], gpu=False)

        # Paths (absolute)
        self.save_path = os.path.join(base, "Captured_No_Helmet")
        self.violation_image_path = os.path.join(base, "violations_images")
        self.violations_json_path = os.path.join(base, "violations.json")
        self.plate_info_file = os.path.join(base, "detected_plate_info.txt")

        # Ensure folders exist
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.violation_image_path, exist_ok=True)

        # Controls
        self.last_capture_time = 0.0
        self.capture_delay = 2.0      # seconds between saved images for same event
        self.confidence_threshold = 0.5
        self.image_captured = False
        self.capture_cooldown = cooldown_seconds  # seconds before allowing next capture
        self.last_violation_time = 0.0

    def get_frame(self):
        """Read camera frame, run detections, optionally save violation and update JSON."""
        if not self.running:
            return None

        success, img = self.cap.read()
        if not success or img is None:
            return None

        # initialize per-frame trackers
        highest_confidence = 0.0
        highest_label = ""
        now = time.time()

        # Run helmet model safely
        try:
            helmet_results = self.model(img, stream=True)
        except Exception as e:
            print("Helmet model error:", e)
            helmet_results = []

        try:
            plate_results = self.plate_model(img, stream=True)
        except Exception as e:
            print("Plate model error:", e)
            plate_results = []

        # Helmet detection boxes / annotations
        for r in helmet_results:
            for box in getattr(r, "boxes", []):
                try:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                except Exception:
                    continue
                label = self.classNames[cls] if 0 <= cls < len(self.classNames) else str(cls)

                if conf > highest_confidence:
                    highest_confidence = conf
                    highest_label = label

                try:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                except Exception:
                    continue

                # clip coordinates to image bounds
                h_img, w_img = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                w, h = x2 - x1, y2 - y1
                try:
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    cvzone.putTextRect(img, f"{label} {conf:.2f}", (x1, max(30, y1)))
                except Exception:
                    pass

        # License plate detection + OCR + save logic
        plate_found_text = None
        for plate in plate_results:
            for box in getattr(plate, "boxes", []):
                try:
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                except Exception:
                    continue

                # clip ROI
                h_img, w_img = img.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w_img - 1, x2), min(h_img - 1, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                plate_roi = img[y1c:y2c, x1c:x2c]

                # OCR safely
                plate_text = self.extract_plate_text(plate_roi)
                plate_found_text = plate_text  # store last detected plate for possible save
                print(f"Detected Plate Text: {plate_text}")

                if conf >= self.confidence_threshold:
                    try:
                        cvzone.cornerRect(img, (x1c, y1c, x2c - x1c, y2c - y1c))
                        cvzone.putTextRect(img, f"Plate: {plate_text} {conf:.2f}", (x1c, max(30, y1c)))
                    except Exception:
                        pass

                    # Save info only if helmet violation is observed and we haven't recently captured
                    if (highest_label == "Without Helmet"
                        and highest_confidence >= self.confidence_threshold
                        and not self.image_captured
                        and now - self.last_violation_time >= self.capture_cooldown):
                        fname = self.save_violation_image(img)
                        violation_info = {
                            'license_plate': plate_text,
                            'violation': 'Helmet Violation',
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'image_path': fname
                        }
                        self.save_violation_info(violation_info)
                        self.image_captured = True
                        self.last_violation_time = now

        # Save no-plate image if there's a helmet violation and we still haven't captured for this event
        if (highest_label == "Without Helmet"
            and highest_confidence >= self.confidence_threshold
            and not self.image_captured):
            if now - self.last_capture_time >= self.capture_delay:
                file = os.path.join(self.save_path, f"no_helmet_{int(now)}.jpg")
                try:
                    cv2.imwrite(file, img)
                    print(f"Saved no-plate image: {file}")
                except Exception as e:
                    print("Failed to save no-plate image:", e)
                self.last_capture_time = now
                self.image_captured = True
                self.last_violation_time = now

        # RESET image_captured when situation clears or after cooldown (so future violations are recorded)
        # If helmet is now present, clear flag immediately.
        if highest_label != "Without Helmet":
            # person wearing helmet again -> ready to capture future violations
            self.image_captured = False
        else:
            # If still without helmet, allow reset after capture_cooldown to permit new capture
            if now - self.last_violation_time >= self.capture_cooldown:
                self.image_captured = False

        return img

    def detect(self, image):
        """Run detection on a supplied image (numpy array). Returns annotated image and may save events."""
        # This mirrors get_frame logic but for external images
        try:
            results = self.model(image)
        except Exception as e:
            print("Helmet model error (detect):", e)
            results = []

        try:
            plate_results = self.plate_model(image)
        except Exception as e:
            print("Plate model error (detect):", e)
            plate_results = []

        highest_confidence = 0.0
        highest_label = ""
        now = time.time()

        for r in results:
            for box in getattr(r, "boxes", []):
                try:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                except Exception:
                    continue
                label = self.classNames[cls] if 0 <= cls < len(self.classNames) else str(cls)
                if conf > highest_confidence:
                    highest_confidence = conf
                    highest_label = label
                try:
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                    h_img, w_img = image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
                    if x2 > x1 and y2 > y1:
                        cvzone.cornerRect(image, (x1, y1, x2 - x1, y2 - y1))
                        cvzone.putTextRect(image, f"{label} {conf:.2f}", (x1, max(30, y1)))
                except Exception:
                    pass

        for plate in plate_results:
            for box in getattr(plate, "boxes", []):
                try:
                    conf = float(box.conf[0])
                    coords = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, coords)
                except Exception:
                    continue

                h_img, w_img = image.shape[:2]
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(w_img - 1, x2), min(h_img - 1, y2)
                if x2c <= x1c or y2c <= y1c:
                    continue

                plate_roi = image[y1c:y2c, x1c:x2c]
                plate_text = self.extract_plate_text(plate_roi)

                if conf >= self.confidence_threshold:
                    try:
                        cvzone.cornerRect(image, (x1c, y1c, x2c - x1c, y2c - y1c))
                        cvzone.putTextRect(image, f"Plate: {plate_text} {conf:.2f}", (x1c, max(30, y1c)))
                    except Exception:
                        pass

                    if (highest_label == "Without Helmet"
                        and highest_confidence >= self.confidence_threshold
                        and not self.image_captured
                        and now - self.last_violation_time >= self.capture_cooldown):
                        fname = self.save_violation_image(image)
                        violation_info = {
                            'license_plate': plate_text,
                            'violation': 'Helmet Violation',
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'image_path': fname
                        }
                        self.save_violation_info(violation_info)
                        self.image_captured = True
                        self.last_violation_time = now

        # reset logic similar to get_frame
        if highest_label != "Without Helmet":
            self.image_captured = False
        else:
            if now - self.last_violation_time >= self.capture_cooldown:
                self.image_captured = False

        return image

    def extract_plate_text(self, plate_roi):
        """Safely run OCR; returns best text or 'Unknown'."""
        if plate_roi is None or plate_roi.size == 0:
            return "Unknown"
        try:
            res = self.reader.readtext(plate_roi)
            if res:
                # pick highest-confidence entry if possible
                # easyocr returns list of (bbox, text, confidence)
                best = max(res, key=lambda r: r[2] if len(r) > 2 else 0)
                return best[1]
        except Exception as e:
            print("OCR error:", e)
        return "Unknown"

    def save_plate_info(self, plate_text):
        try:
            with open(self.plate_info_file, "a", encoding="utf-8") as f:
                f.write(f"Detected License Plate: {plate_text}\n")
        except Exception as e:
            print("Failed to save plate info:", e)

    def save_violation_image(self, img):
        """Save image and return filename only (Flask-friendly)."""
        fname = f"violation_{int(time.time())}.jpg"
        full_path = os.path.join(self.violation_image_path, fname)
        cv2.imwrite(full_path, img)
        return fname

    def save_violation_info(self, violation_info):
        """Append violation to JSON safely and flush to disk."""
        try:
            # load existing data if any
            if os.path.exists(self.violations_json_path) and os.path.getsize(self.violations_json_path) > 0:
                with open(self.violations_json_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except Exception:
                        data = []
            else:
                data = []
        except Exception as e:
            print("Failed reading existing violations.json:", e)
            data = []

        data.append(violation_info)

        # write atomically: write -> flush -> fsync
        try:
            with open(self.violations_json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print("Failed to write violations.json:", e)

    def release(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

if __name__ == "__main__":
    # quick test runner if you want to test this file directly
    det = HelmetDetector(camera_index=0)
    det.running = True
    print("Starting camera. Press 'q' in window to quit.")
    while True:
        frame = det.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        cv2.imshow("Helmet Detector", frame)
        # reset image_captured after cooldown so you can capture again for testing
        if det.image_captured and (time.time() - det.last_violation_time) > det.capture_cooldown:
            det.image_captured = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    det.release()
    cv2.destroyAllWindows()
