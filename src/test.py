import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
import csv
from datetime import datetime
from ultralytics import YOLO
import easyocr
import traceback

# ----------------- Detector Classes ----------------- #

class PlateLocalizer:
    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 80, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        return gray, edges

    def find_plate_regions(self, img):
        _, edges = self.preprocess(img)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        h_img, w_img = img.shape[:2]
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if w == 0 or h == 0: continue
            aspect = w / float(h)
            area = w * h
            if 2.0 <= aspect <= 8.0 and 2000 <= area <= 50000:
                if w < w_img*0.95 and h < h_img*0.6:
                    candidates.append((x,y,w,h))
        return candidates

class IntegratedDetector:
    def __init__(self, helmet_model_path="Weights/best.pt", plate_model_path="Weights/plate.pt",
                 save_root="violations", ocr_langs=['en']):
        self.save_root = save_root
        os.makedirs(self.save_root, exist_ok=True)
        self.person_folder = os.path.join(self.save_root, "persons")
        self.plate_folder = os.path.join(self.save_root, "plates")
        os.makedirs(self.person_folder, exist_ok=True)
        os.makedirs(self.plate_folder, exist_ok=True)

        self.csv_path = os.path.join(self.save_root, "violations.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp","plate_text","person_image","plate_image","ocr_confidence"])

        print("Loading models...")
        self.helmet_model = YOLO(helmet_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.ocr = easyocr.Reader(ocr_langs, gpu=False)
        self.cap = None
        self.recent_plates = set()

    def _save_violation(self, plate_text, plate_crop, person_crop, confidence):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plate_text_safe = "".join(c for c in plate_text if c.isalnum() or c in ("-", "_")).strip() or "UNKNOWN"
        person_path = os.path.join(self.person_folder, f"person_{plate_text_safe}_{timestamp}.jpg")
        plate_path = os.path.join(self.plate_folder, f"plate_{plate_text_safe}_{timestamp}.jpg")
        try:
            cv2.imwrite(person_path, person_crop)
            cv2.imwrite(plate_path, plate_crop)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, plate_text, person_path, plate_path, f"{confidence:.2f}"])
            print(f"[LOG] Violation saved: {plate_text} | person: {person_path} | plate: {plate_path}")
        except Exception as e:
            print("Error saving violation files:", e)
            traceback.print_exc()

    def detect_frame(self, frame):
        annotated = frame.copy()

        # ----- Helmet detection ----- #
        helmet_results = self.helmet_model(frame)
        without_helmet_boxes = []

        for r in helmet_results:
            for box in r.boxes:
                cls = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == 1:
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(annotated, "Helmet: NO", (x1, max(0,y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    without_helmet_boxes.append((x1,y1,x2,y2))
                else:
                    cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(annotated, "Helmet: YES", (x1, max(0,y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ----- Plate detection ----- #
        plate_results = self.plate_model(frame)
        for r in plate_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (255,0,0), 2)
                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size == 0: continue
                try:
                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    h_crop, w_crop = gray.shape[:2]
                    if max(h_crop, w_crop) < 150:
                        scale = int(150 / max(h_crop, w_crop)) + 1
                        gray = cv2.resize(gray, (w_crop*scale, h_crop*scale), interpolation=cv2.INTER_CUBIC)
                    _, thr = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    ocr_res = self.ocr.readtext(thr)
                    if ocr_res:
                        best = max(ocr_res, key=lambda x: x[2])
                        text, conf = best[1].strip(), best[2]
                        cv2.putText(annotated, text, (x1, max(0,y1-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                        # Save violation only if helmet NO
                        for (hx1,hy1,hx2,hy2) in without_helmet_boxes:
                            center_x = (x1+x2)//2
                            center_y = (y1+y2)//2
                            if hx1 <= center_x <= hx2 and hy1 <= center_y <= hy2:
                                person_crop = frame[hy1:hy2, hx1:hx2]
                                self._save_violation(text, plate_crop, person_crop, conf)
                                break
                except Exception as e:
                    print("OCR/plate processing error:", e)
                    traceback.print_exc()
        return annotated

    def start_capture(self, source=0):
        self.cap = cv2.VideoCapture(source)
        return self.cap

    def stop_capture(self):
        if self.cap:
            self.cap.release()
            self.cap = None

# ----------------- GUI ----------------- #
class ViolationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Helmet & License Detector")
        self.detector = IntegratedDetector()
        self.frame = None
        self.running = False

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="Open Webcam", command=self.open_webcam).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Open Video", command=self.open_video).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Open Image", command=self.open_image).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Stop Capture", command=self.stop_capture).pack(side=tk.LEFT)

        # Canvas
        self.canvas = tk.Canvas(root, width=800, height=600)
        self.canvas.pack()

        # Logs
        self.log_frame = tk.Frame(root)
        self.log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_list = tk.Listbox(self.log_frame, height=10)
        self.log_list.pack(fill=tk.BOTH, expand=True)
        self.load_csv_logs()

    def load_csv_logs(self):
        if os.path.exists(self.detector.csv_path):
            self.log_list.delete(0, tk.END)
            with open(self.detector.csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    self.log_list.insert(tk.END, f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")

    def update_canvas(self):
        if not self.running or self.detector.cap is None:
            return
        ret, frame = self.detector.cap.read()
        if not ret:
            self.stop_capture()
            return
        annotated = self.detector.detect_frame(frame)
        self.frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(self.frame).resize((800,600))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0,0,anchor=tk.NW, image=imgtk)
        self.root.after(10, self.update_canvas)
        self.load_csv_logs()

    def open_webcam(self):
        self.stop_capture()
        self.detector.start_capture(0)
        self.running = True
        self.update_canvas()

    def open_video(self):
        self.stop_capture()
        path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if path:
            self.detector.start_capture(path)
            self.running = True
            self.update_canvas()

    def open_image(self):
        self.stop_capture()
        path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if path:
            frame = cv2.imread(path)
            annotated = self.detector.detect_frame(frame)
            self.frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(self.frame).resize((800,600))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0,0,anchor=tk.NW, image=imgtk)
            self.load_csv_logs()

    def stop_capture(self):
        self.running = False
        self.detector.stop_capture()

if __name__ == "__main__":
    root = tk.Tk()
    app = ViolationApp(root)
    root.mainloop()
