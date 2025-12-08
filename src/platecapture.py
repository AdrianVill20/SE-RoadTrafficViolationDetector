import cv2
import torch
from ultralytics import YOLO
import easyocr
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load YOLO model
model = YOLO("Weights/plate.pt")  # your trained model path

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Tkinter window
root = tk.Tk()
root.title("License Plate Detection & Recognition")
root.geometry("900x600")
root.configure(bg="lightgray")

panel = tk.Label(root)
panel.pack(padx=10, pady=10)

cap = None
stop_flag = False

def open_video():
    global cap, stop_flag
    stop_flag = False
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if video_path:
        cap = cv2.VideoCapture(video_path)
        update_frame()

def start_webcam():
    global cap, stop_flag
    stop_flag = False
    cap = cv2.VideoCapture(0)
    update_frame()

def stop_video():
    global stop_flag
    stop_flag = True

def update_frame():
    global cap, panel, stop_flag
    if stop_flag or cap is None:
        if cap:
            cap.release()
        return

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return

    frame = cv2.resize(frame, (800, 450))

    # Detect license plates
    results = model(frame)
    boxes = results[0].boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop plate area for OCR
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size != 0:
            ocr_results = reader.readtext(plate_crop)
            for (bbox, text, conf) in ocr_results:
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Confidence and label
        if box.conf is not None:
            conf = float(box.conf)
            cv2.putText(frame, f"{conf:.2f}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if box.cls is not None:
            cls_id = int(box.cls)
            label = model.names[cls_id]
            cv2.putText(frame, label, (x1, y2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Convert for Tkinter
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    panel.imgtk = imgtk
    panel.config(image=imgtk)

    panel.after(10, update_frame)

# Buttons
btn_frame = tk.Frame(root, bg="lightgray")
btn_frame.pack(pady=10)

tk.Button(btn_frame, text="Open Video", command=open_video, width=15, bg="green", fg="white").pack(side="left", padx=5)
tk.Button(btn_frame, text="Start Webcam", command=start_webcam, width=15, bg="blue", fg="white").pack(side="left", padx=5)
tk.Button(btn_frame, text="Stop", command=stop_video, width=15, bg="red", fg="white").pack(side="left", padx=5)

root.mainloop()
