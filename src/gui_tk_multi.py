# integrated_gui.py
import tkinter as tk
from tkinter import Button, Label, Frame, filedialog
from PIL import Image, ImageTk
import cv2
import threading
import os
import time
from helmet_detector import IntegratedDetector

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Helmet + Plate Detector (Integrated)")
        self.window.geometry("1000x700")

        self.detector = IntegratedDetector()

        self.video_label = Label(window)
        self.video_label.pack()

        btn_frame = Frame(window)
        btn_frame.pack(pady=8)

        Button(btn_frame, text="Load Image", command=self.load_image).pack(side="left", padx=6)
        Button(btn_frame, text="Load Video", command=self.load_video).pack(side="left", padx=6)
        Button(btn_frame, text="Start Webcam", command=self.start_webcam).pack(side="left", padx=6)
        Button(btn_frame, text="Stop Webcam", command=self.stop_webcam).pack(side="left", padx=6)
        Button(btn_frame, text="Open Violations Folder", command=self.open_folder).pack(side="left", padx=6)
        Button(btn_frame, text="Open CSV Log", command=self.open_csv).pack(side="left", padx=6)

        self.running = False
        self.video_thread = None
        self.webcam_thread = None

    # ---------------- Image ----------------
    def load_image(self):
        path = filedialog.askopenfilename(title="Select Image",
                                          filetypes=[("Images","*.jpg *.jpeg *.png"), ("All","*.*")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            print("Failed to load image:", path)
            return
        try:
            annotated, _ = self.detector.detect_frame(cv2.resize(img, (960,540)))
            self.show_frame(annotated)
        except Exception as e:
            print("Error processing image:", e)

    # ---------------- Video ----------------
    def load_video(self):
        path = filedialog.askopenfilename(title="Select Video",
                                          filetypes=[("Videos","*.mp4 *.avi *.mov"), ("All","*.*")])
        if not path:
            return
        if self.video_thread and self.video_thread.is_alive():
            print("Video already playing")
            return
        self.video_thread = threading.Thread(target=self.video_loop, args=(path,))
        self.video_thread.daemon = True
        self.video_thread.start()

    def video_loop(self, path):
        cap = None
        try:
            cap = cv2.VideoCapture(path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    annotated, _ = self.detector.detect_frame(cv2.resize(frame, (960,540)))
                except Exception as e:
                    print("Error during video frame detection:", e)
                    annotated = frame
                # schedule GUI update in main thread
                self.window.after(0, lambda f=annotated: self.show_frame(f))
                time.sleep(0.03)
        except Exception as e:
            print("Video loop error:", e)
        finally:
            if cap:
                cap.release()

    # ---------------- Webcam ----------------
    def start_webcam(self):
        if self.running:
            return
        self.running = True
        self.detector.start_capture(camera_index=0)
        self.webcam_thread = threading.Thread(target=self.webcam_loop)
        self.webcam_thread.daemon = True
        self.webcam_thread.start()

    def stop_webcam(self):
        if not self.running:
            return
        self.running = False
        self.detector.stop_capture()

    def webcam_loop(self):
        try:
            cap = self.detector.cap
            if cap is None:
                print("Camera not opened.")
                self.running = False
                return
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                try:
                    annotated, _ = self.detector.detect_frame(cv2.resize(frame, (960,540)))
                except Exception as e:
                    print("Error during webcam frame detection:", e)
                    annotated = frame
                # schedule GUI update
                self.window.after(0, lambda f=annotated: self.show_frame(f))
                time.sleep(0.03)
        except Exception as e:
            print("Webcam thread crashed:", e)
        finally:
            self.detector.stop_capture()
            self.running = False

    # ---------------- UI helpers ----------------
    def show_frame(self, frame):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (920, 520))
            imgtk = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        except Exception as e:
            print("show_frame error:", e)

    def open_folder(self):
        os.startfile(os.path.abspath(self.detector.save_root))

    def open_csv(self):
        os.startfile(os.path.abspath(self.detector.csv_path))

    # ---------------- Close ----------------
    def on_close(self):
        print("Shutting down...")
        self.stop_webcam()
        # give threads time to stop
        time.sleep(0.1)
        self.detector.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
