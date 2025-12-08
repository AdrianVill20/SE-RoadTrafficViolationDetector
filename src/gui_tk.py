# gui_tk_multi.py
import tkinter as tk
from tkinter import Button, Label, Frame, Scrollbar, Canvas
from PIL import Image, ImageTk
import cv2
import os
from helmet_detector import HelmetDetector

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Helmet Detector")
        self.window.geometry("900x700")

        # Helmet detector
        self.detector = HelmetDetector()

        # Pages
        self.pages = {}
        self.create_detection_page()
        self.create_logs_page()

        # Navigation buttons
        nav_frame = Frame(window)
        nav_frame.pack(side="top", fill="x")

        Button(nav_frame, text="Detection", command=lambda: self.show_page("detection")).pack(side="left", padx=5, pady=5)
        Button(nav_frame, text="Logs", command=lambda: self.show_page("logs")).pack(side="left", padx=5, pady=5)

        # Start with detection page
        self.show_page("detection")
        self.running = False

    # -----------------------------
    # Page: Detection
    # -----------------------------
    def create_detection_page(self):
        frame = Frame(self.window)
        self.pages["detection"] = frame

        self.video_label = Label(frame)
        self.video_label.pack()

        self.start_btn = Button(frame, text="Start Detection", command=self.start)
        self.start_btn.pack(side="left", padx=10, pady=10)

        self.stop_btn = Button(frame, text="Stop Detection", command=self.stop)
        self.stop_btn.pack(side="right", padx=10, pady=10)

    # -----------------------------
    # Page: Logs
    # -----------------------------
    def create_logs_page(self):
        frame = Frame(self.window)
        self.pages["logs"] = frame

        # Canvas + Scrollbar for images
        canvas = Canvas(frame, width=850, height=600)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self.logs_frame = Frame(canvas)
        canvas.create_window((0,0), window=self.logs_frame, anchor="nw")

        # Refresh button
        Button(frame, text="Refresh Logs", command=self.load_logs).pack(pady=5)

    # -----------------------------
    # Navigation
    # -----------------------------
    def show_page(self, name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[name].pack(fill="both", expand=True)

        if name == "logs":
            self.load_logs()

    # -----------------------------
    # Detection Controls
    # -----------------------------
    def start(self):
        self.detector.running = True
        self.running = True
        self.update_frame()

    def stop(self):
        self.detector.running = False
        self.running = False

    def update_frame(self):
        if self.running:
            frame = self.detector.get_frame()
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            self.window.after(30, self.update_frame)

    # -----------------------------
    # Logs
    # -----------------------------
    def load_logs(self):
        # Clear previous images
        for widget in self.logs_frame.winfo_children():
            widget.destroy()

        folder = self.detector.save_path
        if not os.path.exists(folder):
            return

        images = sorted(os.listdir(folder), reverse=True)
        for img_file in images:
            img_path = os.path.join(folder, img_file)
            pil_img = Image.open(img_path).resize((300, 200))
            imgtk = ImageTk.PhotoImage(pil_img)
            lbl = Label(self.logs_frame, image=imgtk)
            lbl.image = imgtk
            lbl.pack(padx=5, pady=5)

    # -----------------------------
    # Close
    # -----------------------------
    def on_close(self):
        self.detector.release()
        self.window.destroy()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
