# gui_tk.py
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
import cv2
from helmet_detector import HelmetDetector

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Helmet Detector")
        self.window.geometry("800x600")

        # Helmet detector
        self.detector = HelmetDetector()

        # Video display label
        self.video_label = Label(window)
        self.video_label.pack()

        # Start button
        self.start_btn = Button(window, text="Start Detection", command=self.start)
        self.start_btn.pack(side="left", padx=10, pady=10)

        # Stop button
        self.stop_btn = Button(window, text="Stop Detection", command=self.stop)
        self.stop_btn.pack(side="right", padx=10, pady=10)

        self.running = False

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
                # Convert to Tkinter format
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)

            # Repeat every 30 ms
            self.window.after(30, self.update_frame)

    def on_close(self):
        self.detector.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
