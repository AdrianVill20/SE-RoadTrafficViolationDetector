import tkinter as tk
from tkinter import Button, Label, Frame, Scrollbar, Canvas, filedialog
from PIL import Image, ImageTk
import cv2
import os
from helmet_detector import HelmetDetector
import time  # <-- Add this import

class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Helmet & License Plate Detection")
        self.window.geometry("900x700")  # Set an initial window size
        self.window.resizable(True, True)  # Allow resizing but with limits

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
        self.video_label.pack(padx=10, pady=10)

        # Add buttons for uploading image and video
        self.upload_image_button = Button(frame, text="Upload Image", command=self.upload_image)
        self.upload_image_button.pack(side="left", padx=10, pady=10)

        self.upload_video_button = Button(frame, text="Upload Video", command=self.upload_video)
        self.upload_video_button.pack(side="left", padx=10, pady=10)

        # Start and Stop buttons for detection
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
        canvas.create_window((0, 0), window=self.logs_frame, anchor="nw")

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
        # Check if an image or video is uploaded, else start the webcam
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.running = True
            self.update_frame()
        else:
            # If no video is uploaded, run the webcam detection
            self.detector.running = True
            self.running = True
            self.update_frame()

    def stop(self):
        if hasattr(self, 'cap'):
            self.cap.release()
        self.detector.running = False
        self.running = False

    def update_frame(self):
        if self.running:
            if hasattr(self, 'cap') and self.cap.isOpened():  # Process video
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Process frame using the detector
                    frame = self.detector.detect(frame)  # Use the detect method from HelmetDetector
                    frame = self.resize_frame(frame)  # Resize the frame to fit window

                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

                    # Save the processed video frame for logs
                    self.save_log_frame(frame)

                else:
                    print("Failed to capture video frame or reached the end of video.")
                    self.stop_video()  # Stop if video ends or no frame is available

            else:  # Process webcam feed for helmet detection
                frame = self.detector.get_frame()  # Get the frame from webcam
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = self.resize_frame(frame)  # Resize the frame to fit window
                    img = Image.fromarray(frame)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.imgtk = imgtk
                    self.video_label.configure(image=imgtk)

                    # Save the processed webcam frame for logs
                    self.save_log_frame(frame)

            self.window.after(30, self.update_frame)

    def resize_frame(self, frame):
        """Resize the frame to fit within the window."""
        if frame is None:  # Check if frame is None
            return frame

        # Get the window's width and height
        max_width = self.window.winfo_width() - 40  # 40px padding
        max_height = self.window.winfo_height() - 150  # Some padding for the buttons

        # Resize the frame
        height, width, _ = frame.shape
        aspect_ratio = width / height

        if width > max_width:
            width = max_width
            height = int(width / aspect_ratio)

        if height > max_height:
            height = max_height
            width = int(height * aspect_ratio)

        # Resize the image while maintaining the aspect ratio
        resized_frame = cv2.resize(frame, (width, height))
        return resized_frame

    def save_log_frame(self, frame):
        """Save the processed frame for logs."""
        if frame is not None:
            save_path = "logs"
            os.makedirs(save_path, exist_ok=True)
            filename = os.path.join(save_path, f"frame_{int(time.time())}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved frame to log: {filename}")

    # -----------------------------
    # Image and Video Upload
    # -----------------------------
    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.process_image(file_path)

    def upload_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if file_path:
            self.process_video(file_path)

    def process_image(self, file_path):
        # Load the image
        image = cv2.imread(file_path)
        
        # Process the image using the detect() method
        frame = self.detector.detect(image)  # Use the detect method from HelmetDetector

        # Convert the processed frame to RGB for Tkinter display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the image to fit within the window
        frame = self.resize_frame(frame)

        # Display the processed image in Tkinter
        img = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=photo)
        self.video_label.image = photo

        # Save the processed image to logs
        self.save_log_frame(frame)

    def process_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        self.running = True
        self.update_frame()

    def stop_video(self):
        self.cap.release()
        self.running = False

    # -----------------------------
    # Logs
    # -----------------------------
    def load_logs(self):
        # Clear previous images
        for widget in self.logs_frame.winfo_children():
            widget.destroy()

        folder = "logs"  # Logs folder
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
        if hasattr(self, 'cap'):
            self.cap.release()
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
