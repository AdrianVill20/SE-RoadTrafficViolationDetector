import cv2
import os
import time
import json  # For saving violation information in JSON format
from ultralytics import YOLO
import easyocr  # Import EasyOCR
import numpy as np
import cvzone
import base64  # For encoding the image to base64

class HelmetDetector:
    def __init__(self):
        self.model = YOLO("Weights/best.pt")  # Load the pretrained YOLO model for helmet detection
        self.plate_model = YOLO("Weights/plate.pt")  # Load the pretrained YOLO model for license plate detection
        self.classNames = ['With Helmet', 'Without Helmet']
        self.cap = cv2.VideoCapture(0)  # Initialize the webcam
        self.running = False

        # Initialize EasyOCR reader
        self.reader = easyocr.Reader(['en'])  # Set language to English

        # Create folder for captured images
        self.save_path = "Captured_No_Helmet"
        os.makedirs(self.save_path, exist_ok=True)

        # Folder for violation images
        self.violation_image_path = "violations_images"
        os.makedirs(self.violation_image_path, exist_ok=True)

        # To prevent saving too many pictures at once
        self.last_capture_time = 0  
        self.capture_delay = 2  # seconds
        self.confidence_threshold = 0.5  # Confidence threshold to save an image

        # Flag to ensure only one image is captured per violation
        self.image_captured = False  # Flag to track if image has been saved for violation

    def get_frame(self):
        """Return processed frame from webcam with both helmet and plate detection."""
        if not self.running:
            return None

        success, img = self.cap.read()
        if not success:
            return None

        # Perform helmet detection
        helmet_results = self.model(img, stream=True)
        
        # Perform license plate detection
        plate_results = self.plate_model(img, stream=True)

        highest_confidence = 0  # Variable to track the highest confidence score
        highest_label = ""  # Variable to store the corresponding label
        highest_confidence_box = None  # To store the bounding box with highest confidence

        # Process helmet detection results and find the highest confidence label
        for r in helmet_results:
            for box in r.boxes:
                conf = float(box.conf[0])  # Confidence score
                cls = int(box.cls[0])

                label = self.classNames[cls]

                # Update the highest confidence if a higher one is found
                if conf > highest_confidence:
                    highest_confidence = conf
                    highest_label = label
                    highest_confidence_box = box

                # Draw the helmet bounding box and label
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f"{label} {conf:.2f}", (x1, max(30, y1)))

        # Process license plate detection results and read text using EasyOCR
        plate_texts = []  # Store detected license plate texts
        for plate in plate_results:
            for box in plate.boxes:
                conf = float(box.conf[0])  # Confidence score
                
                # Extract the region of interest (ROI) for the license plate
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                plate_roi = img[y1:y2, x1:x2]  # Crop the image to the license plate region
                
                # Apply EasyOCR to extract text from the license plate
                plate_text = self.extract_plate_text(plate_roi)
                print(f"Detected Plate Text: {plate_text}")

                # Only display and save the plate if the confidence is high enough
                if conf >= self.confidence_threshold:
                    cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1))
                    cvzone.putTextRect(img, f"Plate: {plate_text} {conf:.2f}", (x1, max(30, y1)))

                    # Save the detected plate information
                    plate_texts.append(plate_text)
                    self.save_plate_info(plate_text)  # Save the detected plate text

                    # Save violation information if "Without Helmet" is detected
                    if highest_label == "Without Helmet" and highest_confidence >= self.confidence_threshold and not self.image_captured:
                        violation_info = {
                            'license_plate': plate_text,
                            'violation': 'Helmet Violation',
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                            'image_path': self.save_violation_image(img)  # Save the image path as proof
                        }
                        self.save_violation_info(violation_info)

                        # Set the flag to True after saving the image
                        self.image_captured = True

        # Save the image only if "Without Helmet" label has the highest confidence and exceeds the threshold
        if highest_label == "Without Helmet" and highest_confidence >= self.confidence_threshold and not self.image_captured:
            current_time = time.time()
            if current_time - self.last_capture_time >= self.capture_delay:
                filename = f"{self.save_path}/no_helmet_{int(time.time())}.jpg"
                cv2.imwrite(filename, img)
                print(f"❗ Saved image: {filename}")

                # Update last capture time
                self.last_capture_time = current_time

                # Set the flag to True after saving the image
                self.image_captured = True

        return img

    def detect(self, image):
        """Process an uploaded image for helmet and plate detection."""
        # Run the model on the uploaded image
        results = self.model(image)
        plate_results = self.plate_model(image)

        highest_confidence = 0
        highest_label = ""
        highest_confidence_box = None

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.classNames[cls]
                
                if conf > highest_confidence:
                    highest_confidence = conf
                    highest_label = label
                    highest_confidence_box = box

                # Draw bounding box for helmet
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(image, (x1, y1, w, h))
                cvzone.putTextRect(image, f"{label} {conf:.2f}", (x1, max(30, y1)))

        # Process license plate results and read text using EasyOCR
        plate_texts = []  # List to store plate texts
        for plate in plate_results:
            for box in plate.boxes:
                conf = float(box.conf[0])
                
                # Extract the plate region
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                plate_roi = image[y1:y2, x1:x2]

                # Apply EasyOCR to extract text from the plate region
                plate_text = self.extract_plate_text(plate_roi)
                print(f"Detected Plate Text: {plate_text}")

                if conf >= self.confidence_threshold:
                    cvzone.cornerRect(image, (x1, y1, x2 - x1, y2 - y1))
                    cvzone.putTextRect(image, f"Plate: {plate_text} {conf:.2f}", (x1, max(30, y1)))

                    plate_texts.append(plate_text)
                    self.save_plate_info(plate_text)

                    # Save violation information if "Without Helmet" is detected
                    if highest_label == "Without Helmet" and highest_confidence >= self.confidence_threshold and not self.image_captured:
                        violation_info = {
                            'license_plate': plate_text,
                            'violation': 'Helmet Violation',
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                            'image_path': self.save_violation_image(image)  # Save the image path as proof
                        }
                        self.save_violation_info(violation_info)

                        # Set the flag to True after saving the image
                        self.image_captured = True

        if highest_label == "Without Helmet" and highest_confidence >= self.confidence_threshold and not self.image_captured:
            current_time = time.time()
            if current_time - self.last_capture_time >= self.capture_delay:
                filename = f"{self.save_path}/no_helmet_{int(time.time())}.jpg"
                cv2.imwrite(filename, image)
                print(f"❗ Saved image: {filename}")
                self.last_capture_time = current_time

                # Set the flag to True after saving the image
                self.image_captured = True

        return image

    def extract_plate_text(self, plate_roi):
        """Use EasyOCR to extract text from the license plate ROI."""
        # Apply EasyOCR to extract text from the plate region
        result = self.reader.readtext(plate_roi)
        
        # EasyOCR returns a list of results (bounding boxes + text)
        if result:
            plate_text = result[0][1]  # Get the text from the first detected result
            return plate_text
        return "Unknown"

    def save_plate_info(self, plate_text):
        """Save the detected license plate text to a file or variable."""
        with open("detected_plate_info.txt", "a") as f:
            f.write(f"Detected License Plate: {plate_text}\n")
        print(f"Saved plate information: {plate_text}")

    def save_violation_image(self, img):
        """Save the violation image as proof."""
        violation_image_filename = f"{self.violation_image_path}/violation_{int(time.time())}.jpg"
        cv2.imwrite(violation_image_filename, img)
        return violation_image_filename

    def save_violation_info(self, violation_info):
        """Save the violation information to a JSON file."""
        violation_file = "violations.json"
        if os.path.exists(violation_file):
            with open(violation_file, "r") as file:
                violations = json.load(file)
        else:
            violations = []

        violations.append(violation_info)

        with open(violation_file, "w") as file:
            json.dump(violations, file, indent=4)
        
        print(f"Saved violation information: {violation_info}")

    def release(self):
        """Release the webcam when done."""
        self.cap.release()
