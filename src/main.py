# Traffic Violation Detection System
from datetime import datetime

class CameraSystem:
    def __init__(self, cameraID, location, status="Active"):
        self.cameraID = cameraID
        self.location = location
        self.status = status

    def captureImage(self):
        print(f"[Camera {self.cameraID}] Capturing image at {self.location}...")
        return f"Image_from_{self.location}"

    def detectViolation(self):
        # Simulate random violation detection
        print(f"[Camera {self.cameraID}] Checking for violations...")
        return True  # for demo, always detects a violation

    def sendData(self):
        print(f"[Camera {self.cameraID}] Sending data to database manager.")

class ViolationRecord:
    def __init__(self, recordID, vehicleNumber, violationType, dateTime, location, cameraID):
        self.recordID = recordID
        self.vehicleNumber = vehicleNumber
        self.violationType = violationType
        self.dateTime = dateTime
        self.location = location
        self.cameraID = cameraID