# Traffic Violation Detection System
from datetime import datetime

#camera system
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
#violation record 
class ViolationRecord:
    def __init__(self, recordID, vehicleNumber, violationType, dateTime, location, cameraID):
        self.recordID = recordID
        self.vehicleNumber = vehicleNumber
        self.violationType = violationType
        self.dateTime = dateTime
        self.location = location
        self.cameraID = cameraID
    def createRecord(self):
        print(f"[Record {self.recordID}] Created for vehicle {self.vehicleNumber}.")
    
    def getRecordDetails(self):
        return f"{self.dateTime} | {self.vehicleNumber} | {self.violationType} | {self.location}"

    def updateStatus(self):
        print(f"[Record {self.recordID}] Status updated.")
#vehicle 
class Vehicle:
    def __init__(self, plateNumber, ownerName, vehicleType):
        self.plateNumber = plateNumber
        self.ownerName = ownerName
        self.vehicleType = vehicleType

    def getVehicleInfo(self):
        return f"{self.plateNumber} ({self.vehicleType}) owned by {self.ownerName}"

