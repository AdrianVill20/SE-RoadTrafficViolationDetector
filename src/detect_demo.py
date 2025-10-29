import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf

# Load pre-trained model (SSD MobileNet)
print("Loading model...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
print("Model loaded successfully!")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert frame to tensor for model
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)
    img_tensor = tf.expand_dims(img_tensor, 0)

    # Run detection
    results = detector(img_tensor)
    boxes = results["detection_boxes"][0].numpy()
    classes = results["detection_classes"][0].numpy().astype(int)
    scores = results["detection_scores"][0].numpy()

    # Draw results
    h, w, _ = frame.shape
    for i in range(len(scores)):
        if scores[i] > 0.5:  # confidence threshold
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1, x2, y2 = int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Object {classes[i]} ({scores[i]:.2f})", 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    cv2.imshow("Traffic Violation Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
