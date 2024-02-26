import cv2
import pyautogui
import numpy as np
import torch
import time

# Load YOLOv5
model = 'yolov5s.pt'  # Path to the YOLOv5 pre-trained weights
weights = f'yolov5/{model}'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=True)

# Load COCO class labels
classes = model.module.names if hasattr(model, 'module') else model.names

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Set the screen region to capture
region = (0, 0, screen_width, screen_height)

# OpenCV window
window_name = 'Real-time Object Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Variables for frame rate calculation
start_time = time.time()
frame_counter = 0

# Variable for person count
person_count = 0

while True:
    # Capture the screen
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(frame_rgb)

    # Reset person count for each frame
    person_count = 0

    # Draw bounding boxes on the frame
    if results.pred is not None:
        for det in results.pred[0]:
            x1, y1, x2, y2, conf, class_id = det.tolist()
            if conf > 0.4:  # Confidence threshold
                if int(class_id) == 0:  # Check if the detected object is a person
                    person_count += 1
                    cv2.rectangle(frame_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame_rgb, f'{classes[int(class_id)]} {conf:.2f}', (int(x1), int(y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow(window_name, frame_rgb)

    # Update frame rate
    frame_counter += 1
    if time.time() - start_time >= 1:
        print(f"FPS: {frame_counter / (time.time() - start_time):.2f}")
        print(f"Person count: {person_count}")
        frame_counter = 0
        start_time = time.time()

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
