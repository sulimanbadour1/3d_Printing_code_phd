import cv2
import numpy as np
from datetime import datetime
import time
import os

# Assuming the real object height and camera focal length are known for demonstration
REAL_OBJECT_HEIGHT = 0.4  # meters (example for a person)
CAMERA_FOCAL_LENGTH_PX = (
    800  # example value, this needs to be calibrated for your camera
)

# Initialization of time values and output file setup
now = datetime.now()
timestr = now.strftime("%H:%M:%S")
date = now.strftime("%d/%m/%Y")
filenamepath = "computer_vision/pyCode/samples/"
filename = os.path.join(os.getcwd(), filenamepath + "results.txt")

with open(filename, "a") as f:
    f.write("\nSession: " + date + " " + timestr + "\n")
print("Data output initialization complete. Writing to:", filename)

# Yolo Files Initialization
folderpath = "computer_vision/pyCode/Models/Best/4k_Dataset/obj.names"
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

modelConfiguration = (
    "computer_vision/pyCode/Models/Best/4k_Dataset/custom-yolov4-tiny-detector.cfg"
)
modelWeight = "computer_vision/pyCode/Models/Best/4k_Dataset/custom-yolov4-tiny-detector_best.weights"

model = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
print("Yolo Initialization Successful")


def calculate_distance(
    real_object_height, object_height_in_pixels, camera_focal_length_in_pixels
):
    """
    Calculate the distance to an object based on its size in the image and its real size.
    """
    distance = (
        camera_focal_length_in_pixels * real_object_height
    ) / object_height_in_pixels
    return distance


def recordData(name, x, y, w, h, distance):
    """
    Record detected object data and estimated distance.
    """
    currnow = datetime.now()
    with open(filename, "a") as f:
        timecurr = str(currnow.strftime("%H:%M:%S"))
        f.write(
            f"{name} - Coordinates: ({x}, {y}), Size: ({w}x{h}), Distance: {distance:.2f}m - {timecurr}\n"
        )
    print(f"Detected: {name} at ({x}, {y}), Size: ({w}x{h}), Distance: {distance:.2f}m")


def findObjects(img):
    start_time = time.time()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
    model.setInput(blob)
    outputNames = model.getUnconnectedOutLayersNames()
    detections = model.forward(outputNames)

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    confThreshold = 0.2
    nmsThreshold = 0

    for detection in detections:
        for det in detection:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    if isinstance(indices, tuple):
        indices = []
    else:
        indices = indices.flatten()

    for i in indices:
        x, y, w, h = bbox[i]
        label = classNames[classIds[i]].upper()
        confidence = f"{confs[i]*100:.2f}%"
        distance = calculate_distance(REAL_OBJECT_HEIGHT, h, CAMERA_FOCAL_LENGTH_PX)
        text = f"{label} {confidence} {distance:.2f}m"
        # Draw bounding box and label

        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x1 = x
        text_y1 = y - 10 - text_size[1]
        text_x2 = x + text_size[0]
        text_y2 = y - 10
        cv2.rectangle(
            img, (text_x1, text_y1), (text_x2, text_y2), (0, 0, 0), cv2.FILLED
        )
        # Choose a contrasting color like white or red for the text
        cv2.putText(
            img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
        # Draw the bounding box with increased thickness
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        recordData(label, x, y, w, h, distance)

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(
        img, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )


def analyze_image_from_path(filepath):
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Unable to load the image from the path: {filepath}")
        return

    # If the image is larger than a specific size, resize it to fit the screen while maintaining the aspect ratio.
    screen_res = 1280, 720  # Example screen resolution. Adjust as necessary.
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)

    # Resized dimensions
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    # Maintain aspect ratio by resizing the image
    img = cv2.resize(img, (window_width, window_height))

    # Convert to RGB for processing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    findObjects(img)

    # Convert back to BGR for displaying
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Create a named window that can be manually resized
    cv2.namedWindow("Detected Objects with Distance", cv2.WINDOW_NORMAL)

    # Set the window to the size of the resized image
    cv2.resizeWindow("Detected Objects with Distance", window_width, window_height)

    # Display the processed image with detections and distance estimations
    cv2.imshow("Detected Objects with Distance", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example use with a static path to an image for testing
image_filepath = r"computer_vision\pyCode\samples\img\sample_seven.jpg"
analyze_image_from_path(image_filepath)
