import cv2
import numpy as np
from datetime import datetime
import time
import os

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
folderpath = "computer_vision/pyCode/Models/obj.names"
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
print("Loading Yolo Models")

modelConfiguration = "computer_vision/pyCode/Models/custom-yolov4-tiny-detector.cfg"
modelWeight = "computer_vision/pyCode/Models/custom-yolov4-tiny-detector_best.weights"
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


def recordData(name, x, y, w, h, distance=None):
    """
    Record detected object data and estimated distance.
    """
    currnow = datetime.now()
    with open(filename, "a") as f:
        timecurr = str(currnow.strftime("%H:%M:%S"))
        f.write(
            f"{name} - Coordinates: ({x}, {y}), Size: ({w}x{h}), Distance: {distance if distance is not None else 'N/A'}m - {timecurr}\n"
        )
    print(
        f"Detected: {name} at ({x}, {y}), Size: ({w}x{h}), Distance: {distance if distance is not None else 'N/A'}m"
    )


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

    confThreshold = 0.3
    nmsThreshold = 0.5

    for output in detections:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x, center_y = int(det[0] * wT), int(det[1] * hT)
                width, height = int(det[2] * wT), int(det[3] * hT)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Ensure that the bounding box does not exceed the image boundaries
                x, y = max(x, 0), max(y, 0)
                width, height = min(width, wT - x), min(height, hT - y)

                bbox.append([x, y, width, height])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices.flatten():
        x, y, w, h = bbox[i]
        object_height_in_pixels = h
        # Example real object height and camera focal length (adjust these values based on your setup)
        real_object_height = 0.1  # meters
        camera_focal_length_in_pixels = 800
        distance = calculate_distance(
            real_object_height, object_height_in_pixels, camera_focal_length_in_pixels
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{classNames[classIds[i]].upper()} {round(distance, 2)}m",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        recordData(classNames[classIds[i]].upper(), x, y, w, h, round(distance, 2))

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time
    cv2.putText(
        img,
        f"FPS: {round(fps, 2)}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def analyze_image_from_path(filepath):
    img = cv2.imread(filepath)
    if img is None:
        print(f"Error: Unable to load the image from the path: {filepath}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    findObjects(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Display the processed image with detections and distance estimations
    cv2.imshow("Detected Objects with Distance", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the image with detections to disk
    result_img_path = os.path.join(filenamepath, "detected_objects.jpg")
    cv2.imwrite(result_img_path, img)
    print(f"Image saved to {result_img_path}")


# Example use with a static path to an image for testing
image_filepath = "computer_vision/pyCode/samples/pic (4).jpg"
analyze_image_from_path(image_filepath)
