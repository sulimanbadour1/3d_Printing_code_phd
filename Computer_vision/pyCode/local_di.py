import cv2
import numpy as np
from datetime import datetime
import time
from model_config import model_configurations as config
import os
import sys

### info about the model configurations
print(
    f"Using the following model with index",
    {config[8]["index"]},
    "and name :",
    config[8]["name"],
)

# Initialization of time values
now = datetime.now()
timestr = str(now.strftime("%H:%M:%S"))
date = str(now.strftime("%d/%m/%Y"))
filename = "localvideo_results.txt"  # Updated filename for local video results

with open(filename, "a") as f:
    f.write(
        "\nSession: " + date + " " + timestr + "\n"
    )  # Fixed space between date and time
print("Initializing Data Output")

# Load local video instead of camera
video_path = "downloaded_videos/vid2.mp4"  # Provide the path to your video file here
if not os.path.exists(video_path):
    print("Video file not found. Please provide the correct path.")
    sys.exit(1)


cam = cv2.VideoCapture(video_path)  # Updated to load video file

# Yolo Files Initialization (assuming the paths are correctly specified for your environment)
folderpath = config[8]["names"]  # YOLO Name Fiile location
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

# Yolo cfg file location
modelConfiguration = config[8]["cfg"]  # YOLO cfg file location

modelWeight = config[8]["weights"]  # YOLO weight file location


# Load the neural network
model = cv2.dnn.readNetFromDarknet(
    modelConfiguration, modelWeight
)  # Loading of YOLO Models

# To run YOLO Models on GPU (make sure your OpenCV is configured with CUDA support)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print("Yolo Initialization Successful")


def recordData(name, distance_cm, confidence, bbox, inference_time):
    currnow = datetime.now()
    x, y, w, h = bbox
    with open(filename, "a") as f:
        timecurr = str(currnow.strftime("%H:%M:%S"))
        # Record data with added inference time in milliseconds
        f.write(
            f"{timecurr} - {name} - Conf: {confidence*100:.2f}% - Inference: {inference_time:.2f}ms - Dist: {distance_cm:.2f}cm - Box: ({x}, {y}, {w}, {h}) \n"
        )


def findObjects(img):
    start_time = time.time()  # Time initialization to compute for FPS

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
    model.setInput(blob)
    outputNames = model.getUnconnectedOutLayersNames()  # Used for getting output layers

    # Time the inference
    inference_start = time.time()  # Start timing the inference process
    detection = model.forward(outputNames)
    inference_end = time.time()  # End timing after inference is done

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    confThreshold = 0.1  # YOLO Confidence Threshold
    nmsThreshold = 0.3  # lower the more aggressive and fewer boxes

    for output in detection:
        for det in output:
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
    indices = np.array(indices).flatten()  # Array list of detected objects

    inference_time = (inference_end - inference_start) * 1000  # Convert to milliseconds

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        name = classNames[classIds[i]].upper()
        label = f"{name} {confs[i]*100:.2f}%"

        # Estimating distance based on the size of the bounding box (assuming a certain real size)
        focal_length = 800  # Example focal length
        known_width = 0.1  # Known width of the object in meters
        distance = (known_width * focal_length) / w
        distance_cm = distance * 100  # Convert to centimeters

        label += f" Distance: {distance_cm:.2f}cm"

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(
            img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3
        )

        recordData(
            name, distance_cm, confs[i], [x, y, w, h], inference_time
        )  # Record data with inference time

    # FPS Calculation
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time
    cv2.putText(
        img, f"FPS: {fps:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )


# Main loop to process the video file
while True:
    success, img = cam.read()
    if not success:
        break  # If no frame is read (end of video), exit the loop

    imgHeight, imgWidth, channels = img.shape
    # print(img.shape) ## checking the size of video feed

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    findObjects(img)  # Calling of Object Detection Function

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (800, 500))

    # Display the frame and check for 'q' press to exit early
    cv2.imshow("Image", img)
    if cv2.waitKey(40) & 0xFF == ord("q"):
        break

cam.release()  # Release the video file
cv2.destroyAllWindows()  # Close all OpenCV windows
