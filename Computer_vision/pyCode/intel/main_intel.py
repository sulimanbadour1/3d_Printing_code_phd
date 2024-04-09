# Initialization of dependencies
import cv2
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
import time

# Initialization of time values
now = datetime.now()
timestr = str(now.strftime("%H:%M:%S"))
date = str(now.strftime("%d/%m/%Y"))
filename = str("results.txt")

with open(filename, "a") as f:
    f.write("\nSession: " + date + " " + timestr + "\n")
print("Initializing Data Output")

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
# Configure to stream color frames (adjust resolution as needed)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Yolo Files Initialization
folderpath = "computer_vision/pyCode/Models/Best/36_epoch/obj.names"
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

modelConfiguration = (
    "computer_vision/pyCode/Models/Best/36_epoch/custom-yolov4-tiny-detector.cfg"
)
modelWeight = "computer_vision/pyCode/Models/Best/36_epoch/custom-yolov4-tiny-detector_best.weights"


# Load the neural network
model = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)

# To run YOLO Models on GPU
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print("Yolo Initialization Successful")


def recordData(name):
    currnow = datetime.now()
    with open(filename, "a") as f:
        timecurr = str(currnow.strftime("%H:%M:%S"))
        f.write(f"{name} - {timecurr}\n")


def findObjects(img):
    start_time = time.time()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
    model.setInput(blob)
    outputNames = model.getUnconnectedOutLayersNames()
    detection = model.forward(outputNames)

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    confThreshold = 0.4
    nmsThreshold = 0.5

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
    indices = np.array(indices).flatten()

    for i in indices:
        box = bbox[i]
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.putText(
            img,
            f"{classNames[classIds[i]].upper()}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
        )
        recordData(classNames[classIds[i]].upper())

    end_time = time.time()
    fps = 1 / (end_time - start_time)
    cv2.putText(
        img,
        f"FPS: {str(round(fps, 2))}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )


while True:
    # Wait for a coherent pair of frames (color and depth)
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # Convert images to numpy arrays
    img = np.asanyarray(color_frame.get_data())

    findObjects(img)  # Object detection

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

pipeline.stop()
