# Initialization of dependencies
import cv2
import numpy as np
from datetime import datetime
import time
from model_config import model_configurations as config


### info about the model configurations
print(
    f"Using the following model with index",
    {config[1]["index"]},
    "and name :",
    config[1]["name"],
)

# Initalization of time values
now = datetime.now()
timestr = str(now.strftime("%H:%M:%S"))
date = str(now.strftime("%d/%m/%Y"))
filename = str("results.txt")

with open(filename, "a") as f:
    f.write("\nSession: " + date + timestr + "\n")
print("Initializing Data Output")

# Camera Initialization
cam = cv2.VideoCapture(0)

# Yolo Files Initalization
folderpath = config[1]["names"]  # YOLO Name Fiile location
# folderpath = 'Models\\obj.names'                                    # YOLO Name Fiile location
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

# YOLO cfg file location
modelConfiguration = config[1]["cfg"]  # YOLO cfg file location

modelWeight = config[1]["weights"]  # YOLO weight file location


model = cv2.dnn.readNetFromDarknet(
    modelConfiguration, modelWeight
)  # Loading of YOLO Models

# To run YOLO Models on GPU
model.setPreferableBackend(
    cv2.dnn.DNN_BACKEND_CUDA
)  # Note that to use this OpenCV must be build within the GPU's Cuda Cores
model.setPreferableTarget(
    cv2.dnn.DNN_TARGET_CUDA
)  # Will automatically switch to CPU whenever unconfigured.

print("Yolo Initialization Successful")


def recordData(name):
    currnow = datetime.now()
    with open(filename, "a") as f:
        timecurr = str(currnow.strftime("%H:%M:%S"))
        f.write(str(f"{name} - ") + timecurr + "\n")


def findObjects(img):
    start_time = time.time()  # Time initialization to compute FPS

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
    model.setInput(blob)
    outputNames = model.getUnconnectedOutLayersNames()  # Used for getting output layers
    detections = model.forward(outputNames)

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    confThreshold = 0.1  # YOLO Confidence Threshold
    nmsThreshold = 0.2  # Lower value, more aggressive NMS

    for output in detections:
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

    for i in indices:
        box = bbox[i]
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = (
            f"{classNames[classIds[i]].upper()} {confs[i]:.2f}%"  # Use confs[i] here
        )
        cv2.putText(
            img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        recordData(label)  # Record detection and time in a text file

    # FPS Calculation
    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time
    cv2.putText(
        img,
        f"FPS: {round(fps, 2)}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )  # Display FPS


while True:
    success, img = cam.read()

    if not success:
        break

    imgHeight, imgWidth, channels = img.shape
    # print(img.shape) ## checking the size of video feed

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    findObjects(img)  # Calling of Object Detection Function

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Image", img)

    # To stop the script press p
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
