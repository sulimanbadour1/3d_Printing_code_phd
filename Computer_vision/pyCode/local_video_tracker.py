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
    {config[6]["index"]},
    "and name :",
    config[6]["name"],
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
video_path = "downloaded_videos/vid.mp4"  # Provide the path to your video file here
if not os.path.exists(video_path):
    print("Video file not found. Please provide the correct path.")
    sys.exit(1)

cam = cv2.VideoCapture(video_path)  # Updated to load video file

# Yolo Files Initialization (assuming the paths are correctly specified for your environment)
folderpath = config[6]["names"]  # YOLO Name File location
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

# Yolo cfg file location
modelConfiguration = config[6]["cfg"]  # YOLO cfg file location
modelWeight = config[6]["weights"]  # YOLO weight file location

# Load the neural network
model = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
print("Yolo Initialization Successful")


# Helper function to initialize Kalman Filters
def initialize_kalman():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kf.processNoiseCov = (
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 5]], np.float32)
        * 0.03
    )
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10
    return kf


# Kalman Filters storage
kalman_filters = {}
kf_id = 0


def recordData(name):
    currnow = datetime.now()
    with open(filename, "a") as f:
        timecurr = str(currnow.strftime("%H:%M:%S"))
        f.write(str(f"{name} - ") + timecurr + "\n")


def findObjects(img):
    start_time = time.time()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
    model.setInput(blob)
    outputNames = model.getUnconnectedOutLayersNames()
    detections = model.forward(outputNames)

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    confThreshold = 0.1
    nmsThreshold = 0.5

    global kf_id, kalman_filters

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
    indices = np.array(indices).flatten()

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        if i not in kalman_filters:
            kf = initialize_kalman()
            kalman_filters[i] = kf
            kf.statePost = np.array([x, y, 0, 0], dtype=np.float32)
        kf = kalman_filters[i]
        kf.correct(np.array([x, y], dtype=np.float32))
        prediction = kf.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        cv2.rectangle(img, (pred_x, pred_y), (pred_x + w, pred_y + h), (0, 255, 0), 5)
        label = f"{classNames[classIds[i]].upper()} {confs[i]*100:.2f}%"
        cv2.putText(
            img,
            label,
            (pred_x, pred_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
        )
        recordData(classNames[classIds[i]].upper())

    elapsed_time = time.time() - start_time
    fps = 1 / elapsed_time
    cv2.putText(
        img, str(round(fps, 2)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )


# Main loop to process the video file
while True:
    success, img = cam.read()
    if not success:
        break  # If no frame is read (end of video), exit the loop

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    findObjects(img)  # Calling of Object Detection Function
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", img)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cam.release()  # Release the video file
cv2.destroyAllWindows()  # Close all OpenCV windows
