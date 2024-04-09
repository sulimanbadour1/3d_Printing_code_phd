# Initialization of dependencies
import cv2
import numpy as np
from datetime import datetime
import time

# Initalization of time values
now = datetime.now()
timestr = str(now.strftime("%H:%M:%S"))
date = str(now.strftime("%d/%m/%Y"))
filename = str("results.txt")

with open(filename, "a") as f:
    f.write("\nSession: " + date + timestr + "\n")
print("Initializing Data Output")

# Yolo Files Initalization
folderpath = "computer_vision\pyCode\models_april\eight_april\obj.names"
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

modelConfiguration = (
    "computer_vision\pyCode\models_april\eight_april\custom-yolov4-tiny-detector.cfg"
)
modelWeight = "computer_vision\pyCode\models_april\eight_april\custom-yolov4-tiny-detector_best.weights"
model = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)

# To run YOLO Models on GPU
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

print("Yolo Initialization Successful")


def recordData(name):
    currnow = datetime.now()
    with open(filename, "a") as f:
        timecurr = str(currnow.strftime("%H:%M:%S"))
        f.write(str(f"{name} - ") + timecurr + "\n")


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

    confThreshold = 0.2
    nmsThreshold = 1

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
        i = i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
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
    elapsed_time = end_time - start_time
    fps = 1 / elapsed_time
    cv2.putText(
        img, str(round(fps, 2)), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )


def analyze_image_from_path(filepath):
    img = cv2.imread(filepath)

    # Check if the image was loaded properly
    if img is None:
        print(f"Error: Unable to load the image from the path: {filepath}")
        return

    imgHeight, imgWidth, channels = img.shape

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    findObjects(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Static path to the image
# image_filepath = "YOUR_STATIC_IMAGE_PATH_HERE"  # Replace this with your actual path
image_filepath = "computer_vision\pyCode\samples\img\sample_seven.jpg"
analyze_image_from_path(image_filepath)
