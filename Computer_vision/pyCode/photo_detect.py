# Initialization of dependencies
import cv2
import numpy as np
from datetime import datetime
import time
from model_config import model_configurations as config


### info about the model configurations
print("Number of models available:", len(config))
print(
    f"Using the following model with index",
    {config[5]["index"]},
    "and name :",
    config[5]["name"],
)

# Initalization of time values
now = datetime.now()
timestr = str(now.strftime("%H:%M:%S"))
date = str(now.strftime("%d/%m/%Y"))
filename = str("results.txt")

with open(filename, "a") as f:
    f.write("\nSession: " + date + timestr + "\n")
print("Initializing Data Output")

# Yolo Files Initalization
folderpath = config[5]["names"]
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

modelConfiguration = config[5]["cfg"]

modelWeight = config[5]["weights"]
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
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
    model.setInput(blob)
    outputs = model.forward(model.getUnconnectedOutLayersNames())

    height, width, _ = img.shape
    bbox = []
    classIds = []
    confs = []

    confThreshold = 0.1
    nmsThreshold = 0

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * width), int(det[3] * height)
                x, y = int((det[0] * width) - w / 2), int((det[1] * height) - h / 2)
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
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{classNames[classIds[i]].upper()} {confs[i]:.2f}"
        cv2.putText(
            img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        recordData(label)


def analyze_image_from_path(filepath):
    img = cv2.imread(filepath)

    # Check if the image was loaded properly
    if img is None:
        print(f"Error: Unable to load the image from the path: {filepath}")
        return

    # Resize the image to the fixed dimensions (416x416)
    img = cv2.resize(img, (800, 800))

    # Convert the image to RGB (from BGR, which is OpenCV's default color format)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect objects in the image
    findObjects(img)

    # Convert the image back to BGR for display purposes
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Display the resulting image
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Static path to the image
# image_filepath = "YOUR_STATIC_IMAGE_PATH_HERE"  # Replace this with your actual path
image_filepath = r"computer_vision\pyCode\samples\img\sample_seven.jpg"
analyze_image_from_path(image_filepath)
