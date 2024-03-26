import cv2
import numpy as np
from datetime import datetime
import time
import os
from pytube import YouTube


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
    timestr = currnow.strftime("%H:%M:%S")
    with open(filename, "a") as f:
        f.write(
            f"{name} - Coordinates: ({x}, {y}), Size: ({w}x{h}), Distance: {distance if distance is not None else 'N/A'}m - {timestr}\n"
        )


# Initialization of time values and output file setup
now = datetime.now()
datestr = now.strftime("%Y-%m-%d")
timestr = now.strftime("%H-%M-%S")
filenamepath = "results/"
filename = os.path.join(filenamepath, f"results_{datestr}_{timestr}.txt")

if not os.path.exists(filenamepath):
    os.makedirs(filenamepath)

with open(filename, "w") as f:
    f.write("Detection Log\n")
    f.write(f"Session started: {datestr} {timestr}\n")
print("Data output initialization complete. Writing to:", filename)

# Update these paths according to your setup.
# Yolo Files Initalization
folderpath = "computer_vision\pyCode\Models\obj.names"
# folderpath = 'Models\\obj.names'                                    # YOLO Name Fiile location
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

# modelConfiguration = "Models\\custom-yolov4-tiny-detector.cfg"  # YOLO cfg file location
modelConfiguration = "computer_vision\pyCode\Models\custom-yolov4-tiny-detector.cfg"
# modelWeight = (
#     "Models\\custom-yolov4-tiny-detector_best.weights"  # YOLO weight file location
# )
modelWeight = (
    "computer_vision\pyCode\Models_march\custom-yolov4-tiny-detector_final.weights"
)
# Load the neural network
model = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
print("Yolo Initialization Successful")


def findObjects(img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
    model.setInput(blob)
    outputNames = model.getUnconnectedOutLayersNames()
    detections = model.forward(outputNames)

    hT, wT, _ = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in detections:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.5:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int(det[0] * wT - w / 2), int(det[1] * hT - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, 0.5, 0.4)
    for i in indices:
        i = i
        box = bbox[i]
        x, y, w, h = box
        label = str(classNames[classIds[i]])
        distance = calculate_distance(0.1, h, 800)  # Example values: adjust accordingly
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            img,
            f"{label} {round(distance, 2)}m",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )
        recordData(label, x, y, w, h, round(distance, 2))


# https://youtu.be/Vqyc0dLeTaU?si=S0ZtMpH0uwVKHZKS  Spaghetti
# https://www.youtube.com/watch?v=tX6qY3oPN34       Cracks
# https://youtu.be/J_h24R-ytqg?si=oI_MOZhNmQI6CFMw  Stringging
# YouTube video URL
youtube_url = "https://youtu.be/6V319bseO3U?si=gdQupnx9X5Eqo3MX"


# Use pytube to get video info
youtube = YouTube(youtube_url)
# Get the best resolution progressive stream (contains both audio and video)
video = youtube.streams.get_highest_resolution()

# Now, use the direct URL for cv2.VideoCapture
cam = cv2.VideoCapture(video.url)

while True:
    success, img = cam.read()
    if not success:
        print("Failed to grab frame")
        break

    # Object detection and other processing...
    findObjects(img)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
