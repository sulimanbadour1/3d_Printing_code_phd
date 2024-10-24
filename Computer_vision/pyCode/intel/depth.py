# Initialization of dependencies
import cv2
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
import time

from model_config import model_configurations as config_file

###### Note : Run the intel depth quality with the pre-set json file in the same folder ######
### info about the model configurations


# Initialization of time values
now = datetime.now()
timestr = str(now.strftime("%H:%M:%S"))
date = str(now.strftime("%d/%m/%Y"))
filename = "results.txt"

print(
    f"Using the following model with index",
    {config_file[6]["index"]},
    "and name :",
    config_file[6]["name"],
)


with open(filename, "a") as f:
    f.write("\nSession: " + date + " " + timestr + "\n")
print("Initializing Data Output")

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
# Configure to stream color and depth frames
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# --------------Test for depth scale -------------------
# # Obtain the depth sensor's depth scale
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# option_range = depth_sensor.get_option_range(rs.option.depth_units)
# print(f"Depth units can be set between {option_range.min} and {option_range.max}.")
# # Set the device to High Accuracy preset for better short range accuracy
# # depth_sensor.set_option(
# #     rs.option.visual_preset, 3
# # )  # Check if '3' corresponds to High Accuracy in your camera model

# --------------Test for depth scale -------------------

# Create an align object
align_to = rs.stream.color
align = rs.align(align_to)


# Load the YOLO model
# Yolo Files Initialization
folderpath = config_file[6]["names"]
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

modelConfiguration = config_file[6]["cfg"]
modelWeight = config_file[6]["weights"]

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


def findObjects(img, depth_frame):
    start_time = time.time()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), [0, 0, 0], 1, crop=False)
    model.setInput(blob)
    outputNames = model.getUnconnectedOutLayersNames()
    detection = model.forward(outputNames)

    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    min_distance = float("inf")  # Initialize minimum distance to infinity

    confThreshold = 0.1  # Confidence threshold
    nmsThreshold = 0.1  # Non-maximum suppression threshold

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
        centerX, centerY = x + w // 2, y + h // 2
        depth = depth_frame.get_distance(centerX, centerY) * 100  # Convert depth to cm
        min_distance = min(
            min_distance, depth
        )  # Update minimum distance if current depth is smaller
        confidence_percent = confs[i] * 100  # Convert confidence to percentage

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{classNames[classIds[i]].upper()} {confidence_percent:.2f}% {depth:.2f}cm",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        recordData(f"{classNames[classIds[i]].upper()} - {depth:.2f}cm")

    # Display the minimum distance found (if any object was detected)
    if min_distance != float("inf"):
        cv2.putText(
            img,
            f"Min Distance: {min_distance:.2f}cm",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

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
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    img = np.asanyarray(color_frame.get_data())

    findObjects(img, depth_frame)  # Object detection with depth

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


pipeline.stop()
