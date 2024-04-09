# Initialization of dependencies
import cv2
import numpy as np
import pyrealsense2 as rs
from datetime import datetime
import time

###### Note : Run the intel depth quality with the pre-set json file in the same folder ######


# Initialization of time values
now = datetime.now()
timestr = str(now.strftime("%H:%M:%S"))
date = str(now.strftime("%d/%m/%Y"))
filename = "results.txt"

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


# -------------------------------------------------------- Configuration for YOLO --------------------------------------------------------
# Define your model configurations
# model_configurations = [
#     {
#         "name": "April Model the first",
#         "cfg": "computer_vision/pyCode/models_april/first_of_april/custom-yolov4-tiny-detector.cfg",
#         "weights": "computer_vision/pyCode/models_april/first_of_april/custom-yolov4-tiny-detector_best.weights",
#         "names": "computer_vision/pyCode/models_april/first_of_april/obj.names",
#     },
#     {
#         "name": "March Model",
#         "cfg": "computer_vision/pyCode/Models_march/custom-yolov4-tiny-detector.cfg",
#         "weights": "computer_vision/pyCode/Models_march/custom-yolov4-tiny-detector_best.weights",
#         "names": "computer_vision/pyCode/Models_march/obj.names",
#     },
#     {
#         "name": "October Model",
#         "cfg": "computer_vision/pyCode/Models/custom-yolov4-tiny-detector.cfg",
#         "weights": "computer_vision/pyCode/Models/custom-yolov4-tiny-detector_best.weights",
#         "names": "computer_vision/pyCode/Models/obj.names",
#     },
# ]

# Load the YOLO model
# Yolo Files Initialization
folderpath = "computer_vision/pyCode/Models/April/eight_april/obj.names"
classNames = []
with open(folderpath, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

print("Loading Yolo Models")

modelConfiguration = (
    "computer_vision/pyCode/Models/April/eight_april/custom-yolov4-tiny-detector.cfg"
)
modelWeight = "computer_vision/pyCode/Models/April/eight_april/custom-yolov4-tiny-detector_best.weights"

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
                # Make bounding box larger
                w, h = w + 20, h + 20  # Increase size by 20 pixels on each side
                x, y = (
                    x - 10,
                    y - 10,
                )  # Adjust starting point to accommodate increased size
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    indices = np.array(indices).flatten()

    for i, idx in enumerate(indices):
        box = bbox[idx]
        x, y, w, h = box
        centerX, centerY = x + w // 2, y + h // 2
        depth = depth_frame.get_distance(centerX, centerY) * 100  # Convert depth to cm

        # Draw adjusted bounding box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Annotation for object name and distance at the top right
        annotation_text = f"{classNames[classIds[idx]].upper()} {depth:.2f}cm"
        text_offset_x = wT - 200  # Adjust based on text length
        cv2.putText(
            img,
            annotation_text,
            (text_offset_x, (i + 1) * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        # Coordinates at the bottom
        coordinates_text = f"({centerX}, {centerY})"
        cv2.putText(
            img,
            coordinates_text,
            (20, hT - (i + 1) * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Record the data
        recordData(
            f"{classNames[classIds[idx]].upper()} Center: ({centerX}, {centerY}), Depth: {depth:.2f}cm"
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
