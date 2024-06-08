import cv2
from ultralytics import YOLO, solutions

# Load YOLO model
model = YOLO("heatmaps/best.pt")

# Open video file
cap = cv2.VideoCapture("downloaded_videos/vid.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Initialize video writer
video_writer = cv2.VideoWriter(
    "heatmap_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Initialize heatmap object
heatmap_obj = solutions.Heatmap(
    colormap=cv2.COLORMAP_PARULA,
    view_img=True,
    shape="circle",
    classes_names=model.names,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    # Perform tracking on the current frame
    tracks = model.track(im0, persist=True, show=False)

    # Generate heatmap on the frame
    im0 = heatmap_obj.generate_heatmap(im0, tracks)

    # Write the frame to the output video
    video_writer.write(im0)

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()
