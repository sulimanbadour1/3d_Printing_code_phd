import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

# Define your model configurations
model_configurations = [
    {
        "name": "April Best 4k dataset",
        "cfg": "computer_vision/pyCode/Models/Best/4k_Dataset/custom-yolov4-tiny-detector.cfg",
        "weights": "computer_vision/pyCode/Models/Best/4k_Dataset/custom-yolov4-tiny-detector_best.weights",
        "names": "computer_vision/pyCode/Models/Best/4k_Dataset/obj.names",
    },
    {
        "name": "April 36K Epochs",
        "cfg": "computer_vision/pyCode/Models/Best/36_epoch/custom-yolov4-tiny-detector.cfg",
        "weights": "computer_vision/pyCode/Models/Best/36_epoch/custom-yolov4-tiny-detector_best.weights",
        "names": "computer_vision/pyCode/Models/Best/36_epoch/obj.names",
    },
    {
        "name": "April Model eight",
        "cfg": "computer_vision/pyCode/Models/April/eight_april/custom-yolov4-tiny-detector.cfg",
        "weights": "computer_vision/pyCode/Models/April/eight_april/custom-yolov4-tiny-detector_best.weights",
        "names": "computer_vision/pyCode/Models/April/eight_april/obj.names",
    },
    {
        "name": "April Model the first",
        "cfg": "computer_vision/pyCode/Models/April/first_of_april/custom-yolov4-tiny-detector.cfg",
        "weights": "computer_vision/pyCode/Models/April/first_of_april/custom-yolov4-tiny-detector_best.weights",
        "names": "computer_vision/pyCode/Models/April/first_of_april/obj.names",
    },
    {
        "name": "March Model",
        "cfg": "computer_vision/pyCode/Models/March/custom-yolov4-tiny-detector.cfg",
        "weights": "computer_vision/pyCode/Models/March/custom-yolov4-tiny-detector_best.weights",
        "names": "computer_vision/pyCode/Models/March/obj.names",
    },
    {
        "name": "October Model",
        "cfg": "computer_vision/pyCode/Models/October/custom-yolov4-tiny-detector.cfg",
        "weights": "computer_vision/pyCode/Models/October/custom-yolov4-tiny-detector_best.weights",
        "names": "computer_vision/pyCode/Models/October/obj.names",
    },
]


def load_model(cfg_file, weights_file, names_file):
    net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
    with open(names_file, "rt") as f:
        classes = f.read().rstrip("\n").split("\n")  # Corrected "/n" to "\n"
    return net, classes


def detect_objects(model, img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers_names = model.getUnconnectedOutLayersNames()
    outputs = model.forward(output_layers_names)
    return outputs


def post_process(
    outputs, classes, img_width, img_height, confidence_threshold=0.1, nms_threshold=0.1
):
    boxes = []
    confidences = []
    class_ids = []
    detected_class_names = []  # List to store detected class names
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                w = int(detection[2] * img_width)
                h = int(detection[3] * img_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_class_names.append(classes[class_id])

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(indices) == 0:  # Check if no boxes were kept
        return [], [], [], []

    # Fix for handling different OpenCV versions output
    if type(indices) is tuple:  # Handling OpenCV output as tuple
        indices = indices[0]

    indices = np.array(indices).flatten()  # Ensuring it is flat

    final_boxes = [boxes[i] for i in indices]
    final_confidences = [confidences[i] for i in indices]
    final_class_ids = [class_ids[i] for i in indices]
    final_detected_class_names = [detected_class_names[i] for i in indices]

    return final_boxes, final_confidences, final_class_ids, final_detected_class_names


def analyze_model_performance(model_config, test_images):
    model, classes = load_model(
        model_config["cfg"], model_config["weights"], model_config["names"]
    )
    print(f"/nAnalyzing {model_config['name']}...")

    results = {
        "inference_times": [],
        "detection_counts": [],
        "detected_classes": set(),
        "detection_times": [],
    }
    detected_images = []  # To store images with detections for comparison

    for image_path in test_images:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image {image_path}. Check file path and integrity.")
            continue
        start_time = time.time()
        outputs = detect_objects(model, img)
        detection_time = time.time() - start_time
        boxes, confidences, class_ids, detected_class_names = post_process(
            outputs, classes, img.shape[1], img.shape[0]
        )
        results["inference_times"].append(detection_time)
        results["detection_counts"].append(len(boxes))
        results["detected_classes"].update(detected_class_names)
        results["detection_times"].append(detection_time)

        img_with_detections = img.copy()
        draw_boxes(img_with_detections, boxes, confidences, class_ids, classes)
        detected_images.append(img_with_detections)  # Add image with detections

    return results, detected_images


def draw_boxes(img, boxes, confidences, class_ids, classes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def display_detected_images(detected_images):
    fig, axs = plt.subplots(1, len(detected_images), figsize=(20, 5))
    for i, img in enumerate(detected_images):
        if len(detected_images) > 1:  # More than one image, use indexed axs
            axs[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[i].axis("off")
        else:  # Single image, axs is not subscriptable
            axs.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs.axis("off")
    plt.show()


def generate_plots(results):
    model_names = list(results.keys())
    inference_times = [
        np.mean(results[name]["inference_times"]) for name in model_names
    ]
    detection_counts = [sum(results[name]["detection_counts"]) for name in model_names]
    unique_classes_counts = [
        len(results[name]["detected_classes"]) for name in model_names
    ]

    # Plot Inference Times
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, inference_times, color="skyblue")
    plt.xlabel("Model")
    plt.ylabel("Average Inference Time (seconds)")
    plt.title("Inference Time Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Detection Counts
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, detection_counts, color="lightgreen")
    plt.xlabel("Model")
    plt.ylabel("Total Detections")
    plt.title("Detection Counts Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot Unique Detected Classes
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, unique_classes_counts, color="salmon")
    plt.xlabel("Model")
    plt.ylabel("Number of Unique Detected Classes")
    plt.title("Unique Detected Classes Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def compare_models_and_document(test_images):
    all_detected_images = (
        []
    )  # Store detected images from all models for side-by-side comparison
    results = {}
    for config in model_configurations:
        model_result, detected_images = analyze_model_performance(config, test_images)
        results[config["name"]] = model_result
        all_detected_images.extend(
            detected_images
        )  # Assuming one image per model for simplicity

    # Generate plots based on collected data
    generate_plots(results)

    display_detected_images(
        all_detected_images
    )  # Display all detected images side by side

    with open(
        "computer_vision/pyCode/models_compare/model_comparison_results.txt", "w"
    ) as file:
        for model_name, data in results.items():
            file.write(f"Model: {model_name}\n")
            file.write(
                f"Average Inference Time: {np.mean(data['inference_times']):.4f} seconds\n"
            )
            file.write(f"Total Detections: {sum(data['detection_counts'])}\n")
            file.write(
                f"Detection Times: {', '.join([f'{t:.4f}s' for t in data['detection_times']])}\n"
            )
            file.write(
                "Detected Classes: " + ", ".join(data["detected_classes"]) + "\n\n"
            )


test_images = ["computer_vision/pyCode/models_compare/img/five.jpg"]

compare_models_and_document(test_images)
