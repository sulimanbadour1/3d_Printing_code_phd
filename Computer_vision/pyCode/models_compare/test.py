import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


def load_model(cfg_file, weights_file, names_file):
    net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)
    with open(names_file, "rt") as f:
        classes = f.read().rstrip("\n").split("\n")
    return net, classes


def detect_objects(model, img):
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    output_layers_names = model.getUnconnectedOutLayersNames()
    outputs = model.forward(output_layers_names)
    return outputs


def post_process(
    outputs, classes, img_width, img_height, confidence_threshold=0.5
):  # Changed confidence_threshold from 0.5 to 0.3
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
                detected_class_names.append(
                    classes[class_id]
                )  # Add class name to the list
                center_x = int(detection[0] * img_width)
                center_y = int(detection[1] * img_height)
                w = int(detection[2] * img_width)
                h = int(detection[3] * img_height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids, detected_class_names


def draw_boxes(img, boxes, confidences, class_ids, classes):
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def plot_inference_times(times, model_names):
    plt.figure(figsize=(10, 6))
    colors = ["skyblue", "lightgreen", "salmon"]
    bars = plt.bar(model_names, times, color=colors)

    plt.xlabel("Model", fontsize=14, fontweight="bold")
    plt.ylabel("Inference Time (seconds)", fontsize=14, fontweight="bold")
    plt.title("YOLO Model Inference Time Comparison", fontsize=16, fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Annotate each bar with inference time
    for bar, time in zip(bars, times):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{time:.2f}s",
            ha="center",
            va="bottom",
        )

    # Annotate each bar with model details
    # for bar, detail in zip(bars):
    #     plt.text(
    #         bar.get_x() + bar.get_width() / 2.0,
    #         0,
    #         detail,
    #         ha="center",
    #         va="top",
    #         rotation=90,
    #         fontsize=9,
    #         color="dimgrey",
    #     )

    plt.tight_layout()
    plt.show()


def plot_detection_counts(detection_counts, model_names):
    plt.figure(figsize=(10, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    plt.bar(model_names, detection_counts, color=colors)

    plt.xlabel("Model", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Detections", fontsize=14, fontweight="bold")
    plt.title("Comparison of YOLO Model Detections", fontsize=16, fontweight="bold")

    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, count in enumerate(detection_counts):
        plt.text(i, count + 0.5, f"{count}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()


############ Summary ############
# The compare_models function is used to compare the performance of three YOLO models.


def summarize_results(
    inference_times, detection_counts, model_names, test_images, classes_detected
):
    plt.figure(figsize=(12, 8))

    # Plotting inference times
    bars = plt.bar(
        model_names, inference_times, color=["skyblue", "lightgreen", "salmon"]
    )

    plt.xlabel("Model", fontsize=14, fontweight="bold")
    plt.ylabel("Inference Time (seconds)", fontsize=14, fontweight="bold")
    plt.title(
        f"YOLO Model Performance Summary for {len(test_images)} Image(s)",
        fontsize=16,
        fontweight="bold",
    )

    # Annotating bars with inference times
    for bar, time in zip(bars, inference_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            time,
            f"{time:.2f}s",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Annotating bars with detection counts
    max_height = max(inference_times)
    for i, (bar, count) in enumerate(zip(bars, detection_counts)):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            max_height * 1.05,
            f"{count} detections",
            ha="center",
            va="bottom",
            fontsize=10,
            color="dimgrey",
        )

    # Displaying detected class names under each bar
    for i, model in enumerate(model_names):
        detected_classes = ", ".join(classes_detected.get(model, []))
        plt.text(
            bars[i].get_x() + bars[i].get_width() / 2.0,
            -0.02,
            detected_classes,
            ha="center",
            va="top",
            fontsize=9,
            color="darkblue",
            rotation=45,
        )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


############ Summary ############
def compare_models():
    model_paths = [
        (
            "computer_vision/pyCode/models_april/custom-yolov4-tiny-detector.cfg",
            "computer_vision/pyCode/models_april/custom-yolov4-tiny-detector_best.weights",
            "computer_vision/pyCode/models_april/obj.names",
        ),
        (
            "computer_vision/pyCode/Models_march/custom-yolov4-tiny-detector.cfg",
            "computer_vision/pyCode/Models_march/custom-yolov4-tiny-detector_best.weights",
            "computer_vision/pyCode/Models_march/obj.names",
        ),
        (
            "computer_vision/pyCode/Models/custom-yolov4-tiny-detector.cfg",
            "computer_vision/pyCode/Models/custom-yolov4-tiny-detector_best.weights",
            "computer_vision/pyCode/Models/obj.names",
        ),
    ]

    test_images = [
        "computer_vision/pyCode/models_compare/img/two.png",
    ]
    inference_times = []
    model_names = []
    detection_counts = []
    classes_detected = {}  # Dictionary to store detected classes for each model

    for cfg_file, weights_file, names_file in model_paths:
        model, classes = load_model(cfg_file, weights_file, names_file)
        model_name = cfg_file.split("/")[-1].split(".")[0]  # Extract model name
        model_names.append(model_name)
        total_time = 0
        total_detections = 0
        all_detected_classes = []  # List to store all detected classes for each model

        for image_path in test_images:
            img = cv2.imread(image_path)
            if img is None:
                print(
                    f"Error loading image {image_path}. Check file path and integrity."
                )
                continue
            img_width, img_height = img.shape[1], img.shape[0]

            start_time = time.time()
            outputs = detect_objects(model, img)
            end_time = time.time()

            inference_time = end_time - start_time
            total_time += inference_time

            boxes, confidences, class_ids, detected_class_names = post_process(
                outputs, classes, img_width, img_height
            )
            total_detections += len(boxes)
            all_detected_classes.extend(detected_class_names)  # Add detected classes

            img_with_detections = img.copy()
            draw_boxes(img_with_detections, boxes, confidences, class_ids, classes)
            cv2.imshow(model_name, img_with_detections)

        avg_time = total_time / len(test_images)
        inference_times.append(avg_time)
        detection_counts.append(total_detections / len(test_images))

        classes_detected[model_name] = list(
            set(all_detected_classes)
        )  # Store unique classes detected

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        model_names = ["April", "March", "October"]

    # with open(
    #     "computer_vision/pyCode/models_compare/detection_results.txt", "w"
    # ) as file:
    #     file.write(
    #         f"Detected in {image_path.split('/')[-1]}: {', '.join(detected_class_names)}\n"
    #     )

    plot_inference_times(
        inference_times,
        model_names,
    )
    plot_detection_counts(detection_counts, model_names)

    # plot_inference_times(inference_times, model_names)
    summarize_results(
        inference_times, detection_counts, model_names, test_images, classes_detected
    )


compare_models()
