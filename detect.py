import pytesseract
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load an image
image_path = 'strawberry_icecream.png'
image = Image.open(image_path)

# Text Detection
text = pytesseract.image_to_string(image)
print("Detected Text:", text)

# Font and Color Detection
def detect_font_and_color(image):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = image.getcolors(maxcolors=1000000)
    most_common_color = max(colors, key=lambda item: item[0])[1]
    return font, most_common_color

font, color = detect_font_and_color(image)
print("Detected Font:", font)
print("Detected Color:", color)

# Image Recognition using TensorFlow
def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def recognize_image(model, image):
    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1]

model = load_model()
preprocessed_image = preprocess_image(image_path)
recognized_object = recognize_image(model, preprocessed_image)
print("Recognized Object:", recognized_object)

# YOLO Object Detection
def load_yolo():
    net = cv2.dnn.readNet("yolo_files/yolov3.weights", "yolo_files/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open("yolo_files/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

def detect_objects_yolo(image_path, net, output_layers, classes):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    results = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            results.append((label, x, y, w, h))
    return results

net, output_layers, classes = load_yolo()
yolo_results = detect_objects_yolo(image_path, net, output_layers, classes)
print("YOLO Detection Results:", yolo_results)

# Color Detection and Pie Chart
def detect_main_colors(image_path, num_colors=5):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = np.bincount(labels)
    return colors, counts

def plot_colors(colors, counts):
    fig, ax = plt.subplots()
    ax.pie(counts, labels=[f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for c in colors], colors=[c/255 for c in colors])
    plt.show()

colors, counts = detect_main_colors(image_path)
plot_colors(colors, counts)
