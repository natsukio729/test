from flask import Flask, request, render_template, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_cors import CORS
import openai
import pytesseract
from PIL import Image, ImageFont, ImageDraw
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from colorthief import ColorThief
import io
import base64
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)
CORS(app)

photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
configure_uploads(app, photos)

# Load YOLO
net = cv2.dnn.readNet("yolo_files/yolov3.weights", "yolo_files/yolov3.cfg")
with open("yolo_files/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['photo']
    image_path = 'uploaded_image.png'
    file.save(image_path)

    # Load and process image
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    font, color = detect_font_and_color(image)
    recognized_object = recognize_image(model, preprocess_image(image_path))
    yolo_result = detect_objects_yolo(image_path, net, output_layers, classes)
    colors, counts = detect_main_colors(image_path)
    pie_chart = plot_colors(colors, counts)

    # OpenAI GPT-3 Integration
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=f"Detected text: {text}\nDetected font: {font}\nDetected color: {color}\nRecognized object: {recognized_object}\nYOLO detection results: {yolo_result}\n\nProvide a summary:",
        max_tokens=100
    )
    ai_summary = response.choices[0].text.strip()

    return render_template('result.html', text=text, font=font, color=color, recognized_object=recognized_object, yolo_result=yolo_result, pie_chart=pie_chart, ai_summary=ai_summary)

def detect_font_and_color(image):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = image.getcolors(maxcolors=1000000)
    most_common_color = max(colors, key=lambda item: item[0])[1]
    return font, most_common_color

def load_model():
    model = tf.keras.applications.MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def recognize_image(model, image):
    predictions = model.predict(image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)
    return decoded_predictions[0][0][1]

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
    wedges, texts = ax.pie(counts, colors=[f'#{r:02x}{g:02x}{b:02x}' for r, g, b in colors], startangle=90)
    ax.axis('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded_image

model = load_model()

if __name__ == '__main__':
    app.run(debug=True)
