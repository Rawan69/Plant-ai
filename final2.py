import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from rembg import remove
from PIL import Image
import joblib

# المسار الأساسي للملف
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# تحميل النماذج
yolo_model = YOLO(os.path.join(BASE_DIR, "model_files", "best.pt"))
classifier_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "model_files", "final_model_to_identify_plants.keras"))

# أسماء النباتات
class_names = ['apple', 'cherry', 'grapes', 'peach', 'pepper', 'potato', 'strawberry', 'tomato']

# تغيير حجم الصورة مع خلفية بيضاء
def resize_with_background(image, size=(224, 224), background_color=(255, 255, 255)):
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    image_pil.thumbnail(size, Image.Resampling.LANCZOS)
    background = Image.new('RGB', size, background_color)
    x = (size[0] - image_pil.width) // 2
    y = (size[1] - image_pil.height) // 2
    background.paste(image_pil, (x, y))
    return cv2.cvtColor(np.array(background), cv2.COLOR_BGR2RGB)

# تحسين الإضاءة
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

# تقطيع الورقة
def segment_leaf(image):
    image = cv2.resize(image, (512, 512))
    image = apply_clahe(image)
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, 450, 450)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return image * mask[:, :, np.newaxis]

# استخراج الخصائص
def extract_color_histogram(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = []
    for i in range(3):
        hist_rgb = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist_hsv = cv2.calcHist([image_hsv], [i], None, [256], [0, 256])
        features.extend(cv2.normalize(hist_rgb, hist_rgb).flatten())
        features.extend(cv2.normalize(hist_hsv, hist_hsv).flatten())
    return np.array(features)

def extract_glcm_features(image):
    from skimage.feature import graycomatrix, graycoprops
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0]
    ])

def extract_lbp_texture(image):
    from skimage.feature import local_binary_pattern
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    return hist.astype("float") / hist.sum()

def combine_features(image):
    return np.hstack([
        extract_color_histogram(image),
        extract_glcm_features(image),
        extract_lbp_texture(image)
    ])

# التنبؤ بالمرض
def predict_disease_with_svm(cropped_image, plant_type):
    bundle_dir = os.path.join(BASE_DIR, "bundles")
    bundle_path = os.path.join(bundle_dir, f"{plant_type}_bundle.pkl")
    if not os.path.exists(bundle_path):
        return "unknown"
    bundle = joblib.load(bundle_path)
    svm, pca, scaler, label_map = bundle['model'], bundle['pca'], bundle['scaler'], bundle['label_map']
    segmented = segment_leaf(cropped_image)
    features = combine_features(segmented).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    return label_map[svm.predict(features_pca)[0]]

# ---------------------- Flask API ----------------------

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image_np = np.frombuffer(file.read(), np.uint8)
    original = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if original is None:
        return jsonify({"error": "Invalid image"}), 400

    rgb_image = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    no_bg = remove(rgb_image)
    if no_bg.shape[2] == 4:
        no_bg = cv2.cvtColor(no_bg, cv2.COLOR_RGBA2RGB)
    yolo_input = cv2.resize(no_bg, (640, 640))

    results = yolo_model.predict(yolo_input, save=False)

    if results[0].masks is not None and len(results[0].masks.data) > 0:
        masks = results[0].masks.data.cpu().numpy()
        largest_mask_idx = np.argmax([np.sum(m) for m in masks])
        mask = (masks[largest_mask_idx] * 255).astype(np.uint8)

        masked = cv2.bitwise_and(yolo_input, yolo_input, mask=mask)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            return jsonify({"error": "Empty mask"}), 200

        x1, y1 = max(0, min(xs) - 5), max(0, min(ys) - 5)
        x2, y2 = min(masked.shape[1], max(xs) + 5), min(masked.shape[0], max(ys) + 5)
        crop = masked[y1:y2, x1:x2]

        final_image = resize_with_background(crop)
        img_array = image.img_to_array(final_image)
        img_batch = np.expand_dims(img_array, axis=0)
        img_batch = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)

        preds = classifier_model.predict(img_batch, verbose=0)[0]
        top3 = np.argsort(preds)[-3:][::-1]
        plant_predictions = [{"label": class_names[i], "score": float(preds[i])} for i in top3]

        try:
            plant_type = plant_predictions[0]["label"]
            disease = predict_disease_with_svm(crop, plant_type)
        except Exception as e:
            print(f"Disease prediction error: {e}")
            disease = "unknown"

        return jsonify({
            "plant_predictions": plant_predictions,
            "disease": disease
        })

    return jsonify({"error": "No plant detected", "plant_predictions": [], "disease": "unknown"}), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)
