from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load models
models = {
    "potato": tf.keras.models.load_model('models/potato_model.keras'),
    "tomato": tf.keras.models.load_model('models/tomato_model.keras')
}

# Class names for each crop
class_names = {
    "potato": ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
    "tomato": ['Tomato___Early_blight', 'Tomato___Late_blight',
               'Tomato___Septoria_leaf_spot', 'Tomato___healthy']
}

# Image properties
IMAGE_SIZE = 224
resizing_layer = tf.keras.layers.Resizing(height=IMAGE_SIZE, width=IMAGE_SIZE, interpolation='bilinear')

# Preprocess and Predict Function
def predict(img, crop_type):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = resizing_layer(img_array)
    img_array = tf.expand_dims(img_array, 0)

    model = models[crop_type]
    predictions = model.predict(img_array)

    predicted_class = class_names[crop_type][np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

### ------------------- ROUTES ------------------- ###

@app.route('/')
def home():
    return render_template('Main.html')

@app.route('/frontend')
def frontend():
    return render_template('frontend.html')

@app.route('/tomato')
def tomato_ui():
    return render_template('Tomato_frontend.html')

@app.route('/potato')
def potato_ui():
    return render_template('Potato_frontend.html')

@app.route('/tomato_detection', methods=['GET', 'POST'])
def tomato_detection():
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = tf.keras.preprocessing.image.load_img(filepath)
            predicted_label, confidence = predict(img, "tomato")

            # Prepare image path for HTML (web-friendly)
            web_path = filepath.replace("\\", "/")

            return render_template('Tomato_frontend.html',
                                   image_path=web_path,
                                   predicted_label=predicted_label,
                                   confidence=confidence)
    return render_template('Tomato_frontend.html')


@app.route('/potato_detection', methods=['GET', 'POST'])
def potato_detection():
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img = tf.keras.preprocessing.image.load_img(filepath)
            predicted_label, confidence = predict(img, "potato")

            web_path = filepath.replace("\\", "/")

            return render_template('Potato_frontend.html',
                                   image_path=web_path,
                                   predicted_label=predicted_label,
                                   confidence=confidence)
    return render_template('Potato_frontend.html')

### ------------------ MEDICINE ROUTES ------------------ ###

@app.route('/medicine_recommendation')
def medicine_recommendation():
    return render_template('medicine_frontend.html')

@app.route('/tomato_medicine')
def tomato_medicine():
    return render_template('Tomato_Medicine.html')

@app.route('/potato_medicine')
def potato_medicine():
    return render_template('Potato_medicine.html')

# Tomato Disease Pages
@app.route('/tomato_early_blight')
def tomato_early_blight():
    return render_template('Tomato_early_blight.html')

@app.route('/tomato_late_blight')
def tomato_late_blight():
    return render_template('Tomato_Late_blight.html')

@app.route('/tomato_septorial')
def tomato_septoria():
    return render_template('Tomato_septorial.html')

# Potato Disease Pages
@app.route('/potato_early_blight')
def potato_early_blight():
    return render_template('potato_early_blight.html')

@app.route('/potato_late_blight')
def potato_late_blight():
    return render_template('potato_late_blight.html')

### -------------------- MAIN -------------------- ###
if __name__ == '__main__':
    app.run(debug=True)
