from flask import Flask, request, render_template, send_from_directory
import os
from pathlib import Path

import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
MODEL_PATH = BASE_DIR / 'deepfake_model.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'wav', 'mp3', 'm4a', 'ogg'}

model = None
model_error = None

image_height = 640
image_width = 480

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global model, model_error
    if model is not None:
        return model
    if not MODEL_PATH.exists():
        model_error = 'The trained model is missing. Run train.py first.'
        return None
    try:
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH)
    except Exception as error:
        model_error = f'Model could not be loaded: {error}'
    return model


def process_file(file):
    filename = secure_filename(file.filename)
    UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = UPLOAD_DIR / filename
    file.save(file_path)
    extension = file_path.suffix.lower()
    if extension in {'.wav', '.mp3', '.m4a', '.ogg'}:
        image_path = create_waveform_image(file_path)
    else:
        image_path = file_path

    from tensorflow.keras.preprocessing import image
    img = image.load_img(image_path, target_size=(image_height, image_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img, filename


def create_waveform_image(audio_path):
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    samples, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    image_path = audio_path.with_suffix('.png')
    figure, axis = plt.subplots(figsize=(8, 4), dpi=80)
    librosa.display.waveshow(samples, sr=sample_rate, ax=axis, color='#1d4ed8')
    axis.set_axis_off()
    figure.tight_layout(pad=0)
    figure.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close(figure)
    return image_path

@app.route('/')
def home():
    load_model()
    return render_template('upload.html', prediction=None, error=model_error)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('upload.html', prediction=None, error='No file uploaded.')

    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', prediction=None, error='No file selected.')

    if not allowed_file(file.filename):
        return render_template('upload.html', prediction=None, error='Please upload an audio file or PNG/JPG waveform image.')

    active_model = load_model()
    if active_model is None:
        return render_template('upload.html', prediction=None, error=model_error)

    img, filename = process_image(file)
    predictions = active_model.predict(img, verbose=0)
    class_names = ['fake', 'real']
    predicted_class_index = int(np.argmax(predictions[0]))
    predicted_class = class_names[predicted_class_index]

    return render_template('upload.html', prediction=predicted_class, image_path=f'uploads/{filename}', error=None)

if __name__ == '__main__':
    UPLOAD_DIR.mkdir(exist_ok=True)
    app.run(debug=True)