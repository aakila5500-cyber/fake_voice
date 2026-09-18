# Deepfake Voice Detection

This project converts voice notes into waveform images and classifies them as `fake` or `real` with a TensorFlow model. It provides a small Flask web interface and also accepts existing PNG/JPG waveform images.

## Setup

Use Python 3.10 or 3.11 in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

The sample dataset is already arranged in `fake/` and `real/`. Add more labeled waveform images to those folders before training. Then run:

```powershell
python train.py --epochs 10
```

This creates `deepfake_model.h5` in the project directory. The current checked-in sample set is very small, so its accuracy is not meaningful for production use.

## Run the web app

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` and upload a WAV, MP3, M4A, or OGG voice note. The app converts the audio to a waveform image before prediction. PNG and JPG waveform images are also supported. The app starts without a model, but prediction remains unavailable until `deepfake_model.h5` has been generated.

## Project files

- `app.py`: Flask upload, waveform conversion, and prediction server.
- `train.py`: local TensorFlow training entry point.
- `model.ipynb`: original Google Colab training notebook.
- `predict.py`: command-line prediction example.
