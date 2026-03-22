# Instrument Classifier App

A Streamlit web app that predicts the musical instrument in a WAV audio clip using a trained XGBoost model.

## Live Demo

https://instrumentclassifier-6hwcq48cmtz3tl2fpbg2eo.streamlit.app/

## Features

- Upload a WAV file and preview audio in the browser
- Predict instrument class from the first 3 seconds of audio
- Show prediction confidence score
- Clean UI with instrument labels and color-coded results

## Supported Instruments

- Piano
- Harpsichord
- Violin
- Cello
- Clarinet

## Project Structure

```text
instrument_classifier_app/
├─ app.py
├─ requirements.txt
├─ instrument_classifier_xgboost.pkl
└─ label_encoder.pkl
```

## Tech Stack

- Streamlit
- librosa
- NumPy
- XGBoost
- joblib

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies.
3. Launch Streamlit.

```powershell
# from project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

If `streamlit` command is not recognized, always run it as a Python module:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Input Notes

- File type: WAV only
- Recommended: clear single-instrument recordings
- The app analyzes up to the first 3 seconds of audio

## Model Limitations

- Trained only on five instruments: Piano, Harpsichord, Violin, Cello, Clarinet
- Performance may drop on noisy recordings or overlapping instruments
- Short or very quiet clips can reduce prediction reliability
- Confidence score is model confidence, not guaranteed real-world accuracy
- Results can vary with recording quality, microphone type, and distance

## Deployment

This app is deployed on Streamlit Community Cloud.
