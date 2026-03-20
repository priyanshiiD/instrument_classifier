import streamlit as st
import librosa
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Instrument Classifier", page_icon="🎵", layout="wide")

# Load model and encoder
@st.cache_resource
def load_models():
    model = joblib.load("instrument_classifier_xgboost.pkl")
    le = joblib.load("label_encoder.pkl")
    return model, le

model, le = load_models()

instrument_map = {
    1: "🎹 Piano",
    7: "🎼 Harpsichord",
    41: "🎻 Violin",
    43: "🎻 Cello",
    72: "🎷 Clarinet"
}

instrument_colors = {
    "🎹 Piano": "#FF6B6B",
    "🎼 Harpsichord": "#4ECDC4",
    "🎻 Violin": "#45B7D1",
    "🎻 Cello": "#96CEB4",
    "🎷 Clarinet": "#FFEAA7"
}

def extract_features(signal, sr, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    features = np.mean(mfcc.T, axis=0)
    return features

def predict_instrument(audio_file):
    try:
        signal, sr = librosa.load(audio_file, sr=16000)
        segment = signal[:3*sr]
        features = extract_features(segment, sr)
        
        # Get prediction and confidence
        prediction = model.predict([features])
        probabilities = model.predict_proba([features])
        confidence = np.max(probabilities) * 100
        
        instrument_code = le.inverse_transform(prediction)[0]
        instrument_name = instrument_map.get(instrument_code, "Unknown")
        
        return instrument_name, confidence
    except Exception as e:
        return None, str(e)

# Sidebar
with st.sidebar:
    st.title("ℹ️ About")
    st.info(
        """
        **Instrument Classifier** uses AI to recognize musical instruments from audio files.
        
        **Supported Instruments:**
        - 🎹 Piano
        - 🎼 Harpsichord
        - 🎻 Violin
        - 🎻 Cello
        - 🎷 Clarinet
        """
    )
    
    st.divider()
    
    st.markdown("### 📋 Instructions")
    st.markdown(
        """
        1. Upload a WAV audio file
        2. Listen to the preview
        3. Get instant classification
        4. See confidence score
        """
    )

# Main title
st.markdown(
    """
    <h1 style='text-align: center; color: #00D9FF;'>
    🎵 Instrument Classifier
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>Identify musical instruments from audio files</p>",
    unsafe_allow_html=True
)

st.divider()

# Upload section
col1, col2 = st.columns([2, 1], gap="medium")

with col1:
    st.subheader("📤 Upload Audio File")
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"], label_visibility="collapsed")

with col2:
    st.subheader("📊 Stats")
    st.metric("Supported Formats", "WAV")
    st.metric("Audio Length (max)", "3 seconds")

if uploaded_file is not None:
    st.divider()
    
    # Audio preview
    col_audio, col_info = st.columns([2, 1])
    
    with col_audio:
        st.subheader("🔊 Audio Preview")
        st.audio(uploaded_file)
    
    with col_info:
        st.subheader("📁 File Info")
        st.write(f"**Name:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
    
    st.divider()
    
    # Prediction
    st.subheader("🎙️ Classification Result")
    
    with st.spinner("🔍 Analyzing audio..."):
        instrument_name, result = predict_instrument(uploaded_file)
    
    if instrument_name:
        # Display result with styling
        color = instrument_colors.get(instrument_name, "#00D9FF")
        confidence = result
        
        col_result, col_confidence = st.columns(2)
        
        with col_result:
            st.markdown(
                f"""
                <div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>{instrument_name}</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_confidence:
            st.markdown(
                f"""
                <div style='background-color: #2D3436; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {color};'>
                    <p style='color: gray; margin: 0; font-size: 14px;'>Confidence Score</p>
                    <h2 style='color: {color}; margin: 0;'>{confidence:.1f}%</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.error(f"❌ Error analyzing file: {result}")
        st.info("Make sure the file is a valid WAV audio file.")
else:
    st.info("👆 Upload an audio file to get started!")
