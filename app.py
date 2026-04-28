import streamlit as st
import cv2
import numpy as np
from tensorflow import keras
import os

# ── Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'fer_model_final.keras')
    return keras.models.load_model(model_path)

model = load_model()

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_COLORS = {
    'Angry':    (255, 50,  50),
    'Disgust':  (180, 50, 255),
    'Fear':     (50,  50, 255),
    'Happy':    (50, 255, 100),
    'Sad':      (50, 200, 255),
    'Surprise': (255, 180, 50),
    'Neutral':  (200, 200, 200)
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Prediction
def predict_emotion(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = face.reshape(1, 48, 48, 1)

        preds      = model.predict(face, verbose=0)[0]
        emotion    = EMOTION_LABELS[int(np.argmax(preds))]
        confidence = float(np.max(preds)) * 100
        color      = EMOTION_COLORS[emotion]

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{emotion} ({confidence:.1f}%)",
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        results.append((emotion, confidence))

    return frame, results

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FER System", page_icon="😊", layout="wide")
st.title("😊 Facial Expression Recognition")
st.markdown("Real-time emotion detection — CNN trained on FER-2013")

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### Detected Emotion")
    emotion_display = st.empty()
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Dataset: FER-2013")
    st.markdown("- Classes: 7 emotions")
    st.markdown("- Input: 48×48 grayscale")

with col1:
    run = st.checkbox("▶ Start Webcam")
    frame_window = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Could not access webcam!")
        break

    frame, results = predict_emotion(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb, use_column_width=True)

    if results:
        text = ""
        for emotion, conf in results:
            text += f"**{emotion}**\n\n`{conf:.1f}% confidence`\n\n"
        emotion_display.markdown(text)
    else:
        emotion_display.markdown("_No face detected_")

cap.release()