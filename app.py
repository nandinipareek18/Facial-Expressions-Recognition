import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Facial Expression Recognition", layout="centered")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_emotion_model():
    return load_model("Emotion_little_vgg.h5")

classifier = load_emotion_model()
face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ---------------- FUNCTIONS ----------------
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        roi = roi_gray.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = classifier.predict(roi, verbose=0)[0]
        label = class_labels[np.argmax(preds)]
        confidence = np.max(preds)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} ({confidence:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    return frame

# ---------------- UI ----------------
st.title("😊 Facial Expression Recognition")
st.markdown("Choose **Camera** or **Upload Image** to detect facial emotions.")

option = st.radio("Select Input Method:", ["📷 Camera", "🖼 Upload Image"])

# ---------------- CAMERA OPTION ----------------
if option == "📷 Camera":
    st.warning("Camera works only in local deployment")
    img = st.camera_input("Take a picture")

    if img is not None:
        image = Image.open(img)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        result = detect_emotion(frame)

        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Emotion")

# ---------------- UPLOAD IMAGE OPTION ----------------
elif option == "🖼 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        result = detect_emotion(frame)

        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected Emotion")
