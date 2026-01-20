import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model

# Load pre-trained InceptionV3 for feature extraction
base_model = InceptionV3()
feature_extractor_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

# Load the trained LSTM model
lstm_model = load_model("vd.h5")  # Update this path if needed

# Constants
SEQUENCE_LENGTH = 16
IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299

# Streamlit App
st.title("Violence Detection App")

# File uploader for video or image
uploaded_file = st.file_uploader("Upload a Video or Image", type=["mp4", "avi", "mov", "jpg", "jpeg", "png"])

if uploaded_file:
    file_type = uploaded_file.type

    if "video" in file_type:
        # Handle video input
        video_path = uploaded_file.name

        # Save the uploaded video to a temporary file
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        st.video(video_path)

        # Process video
        cap = cv2.VideoCapture(video_path)
        frame_buffer = []
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized_frame = cv2.resize(frame_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
            normalized_frame = resized_frame / 255.0
            img = np.expand_dims(normalized_frame, axis=0)

            # Extract features
            feature_vector = feature_extractor_model.predict(img, verbose=0)
            frame_buffer.append(feature_vector)

            # Maintain buffer size
            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)

            if len(frame_buffer) == SEQUENCE_LENGTH:
                input_sequence = np.array(frame_buffer).reshape((1, SEQUENCE_LENGTH, 2048))
                prediction = lstm_model.predict(input_sequence, verbose=0)[0][0]
                label = "Violence" if prediction < 0.5 else "Non-Violence"
                results.append(label)

        cap.release()

        # Display results
        st.write("Analysis Complete.")
        st.write(f"Detected: {results[-1]}")

    elif "image" in file_type:
        # Handle image input
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Preprocess image
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
        normalized_frame = resized_frame / 255.0
        img = np.expand_dims(normalized_frame, axis=0)

        # Extract features
        feature_vector = feature_extractor_model.predict(img, verbose=0)
        input_sequence = np.expand_dims(feature_vector, axis=0)

        # Predict using the LSTM model (for single image)
        prediction = lstm_model.predict(input_sequence, verbose=0)[0][0]
        label = "Violence" if prediction < 0.5 else "Non-Violence"

        # Display image and result
        st.image(frame_rgb, caption="Uploaded Image", use_column_width=True)
        st.write(f"Prediction: {label}")
