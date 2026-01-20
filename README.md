# Violence Detection in Video Sequences

This project implements a deep learning pipeline to automatically detect violence in video sequences using a combination of Convolutional LSTM (ConvLSTM) and Inception V3 networks. The goal is to classify whether a given video contains violent or non-violent actions with high accuracy.

## Overview

Violence detection in surveillance and public safety systems is critical for proactive response and threat prevention. This project leverages spatial features using Inception V3 and temporal dynamics using ConvLSTM to build an effective violence classification model.

## Model Architecture

- **Inception V3**: Used for extracting spatial features from individual video frames.
- **ConvLSTM**: Used to capture temporal dependencies between frames for better sequence understanding.
- **Frameworks**: Built using TensorFlow/Keras and OpenCV.

## Features

- Implemented hybrid architecture with **Inception V3 + Conv-LSTM**.
- Performed **data preprocessing and augmentation** on video frames to enhance model accuracy and generalization.
- Achieved **97% accuracy** in distinguishing between violent and non-violent actions.
- End-to-end pipeline for frame extraction, preprocessing, model training, and prediction.

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib (for visualization)

## Project Structure

├── model/
│ ├── inception_model.py
│ ├── conv-lstm_model.py
├── data/
│ ├── raw_videos/
│ ├── processed_frames/
├── utils/
│ ├── preprocessing.py
│ ├── augmentation.py
├── train.py
├── predict.py
├── README.md


## Results

- **Validation Accuracy**: 97%
- **Model Behavior**: Robust against diverse lighting, backgrounds, and action speeds.
