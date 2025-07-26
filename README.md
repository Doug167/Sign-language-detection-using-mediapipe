# Sign-language-detection-using-mediapipe

This is a sign language detecion program able to detect 36 classes (A-Z, 0-9).
The training dataset was sourced from https://www.kaggle.com/datasets/ayuraj/asl-dataset
The modules used are : Mediapipe, OpenCV, pickle and sklearn. Mediapipe is used to detect the hands in the images and video feeds.

#create_pickle.py:
This program extracts the landmarks from the dataset and stores in pickle file along with the labels.

#create_model.py:
This program trains a machine model using the extracted landmarks and stores in a .p file. We use Random Forest model for this purpose.

#detect.py:
This program is used to detect the sign language gestures directly from the camera.

Prepared for the triwaizardathon 1.0 hackathon.
