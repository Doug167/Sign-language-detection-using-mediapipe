import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

Mp_hands=mp.solutions.hands
Mp_utils=mp.solutions.drawing_utils
Hand1=Mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
dt_path = os.path.join(base_dir, 'asl_dataset/')

data=[]
labels=[]


for direct in os.listdir(dt_path):
    label_pt=os.path.join(dt_path,direct)
    if not os.path.isdir(label_pt):
        continue
    for img_path in os.listdir(label_pt):
        try:
            img_full_path = os.path.join(label_pt, img_path)
            img = cv2.imread(img_full_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img_rgb.shape
            results = Hand1.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_, y_ = [], []

                    for landmark in hand_landmarks.landmark:
                        x_pixel = landmark.x * width
                        y_pixel = landmark.y * height
                        x_.append(x_pixel)
                        y_.append(y_pixel)

                    min_x, min_y = min(x_), min(y_)
                    for landmark in hand_landmarks.landmark:
                        data_aux.append(landmark.x * width - min_x)
                        data_aux.append(landmark.y * height - min_y)

                    data.append(data_aux)
                    labels.append(direct)
        except Exception as e:
            print("Error processing:",img_path)
with open ('data.pickle','wb') as f:
    pickle.dump({'data':data,'labels':labels},f)
print("Saved to pickle file")