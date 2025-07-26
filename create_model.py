import os
import pickle
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import mediapipe as mp
import string
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
dt_path = os.path.join(base_dir, './asl_dataset/')
Mp_hands= mp.solutions.hands
Hand1=Mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = Hand1.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]

            data_aux = []
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))
            return np.array(data_aux)
    return None
data = []
labels = []
label_names = sorted([label for label in os.listdir(dt_path) if os.path.isdir(os.path.join(dt_path, label))])
label_to_index = {label: idx for idx, label in enumerate(label_names)}
index_to_label = {idx: label for label, idx in label_to_index.items()}
all_classes = list(string.ascii_uppercase) + list(string.digits)
corrected_index_to_label = {i: label for i, label in enumerate(all_classes[:len(label_names)])}



for label in label_names:
    label_dir = os.path.join(dt_path, label)
    for image_name in os.listdir(label_dir):
        image_path = os.path.join(label_dir, image_name)
        if not image_path.endswith(('.jpg', '.png', '.jpeg')):
            continue

        landmarks = extract_landmarks_from_image(image_path)
        if landmarks is not None:
            data.append(landmarks)
            labels.append(label_to_index[label]) 

print(f"Loaded {len(data)} samples from {len(label_names)} classes.")

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
all_class_indices = sorted(index_to_label.keys())
target_names = [index_to_label[i] for i in all_class_indices]

print(classification_report(y_test, y_pred, labels=all_class_indices, target_names=target_names))
with open("model1.p", "wb") as f:
    pickle.dump({
        'model': model,
        'index_to_label': index_to_label,
        'label_to_index': label_to_index
    }, f)


print("Model and label mappings saved to model1.p")