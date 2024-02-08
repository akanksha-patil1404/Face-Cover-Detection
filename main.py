import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# Load pre-trained face mask detection model
maskNet = load_model("model.h5")


# Function to detect face masks
def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    faces_list=[]
    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face, (224, 224))  # Resize to match model input size
        face_resized = img_to_array(face_resized)
        face_resized = preprocess_input(face_resized)
        face_resized = np.expand_dims(face_resized, axis=0)

        faces_list.append(face_resized)
        if len(faces_list) > 0:
            preds = maskNet.predict(faces_list)
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Face Covered" if mask > withoutMask else "Face Not Covered"
        if label == "Face Covered":  # Adjust this threshold as needed
            cv2.putText(frame, "Suspicious Person", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        color = (0, 255, 0) if label == "Face Covered" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        # Predict whether the face is wearing a mask or not


    return frame


# Main loop to read frames from the video stream
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = detect_mask(frame)

    cv2.imshow('Face Mask Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
