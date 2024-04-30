import cv2
import face_recognition
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# Load the trained SVM model and scaler
loaded_model = joblib.load('face_recognition_model.joblib')
scaler = joblib.load('scaler.joblib')  # Update with the actual filename

# Load label-to-person mapping created during training
label_to_person = {0: 'Abdullojon', 1: 'Alijon', 2: 'Alisher', 
                   3: 'Asolat',4: 'Bobjon',5: 'Hasan', 6: 'Husen', 7: 'Manija', 
                   8: 'Masrur', 9: 'Mavlon', 10: 'Nazira', 11: 'Ochajon', 12: 'Zarina'}  # Update with the actual filename

# Capture a face image using OpenCV
cap = cv2.VideoCapture(700)  # Use the correct camera index
while True:
    ret, frame = cap.read()

    # Perform face detection and encoding
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if len(face_encodings) > 0:
        # Scale face encodings
        scaled_face_encodings = scaler.transform(face_encodings)

        # Predict labels using the loaded model
        predicted_labels = loaded_model.predict(scaled_face_encodings)

        # Map labels to person names
        recognized_people = [label_to_person[label] for label in predicted_labels]

        # Display or output the results
        for (top, right, bottom, left), person_name in zip(face_locations, recognized_people):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, person_name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
