import os
import pickle
import numpy as np
import pandas as pd

# Set the path to the dataset folder
dataset_path = 'encodings'

# Load face encodings and labels from the dataset folder
face_encodings = []
labels = []

person_label_mapping = {}  # To keep track of the label assigned to each person

for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_folder)

    if os.path.isdir(person_path):
        if person_folder not in person_label_mapping:
            person_label_mapping[person_folder] = len(person_label_mapping)

        label = person_label_mapping[person_folder]
        print(f"Processing person {label} in folder {person_folder}")

        for filename in os.listdir(person_path):
            if filename.startswith(person_folder) and filename.endswith('.pkl'):
                print(f"saving {filename} to the CSV file: ", end="")
                with open(os.path.join(person_path, filename), 'rb') as f:
                    encoding = pickle.load(f)
                    # Debug information
                    print(f"Processing file: {filename}")
                    print(f"Shape of encoding: {np.array(encoding).shape}")

                    face_encodings.append(encoding)
                    labels.append(label)
                print("complete")
# Print information for debugging
print("Number of face encodings found:", len(face_encodings))
print("Shape of face encodings array:", np.array(face_encodings).shape)

# Convert lists to numpy arrays
face_encodings = np.array(face_encodings)
labels = np.array(labels)

# Determine the number of features dynamically
num_features = face_encodings.shape[1] if face_encodings.shape[0] > 0 and face_encodings.shape[1] > 0 else 128

# Create DataFrame only if face_encodings is not empty
if face_encodings.shape[0] > 0:
    # Save face encodings and labels to a CSV file
    df = pd.DataFrame(np.column_stack((face_encodings.tolist(), labels)), columns=[f'feature_{i}' for i in range(num_features)] + ['label'])
    df.to_csv('face_data.csv', index=False)
    print("Face encodings saved to face_data.csv")
else:
    print("No face encodings found. The DataFrame will only contain labels.")
