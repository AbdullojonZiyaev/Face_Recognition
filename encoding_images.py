import os
import cv2
import face_recognition
import pickle

def encode_images_in_folders(parent_directory, encoding_directory):
    """
    Encode face images in different folders within the parent directory and save encodings.
    
    Parameters:
    - parent_directory: The directory containing subdirectories with face images.
    - encoding_directory: The directory to save face encodings.
    """
    # Create the encoding directory if it doesn't exist
    if not os.path.exists(encoding_directory):
        os.makedirs(encoding_directory)

    for person_folder in os.listdir(parent_directory):
        person_path = os.path.join(parent_directory, person_folder)

        if os.path.isdir(person_path):
            person_encoding_directory = os.path.join(encoding_directory, person_folder)

            # Create a folder for the person's encodings
            if not os.path.exists(person_encoding_directory):
                os.makedirs(person_encoding_directory)

            print(f"Processing images in folder: {person_folder}")

            # Process each image in the person's folder
            for index, filename in enumerate(os.listdir(person_path)):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    print("Processing", filename, ": ", end='')
                    image_path = os.path.join(person_path, filename)

                    # Load the image
                    face_image = face_recognition.load_image_file(image_path)

                    # Encode the face in the image
                    face_encodings = face_recognition.face_encodings(face_image)

                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]

                        # Save the face encoding (pickle)
                        encoding_filename = f'{person_folder}_{index}.pkl'
                        encoding_path = os.path.join(person_encoding_directory, encoding_filename)
                        
                        with open(encoding_path, 'wb') as f:
                            pickle.dump(face_encoding, f)
                        print("complete")
                    else:
                        print("no face detected")

def main():
    parent_directory = 'dataset'
    encoding_directory = 'encodings'
    encode_images_in_folders(parent_directory, encoding_directory)

if __name__ == "__main__":
    main()
