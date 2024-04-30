import os
import cv2
import face_recognition
import pickle

# Set the path to the dataset folder
DATASET_PATH = 'dataset/alijon_ziyaev'

def capture_face_images(num_frames, path=DATASET_PATH):
    """
    Using cv2 library, it will capture the face images and later encode them in the path directory
    """
    # Initialize the camera
    try:
        cap = cv2.VideoCapture(700)  # Make sure this is the correct camera index
        if not cap.isOpened():
            raise Exception("Could not open camera.")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    # Create the dataset folder if it doesn't exist
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    # Collect images
    images = []
    try:
        for image_count in range(num_frames):
            ret, frame = cap.read()

            # Display the frame for a brief moment
            cv2.imshow('Capture Images', frame)
            cv2.waitKey(1)

            # Save the image
            img_path = os.path.join(path, f'img{image_count}.jpg')
            cv2.imwrite(img_path, frame)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Release the camera
        cap.release()
        cv2.destroyAllWindows()

def main():
    num_frames_to_capture = 1000  # Adjust as needed
    capture_face_images(num_frames_to_capture)

if __name__ == "__main__":
    main()
