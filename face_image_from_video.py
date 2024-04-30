import cv2
import face_recognition

def capture_faces_from_video(video_path, output_directory, target_frames=1000):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Create the output directory if it doesn't exist
    if not cv2.os.path.exists(output_directory):
        cv2.os.makedirs(output_directory)

    # Counter for captured frames
    captured_frames = 0

    while captured_frames < target_frames:
        # Read a frame
        ret, frame = video_capture.read()

        if not ret:
            print("Error reading the video file.")
            break

        # Find all face locations in the current frame
        face_locations = face_recognition.face_locations(frame)

        # Process each face found in the frame
        for face_location in face_locations:
            top, right, bottom, left = face_location

            # Extract the face region
            face_image = frame[top:bottom, left:right]

            # Save the face image
            image_path = f"{output_directory}/frame_{captured_frames}.jpg"
            cv2.imwrite(image_path, face_image)

            captured_frames += 1

        # Display the current progress
        print(f"Frames captured: {captured_frames}/{target_frames}")

    # Release the video capture object
    video_capture.release()

if __name__ == "__main__":
    # Set the path to the input video file
    input_video_path = "Your video path"

    # Set the output directory to save face images
    output_directory = "dataset/person_name"

    # Call the function to capture faces from the video
    capture_faces_from_video(input_video_path, output_directory)
