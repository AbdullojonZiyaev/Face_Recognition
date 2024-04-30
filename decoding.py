import cv2
import numpy as np
import pickle

def show_decoded_image(encoding):
    # Decode the face encoding
    face_image = np.array(encoding)
    face_image = face_image.astype(np.uint8)

    # Display the image
    cv2.imshow("Decoded Image", face_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the encoding from a pickle file
with open('alisher_4.pkl', 'rb') as f:
    encoding = pickle.load(f)

# Show the decoded image
show_decoded_image(encoding)
