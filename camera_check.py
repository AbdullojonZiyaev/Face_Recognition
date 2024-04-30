import cv2

def camera_check():
    print("Script start.")
    camera_index = []
    
    # Explicitly set the known index of your USB camera
    for i in range(1,1000):
        usb_camera_index = i  # Replace with the correct index
        
        cap = cv2.VideoCapture(usb_camera_index)
        ret, frame = cap.read()
        if ret:
            camera_index.append(usb_camera_index)
        cap.release()
        
        print(f"Detected cameras at indices: {camera_index}")
    print("Script end.")

def main():
    camera_check()

if __name__ == "__main__":
    main()
