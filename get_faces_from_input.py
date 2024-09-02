# -*- coding: utf-8 -*-
import sys
import os
import cv2
import dlib

input_dir = './input_img'
output_dir = './other_faces'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Use dlib's built-in frontal face detector as our feature extractor
detector = dlib.get_frontal_face_detector()

index = 1
for (path, dirnames, filenames) in os.walk(input_dir):
    for filename in filenames:
        if filename.endswith('.jpg'):
            print('Being processed picture %s' % index)
            img_path = path + '/' + filename
            # Read the image from file
            img = cv2.imread(img_path)
            # Convert to grayscale image
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Use the detector for face detection; `dets` is the result returned
            dets = detector(gray_img, 1)

            # Use the enumerate function to iterate over the elements in the sequence along with their indices
            # Index `i` represents the face number
            # `left`: distance of the left side of the face from the left boundary of the image
            # `right`: distance of the right side of the face from the left boundary of the image
            # `top`: distance of the top side of the face from the top boundary of the image
            # `bottom`: distance of the bottom side of the face from the top boundary of the image
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                # img[y:y+h, x:x+w]
                face = img[x1:y1, x2:y2]
                # Resize the image
                face = cv2.resize(face, (size, size))
                cv2.imshow('image', face)
                # Save the image
                cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
                index += 1

            key = cv2.waitKey(30) & 0xff
            if key == 27:
                sys.exit(0)
