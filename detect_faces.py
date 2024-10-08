import tensorflow as tf
import cv2
import dlib
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split

my_faces_path = './my_faces'
other_faces_path = './other_faces'
size = 64

imgs = []
labs = []

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0, 0, 0, 0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # // indicates the integer division operator
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path, h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top, bottom, left, right = getPaddingSize(img)
            # Enlarge the image, expanding the edges of the image
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)

readData(my_faces_path)
readData(other_faces_path)
# Convert image data and labels into arrays
imgs = np.array(imgs)
labs = np.array([[0, 1] if lab == my_faces_path else [1, 0] for lab in labs])
# Randomly split the data into a training set and a test set
train_x, test_x, train_y, test_y = train_test_split(imgs, labs, test_size=0.05, random_state=random.randint(0, 100))
# Parameters: the total number of images, image height, width, and channels
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# Convert the data to values less than 1
train_x = train_x.astype('float32') / 255.0
test_x = test_x.astype('float32') / 255.0

print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# Image batch, taking 128 images at a time
batch_size = 128
num_batch = len(train_x) // 128

x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)

def cnnLayer():
    # First layer
    W1 = weightVariable([3, 3, 3, 32])  # Convolution kernel size (3,3), input channels (3), output channels (32)
    b1 = biasVariable([32])
    # Convolution
    conv1 = tf.nn.relu(conv2d(x, W1) + b1)
    # Pooling
    pool1 = maxPool(conv1)
    # Reduce overfitting, randomly drop some weights
    drop1 = dropout(pool1, keep_prob_5)

    # Second layer
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # Third layer
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # Fully connected layer
    Wf = weightVariable([8 * 16 * 32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8 * 16 * 32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # Output layer
    Wout = weightVariable([512, 2])
    bout = biasVariable([2])
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

output = cnnLayer()  
predict = tf.argmax(output, 1)  
   
saver = tf.train.Saver()  
sess = tf.Session()  
saver.restore(sess, tf.train.latest_checkpoint('.'))  
   
def is_my_face(image):  
    res = sess.run(predict, feed_dict={x: [image / 255.0], keep_prob_5: 1.0, keep_prob_75: 1.0})  
    if res[0] == 1:  
        return True  
    else:  
        return False  

# Use dlib's built-in frontal_face_detector as our feature extractor
detector = dlib.get_frontal_face_detector()

cam = cv2.VideoCapture(0)  
   
while True:  
    _, img = cam.read()  
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(gray_image, 1)
    if not len(dets):
        # print('Can`t get face.')
        cv2.imshow('img', img)
        key = cv2.waitKey(30) & 0xff  
        if key == 27:
            sys.exit(0)
            
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2]
        # Resize the image
        face = cv2.resize(face, (size, size))
        print('Is this my face? %s' % is_my_face(face))

        cv2.rectangle(img, (x2, x1), (y2, y1), (255, 0, 0), 3)
        cv2.imshow('image', img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            sys.exit(0)
  
sess.close() 
