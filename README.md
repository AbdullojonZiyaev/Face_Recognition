# Face Detection and Recognition System

This repository contains scripts for extracting, training, and detecting faces using a combination of Python, OpenCV, Dlib, and TensorFlow.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Face Extraction](#face-extraction)
    - [From Video](#from-video)
    - [From Images](#from-images)
  - [Model Training](#model-training)
  - [Face Detection](#face-detection)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides a complete pipeline for face detection and recognition:
1. **Face Extraction**: Extract faces from video or image files.
2. **Model Training**: Train a Convolutional Neural Network (CNN) model using the extracted faces.
3. **Face Detection**: Use the trained model to detect faces in new images or videos.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/face-detection-recognition.git
   cd face-detection-recognition
