import cv2
import numpy as np

# Quick face recognition demo
# Install: pip install opencv-python

def detect_faces(image_path):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the input image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    print(f'Found {len(faces)} faces')
    return img

if __name__ == '__main__':
    print('Face Recognition Demo - Ready in under 2 mins!')
    # Example usage:
    # result = detect_faces('your_image.jpg')
    # cv2.imshow('Faces', result)
    # cv2.waitKey()
