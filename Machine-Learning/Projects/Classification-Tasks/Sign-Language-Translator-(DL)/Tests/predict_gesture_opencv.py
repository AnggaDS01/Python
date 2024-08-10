# Import modules
import sys
from pathlib import Path

config_path = Path('./Machine-Learning/Projects/Classification-Tasks/Sign-Language-Translator-(DL)')
sys.path.append(str(config_path))

from config import CLASSES_LIST, MODEL_PATH, OFFSET_PREP_IMAGE_FOR_PREDDICTION, IMAGE_SIZE
from test_preprocess import predict_gesture

import mediapipe as mp
import tensorflow as tf
import numpy as np
import cv2

# Define drawing specifications for landmarks and connections
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_drawing_styles = mp.solutions.drawing_styles

landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 0))
connection_spec = mp_drawing.DrawingSpec(color=(255, 255, 255))

# Create a hand detector object with parameters
detector = mp_hands.Hands(
  max_num_hands=1, 
  static_image_mode=True,
  min_detection_confidence=0.2, 
  min_tracking_confidence=0.2
)

best_model = tf.keras.models.load_model(MODEL_PATH)

# Open the webcam and start capturing frames
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the webcam
    ret, frame = cap.read() # frame output: RGB or 3 channels already, no need to cvt to rgb
    if not ret:
        break
    # Make a copy of the original frame for output
    img_output = frame.copy()
    # Process the frame with the hand detector and get the results
    results = detector.process(frame)

    # If there are any hand landmarks detected in the frame
    if results.multi_hand_landmarks:
        # Get the first hand landmarks from the results
        hand_landmarks = results.multi_hand_landmarks[0]
        # Draw the landmarks and connections on the original frame
        mp.solutions.drawing_utils.draw_landmarks(
        frame, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS, 
        landmark_drawing_spec=landmark_spec,
        connection_drawing_spec=connection_spec
            )

        # Create an empty list to store the landmark coordinates
        points = []
        
        # Loop through each landmark and append its x and y coordinates to the list
        for data_point in hand_landmarks.landmark:
            points.append([data_point.x * frame.shape[1], data_point.y* frame.shape[0]])
            # Get the bounding rectangle of the landmark coordinates
        x,y,w,h = cv2.boundingRect(np.array(points).astype('float32'))

        # Crop the face area with an additional margin
        detected_hand = frame[
            int(max(0, y - OFFSET_PREP_IMAGE_FOR_PREDDICTION)):int(min(y + h + OFFSET_PREP_IMAGE_FOR_PREDDICTION, frame.shape[0])),
            int(max(0, x - OFFSET_PREP_IMAGE_FOR_PREDDICTION)):int(min(x + w + OFFSET_PREP_IMAGE_FOR_PREDDICTION, frame.shape[1]))
        ]

        if detected_hand.size > 0:
            label_pred = predict_gesture(
                model=best_model, 
                classes_list=CLASSES_LIST, 
                input_image=detected_hand,
                image_size_for_model=IMAGE_SIZE,
                )

            cv2.rectangle(img_output, (x+30, y-30), (x + 100, y - 85), (255, 0, 255), cv2.FILLED)
            cv2.putText(img_output, label_pred, (x+50, y-50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Show the original frame with landmarks and connections in a window named "Original"
    cv2.imshow("Original", img_output)

    # Wait for a key press and check if it is 'q' or if the counter reaches the target amount
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Break out of the loop and stop capturing frames
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
