import cv2
import numpy as np
import tensorflow as tf

# Function to preprocess the image for the model
def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (96, 96))
    normalized_image = resized_image / 255.0
    reshaped_image = np.reshape(normalized_image, (1, 96, 96, 1))
    return reshaped_image

# Function to draw keypoints on the image
def draw_keypoints(image, keypoints):
    for i in range(0, len(keypoints), 2):
        x = int(keypoints[i] * image.shape[1])
        y = int(keypoints[i + 1] * image.shape[0])
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    return image

# Fungsi masked loss
def masked_mse_loss(y_true, y_pred):
    mask = tf.not_equal(y_true, 0)  # Masker untuk nilai yang tidak nol (keypoint yang tersedia)
    mask = tf.cast(mask, tf.float32)  # Konversi ke float
    loss = tf.square(y_true - y_pred) * mask  # Hitung MSE hanya pada keypoint yang tersedia
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)  # Rata-rata loss berdasarkan jumlah keypoint yang tersedia
    return loss

# Load the trained model
model_path = 'C:/Workspace/Python/Machine-Learning/Documentations/Computer-Vision/Facial-Detection-Using-OpenCV-and-CNN/Assets/facial-keypoints-detection-model/facial_keypoints_detection_model.keras'
model = tf.keras.models.load_model(
    model_path,
    custom_objects={'masked_mse': masked_mse_loss}
)

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Start video capture
cap = cv2.VideoCapture(0)

# Initialize variables for smoothing
smooth_x, smooth_y, smooth_w, smooth_h = [], [], [], []
smoothing_factor = 8

def moving_average(values, window_size):
    if len(values) < window_size:
        return values[-1]
    return np.mean(values[-window_size:], axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale as the face detector expects gray images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        # Get the first detected face
        x, y, w, h = faces[0]

        # Update the smoothing variables
        smooth_x.append(x)
        smooth_y.append(y)
        smooth_w.append(w)
        smooth_h.append(h)

        # Apply moving average to smooth the coordinates
        avg_x = moving_average(smooth_x, smoothing_factor)
        avg_y = moving_average(smooth_y, smoothing_factor)
        avg_w = moving_average(smooth_w, smoothing_factor)
        avg_h = moving_average(smooth_h, smoothing_factor)

        # ================================================================
        face = frame[int(avg_y):int(avg_y)+int(avg_h), int(avg_x):int(avg_x)+int(avg_w)]
        resized_face = cv2.resize(face, (96, 96))
        preprocessed_face = preprocess_image(resized_face)
        keypoints = model.predict(preprocessed_face, verbose=0)[0]
        cv2.imshow('Resized Face', preprocessed_face[0])
        
        left_eye_center_x = keypoints[0]
        left_eye_center_y = keypoints[1]

        right_eye_center_x = keypoints[2]
        right_eye_center_y = keypoints[3]

        # Scale keypoints back to the original face dimensions
        scale_x = avg_w / 96.0
        scale_y = avg_h / 96.0

        left_eye_center_x = keypoints[0] * scale_x + avg_x
        left_eye_center_y = keypoints[1] * scale_y + avg_y
        cv2.circle(frame, (int(left_eye_center_x), int(left_eye_center_y)), 2, (0, 255, 0), -1)

        left_eye_inner_corner_x = keypoints[4] * scale_x + avg_x
        left_eye_inner_corner_y = keypoints[5] * scale_x + avg_x
        cv2.circle(frame, (int(left_eye_inner_corner_x), int(left_eye_inner_corner_y)), 2, (0, 255, 0), -1)

        left_eye_outer_corner_x = keypoints[6] * scale_x + avg_x
        left_eye_outer_corner_y = keypoints[7] * scale_x + avg_x
        cv2.circle(frame, (int(left_eye_outer_corner_x), int(left_eye_outer_corner_x)), 2, (0, 255, 0), -1)

        right_eye_center_x = keypoints[2] * scale_x + avg_x
        right_eye_center_y = keypoints[3] * scale_y + avg_y
        cv2.circle(frame, (int(right_eye_center_x), int(right_eye_center_y)), 2, (0, 255, 0), -1)
        # ================================================================
        
        # Draw a rectangle around the detected face with smoothed coordinates
        cv2.rectangle(frame, (int(avg_x), int(avg_y)), (int(avg_x + avg_w), int(avg_y + avg_h)), (255, 0, 0), 2)

    # Display the output frame with detected faces and keypoints
    cv2.imshow('Facial Keypoints Detection', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()