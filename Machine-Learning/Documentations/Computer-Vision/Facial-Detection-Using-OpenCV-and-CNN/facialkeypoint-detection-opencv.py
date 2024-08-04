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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale as the face detector expects gray images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract the face ROI
        face = frame[y:y+h, x:x+w]

        # Resize the face to 96x96 pixels
        resized_face = cv2.resize(face, (96, 96))

        # Preprocess the face for the model
        preprocessed_face = preprocess_image(resized_face)

        # For demonstration purposes, we will display the resized face
        cv2.imshow('Resized Face', preprocessed_face[0])

        # ==================================================================================
        # Predict keypoints
        keypoints = model.predict(preprocessed_face, verbose=0)[0]
        print(keypoints[0])

        # # Rescale keypoints to match the face ROI
        # keypoints = keypoints * 48 + 48  # Reverse the normalization
        # keypoints = keypoints.reshape(-1, 2) * [w / 96, h / 96]
        # print(keypoints)

        # Draw keypoints on the original frame
        # for (kx, ky) in keypoints:
        #     cv2.circle(frame, (int(x + kx), int(y + ky)), 1, (0, 255, 0), 2)
        # ==================================================================================

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the output frame with detected faces and keypoints
    cv2.imshow('Facial Keypoints Detection', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

### Perlu ada penyesuaian lagi, untuk hasil prediksi dari keypoint, karna koordinat prediksi, berlaku untuk ukuran 96x96 pixel sehingga perlu ada kalibrasi lagi.