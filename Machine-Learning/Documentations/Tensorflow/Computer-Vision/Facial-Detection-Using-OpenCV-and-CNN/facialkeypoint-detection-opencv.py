import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image, image_size):
    """
    Preprocess the input image for the model.

    Args:
    image (numpy array): Input image.
    image_size (tuple): Target size (height, width) for resizing.

    Returns:
    numpy array: Preprocessed image ready for prediction.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, image_size)
    normalized_image = resized_image / 255.0
    reshaped_image = np.reshape(normalized_image, (1, image_size[0], image_size[1], 1))
    return reshaped_image

def predict_and_scale_keypoints(face, model, image_size, avg_x, avg_y, avg_w, avg_h):
    """
    Predict and scale the facial keypoints for the detected face.

    Args:
    face (numpy array): Detected face image.
    model (tf.keras.Model): Trained facial keypoints detection model.
    image_size (tuple): Size to which the face image should be resized for the model.
    avg_x (float): Smoothed x-coordinate of the face.
    avg_y (float): Smoothed y-coordinate of the face.
    avg_w (float): Smoothed width of the face.
    avg_h (float): Smoothed height of the face.

    Returns:
    list of tuples: Scaled keypoints coordinates.
    """
    resized_face = cv2.resize(face, image_size)
    preprocessed_face = preprocess_image(resized_face, image_size)
    keypoints = model.predict(preprocessed_face, verbose=0)[0]

    scale_x = avg_w / image_size[1]
    scale_y = avg_h / image_size[0]

    scaled_keypoints = []
    for i in range(0, len(keypoints), 2):
        x = keypoints[i] * scale_x + avg_x
        y = keypoints[i + 1] * scale_y + avg_y
        scaled_keypoints.append((int(x), int(y)))

    return scaled_keypoints

def masked_mse_loss(y_true, y_pred):
    """
    Custom loss function to calculate Mean Squared Error (MSE) for available keypoints.

    Args:
    y_true (tf.Tensor): Ground truth keypoints.
    y_pred (tf.Tensor): Predicted keypoints.

    Returns:
    tf.Tensor: Masked MSE loss value.
    """
    mask = tf.not_equal(y_true, 0)
    mask = tf.cast(mask, tf.float32)
    loss = tf.square(y_true - y_pred) * mask
    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss

def moving_average(values, window_size):
    """
    Compute the moving average of the given list of values.

    Args:
    values (list): List of values to smooth.
    window_size (int): Number of values to include in the moving average.

    Returns:
    float: Smoothed value.
    """
    if len(values) < window_size:
        return values[-1]
    return np.mean(values[-window_size:], axis=0)

def main():
    """
    Main function to run the facial keypoints detection on live video feed.
    """
    # Load the trained model
    model_path = 'C:/Workspace/Python/Machine-Learning/Documentations/Computer-Vision/Facial-Detection-Using-OpenCV-and-CNN/Assets/facial-keypoints-detection-model/facial_keypoints_detection_model.keras'
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'masked_mse': masked_mse_loss}
    )

    # Initialize the face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    # Variables for smoothing the detected face coordinates
    smooth_x, smooth_y, smooth_w, smooth_h = [], [], [], []
    smoothing_factor = 8
    image_size_for_model = (96, 96)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]

            smooth_x.append(x)
            smooth_y.append(y)
            smooth_w.append(w)
            smooth_h.append(h)

            # Apply moving average to smooth the coordinates
            avg_x = moving_average(smooth_x, smoothing_factor)
            avg_y = moving_average(smooth_y, smoothing_factor)
            avg_w = moving_average(smooth_w, smoothing_factor)
            avg_h = moving_average(smooth_h, smoothing_factor)

            face = frame[int(avg_y):int(avg_y)+int(avg_h), int(avg_x):int(avg_x)+int(avg_w)]
            keypoints = predict_and_scale_keypoints(face, model, image_size_for_model, avg_x, avg_y, avg_w, avg_h)

            # Draw the keypoints on the original frame
            for (kx, ky) in keypoints:
                cv2.circle(frame, (kx, ky), 2, (0, 255, 0), -1)
            
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

if __name__ == "__main__":
    main()