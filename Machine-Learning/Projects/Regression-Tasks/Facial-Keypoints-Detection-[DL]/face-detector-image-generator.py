import cv2
import numpy as np

def face_detection_and_capture(face_cascade_path, output_dir='.', img_size=(299, 299), smoothing_factor=8, offset=50):
    """
    Perform face detection and capture using the webcam.

    Args:
        face_cascade_path (str): Path of the Haar cascade XML file for face detection.
        output_dir (str): Directory to save captured images. Default is current directory.
        img_size (tuple): Size of the output image (width, height). Default is (299, 299).
        smoothing_factor (int): Number of frames to consider for moving average smoothing. Default is 8.
        offset (int): Margin around the detected face to be included in the captured image. Default is 50.

    Returns:
        None

    Example:
        face_cascade_path = './Assets/data/haarcascades/haarcascade_frontalface_default.xml'
        face_detection_and_capture(face_cascade_path=face_cascade_path, img_size=(299, 299), smoothing_factor=8, offset=50)
    """

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

    # Load Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Open camera
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Detection")
    img_counter = 0

    # Variables to store previous bounding box
    smooth_x, smooth_y, smooth_w, smooth_h = [], [], [], []

    while True:
        # Read frame from camera
        ret, frame = cam.read()
        if not ret:
            print("Failed to read frame from the camera. Exiting...")
            break

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            # Get the first detected face
            x, y, w, h = faces[0]

            # Smooth bounding box using exponential moving average
            smooth_x.append(x)
            smooth_y.append(y)
            smooth_w.append(w)
            smooth_h.append(h)

            # Apply moving average to smooth the coordinates
            avg_x = moving_average(smooth_x, smoothing_factor)
            avg_y = moving_average(smooth_y, smoothing_factor)
            avg_w = moving_average(smooth_w, smoothing_factor)
            avg_h = moving_average(smooth_h, smoothing_factor)

            # Crop the face area with an additional margin
            detected_face = frame[
                int(max(0, avg_y - offset)):int(min(avg_y + avg_h + offset, frame.shape[0])),
                int(max(0, avg_x - offset)):int(min(avg_x + avg_w + offset, frame.shape[1]))
            ]

            if detected_face.size > 0:
                # Resize the cropped image to match the specified size
                detected_face = cv2.resize(detected_face, img_size)
                # Show the detected face in a window named "Face Detection"
                cv2.imshow("Face Detection", detected_face)

        # Wait for user input
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # 'q' pressed
            print("Exiting the program.")
            break
        elif k == ord('s'):  # 's' pressed
            img_name = f"{output_dir}/opencv_frame_{img_counter}.jpg"
            cv2.imwrite(img_name, detected_face)
            print(f"{img_name} saved!")
            img_counter += 1

    # Release camera and close windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    output_path='C:/Workspace/Python/Machine-Learning/Documentations/Computer-Vision/Facial-Detection-Using-OpenCV-and-CNN/Assets/Images/'
    face_detection_and_capture(face_cascade_path=face_cascade_path, output_dir=output_path, img_size=(299, 299), smoothing_factor=10, offset=50)