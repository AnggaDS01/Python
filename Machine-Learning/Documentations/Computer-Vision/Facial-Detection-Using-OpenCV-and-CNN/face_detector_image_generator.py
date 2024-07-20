import cv2
import numpy as np

def face_detection_and_capture(face_cascade_path, output_dir='.', img_size=(299, 299), alpha=0.1, offset=50):
    """
    Perform face detection and capture using the webcam.

    Args:
        face_cascade_path (str): path of the facial cascade.
        output_dir (str): Directory to save captured images. Default is 'dataset'.
        img_size (int): Size of the output image. Default is 299.
        alpha (float): Smoothing factor for bounding box coordinates. Default is 0.1.
        offset (int): Margin around the detected face to be included in the captured image. Default is 50.

    Returns:
        None

    Example:
        face_cascade_path = './Assets/data/haarcascades/haarcascade_frontalface_default.xml'
        face_detection_and_capture(face_cascade_path=face_cascade_path, img_size=(299, 299), alpha=0.1, offset=50)
    """
    # Load Haar cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Open camera
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Face Detection")
    img_counter = 0

    # Variables to store previous bounding box
    prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0

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
            x = int(alpha * x + (1 - alpha) * prev_x)
            y = int(alpha * y + (1 - alpha) * prev_y)
            w = int(alpha * w + (1 - alpha) * prev_w)
            h = int(alpha * h + (1 - alpha) * prev_h)

            # Update previous bounding box
            prev_x, prev_y, prev_w, prev_h = x, y, w, h

            # Crop the face area with an additional margin
            detected_face = frame[max(0, y - offset):min(y + h + offset, frame.shape[0]),
                                max(0, x - offset):min(x + w + offset, frame.shape[1])]

            if detected_face.size > 0:
                # Resize the cropped image to match the specified size
                detected_face = cv2.resize(detected_face, (img_size[0], img_size[1]))
                # Show the white image in a window named "Face Detection"
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