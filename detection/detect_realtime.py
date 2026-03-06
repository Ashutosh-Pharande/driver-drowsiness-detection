from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

from detection.alarm import start_alarm, stop_alarm


def preprocess_eye_roi(eye_roi_bgr):
    """
    Convert eye ROI into CNN input format:
    - grayscale
    - resize to 24x24
    - normalize
    - reshape to (1, 24, 24, 1)
    """
    gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (24, 24))
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, 24, 24, 1)


def main():
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "models" / "cnn_model.h5"

    print("Loading trained model...")
    model = tf.keras.models.load_model(str(model_path))
    print("Model loaded successfully.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # 🔥 Create resizable window
    window_name = "Driver Drowsiness Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    closed_frames = 0
    threshold_frames = 5  # responsive detection
    fullscreen = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        eye_state = "NO FACE"
        drowsy = False

        if len(faces) > 0:
            x, y, w, h = faces[0]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)

            if len(eyes) > 0:
                ex, ey, ew, eh = eyes[0]

                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                x_in = preprocess_eye_roi(eye_roi)

                pred = float(model.predict(x_in, verbose=0)[0][0])

                # closed = 0, open = 1
                if pred < 0.5:
                    eye_state = "CLOSED"
                    closed_frames += 1
                else:
                    eye_state = "OPEN"
                    closed_frames = max(0, closed_frames - 1)

                if closed_frames >= threshold_frames:
                    drowsy = True

                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            else:
                closed_frames = max(0, closed_frames - 1)
        else:
            closed_frames = 0

        # Alarm logic
        if drowsy:
            start_alarm()
        else:
            stop_alarm()

        # Display text
        cv2.putText(
            frame,
            f"Eye: {eye_state}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        if drowsy:
            cv2.putText(
                frame,
                "DROWSY!",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord("q") or key == 27:
            break

        # Toggle fullscreen with 'f'
        if key == ord("f"):
            fullscreen = not fullscreen
            if fullscreen:
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )
            else:
                cv2.setWindowProperty(
                    window_name,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )

    cap.release()
    cv2.destroyAllWindows()
    stop_alarm()


if __name__ == "__main__":
    main()