import cv2
import numpy as np
import tensorflow as tf
from face import blazeFaceDetector


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter 

def preprocess_image(image, input_size, detector):
    original_image = image.copy()
    img_res = detector.detectFaces(original_image)

    if len(img_res.boxes) > 0:
        bbox = img_res.boxes[0]  # Get لاbox
        x1, y1, x2, y2 = bbox
        x1 = int(x1 * image.shape[1])
        x2 = int(x2 * image.shape[1])
        y1 = int(y1 * image.shape[0])+2
        y2 = int(y2 * image.shape[0])

        # Adding  margin
        w = x2 - x1
        h = y2 - y1
        margin_h = int(0.25 * h)
        margin_w = int(0.25 * w)

        y1 = max(0, y1 - margin_h)
        y2 = min(original_image.shape[0], y2 + margin_h)
        x1 = max(0, x1 - margin_w)
        x2 = min(original_image.shape[1], x2 + margin_w)
        
        extended_image = original_image[y1:y2, x1:x2]

        if extended_image.size == 0:
            print("Extended crop results in empty image. Check bounding box and margin calculations.")
            return None, original_image, None

        # Resize the extended image to the input size required by the model
        image = cv2.resize(extended_image, input_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = (image - 127.5) / 127.5  # Normalize the image (0,1)
        image = np.expand_dims(image, axis=0)
        return image, original_image, (x1, y1, x2, y2)
    return None, original_image, None
def run_inference(interpreter, frame):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], frame)
    interpreter.invoke()

    landmarks = interpreter.get_tensor(output_details[0]['index'])
    face_flag = interpreter.get_tensor(output_details[1]['index'])
    return landmarks, face_flag
def process_output(landmarks, face_flag, preprocessed_frame, original_image, input_size, bbox):
    if preprocessed_frame is None:
        print("No face processed.")
        return original_image

    if face_flag > 0.5:
        landmarks = landmarks.reshape(-1, 3)


        # Assuming the preprocessed_frame is the image returned by preprocess_image and is normalized and resized
        img_h, img_w, _ = preprocessed_frame.shape[1:4]

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        # Scale and translate the landmark coordinates back to the original image
        for (x, y, z) in landmarks:
            x = int(x1 + (x * w / input_size[0]))
            y = int(y1 + (y * h / input_size[1]))
            cv2.circle(original_image, (x, y), 1, (0, 255, 0), -1)

        return original_image


def main():
    model_path = "C:\\Users\\edrie\\Downloads\\face_mesh.tflite"
    input_size = (192, 192)
    interpreter = load_tflite_model(model_path)
    detector = blazeFaceDetector("front", 0.7, 0.3)

    cap = cv2.VideoCapture("C:\\Users\\edrie\\Downloads\\ved.mp4")
    #cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            preprocessed_frame, original_frame, bbox = preprocess_image(frame, input_size, detector)
            if preprocessed_frame is not None:
                landmarks, face_flag = run_inference(interpreter, preprocessed_frame)
                processed_frame = process_output(landmarks, face_flag, preprocessed_frame, original_frame, input_size, bbox)
                cv2.imshow('Real-time Face Landmarks', processed_frame)
            else:
                cv2.imshow('Real-time Face Landmarks', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()