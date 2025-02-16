import cv2
import cv2.data
import numpy as np
import tf_keras

# Load trained model
model = tf_keras.models.load_model("./facerec_model.h5")
# Load class labels
class_labels = ["vishwa"]

# Load face detector
cas_path = cv2.data.haarcascades + "./haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cas_path)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224)) / 255.0  # Normalize
        face = np.expand_dims(face, axis=0)

        predictions = model.predict(face)
        label = np.argmax(predictions)
        confidence = np.max(predictions)

        if confidence > 0.8:
            name = class_labels[label]
        else:
            name = "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
