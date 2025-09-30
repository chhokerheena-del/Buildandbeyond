import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("sign_language_model.h5")  # make sure your .h5 is in same folder

# Map class indices to letters (A-Y, skipping J and Z)
letters = [chr(i) for i in range(65, 91) if chr(i) not in ['J','Z']]

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28))
    norm = resized.astype("float32") / 255.0
    return norm.reshape(1,28,28,1)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    h, w = frame.shape[:2]
    box_size = 200
    x1, y1 = w//2 - box_size//2, h//2 - box_size//2
    x2, y2 = x1 + box_size, y1 + box_size
    roi = frame[y1:y2, x1:x2]

    if roi.size > 0:
        inp = preprocess(roi)
        preds = model.predict(inp, verbose=0)
        cls = np.argmax(preds)
        conf = float(np.max(preds))
        pred_letter = letters[cls] if conf > 0.5 else "?"
        
        # Draw ROI box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        # Display prediction
        cv2.putText(frame, f"{pred_letter} ({conf:.2f})",
                    (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    # Show the frame
    cv2.imshow("Sign Language Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
