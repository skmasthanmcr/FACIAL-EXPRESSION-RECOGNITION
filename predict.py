import numpy as np
import tensorflow as tf
import cv2
import time

# Load the trained model
model = tf.keras.models.load_model('emotion_recognition_model.keras')

def names(number):
    if number == 0:
        return 'happy'
    elif number == 1:
        return 'angry'
    elif number == 2:
        return 'neutral'
    elif number == 3:
        return 'sad'
    elif number == 4:
        return 'surprise'
    else:
        return 'Error in Prediction'

# Function to predict class from webcam input
def predict_from_webcam(model):
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    start_time = time.time()
    elapsed_time = 0
    
    while elapsed_time < 10:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert frame to RGB format and resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (128, 128))
        
        # Preprocess image
        img_array = np.array(frame_resized)
        img_array = img_array.reshape(1, 128, 128, 3)
        
        # Predict class
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        predicted_class_name = names(predicted_class)
        
        # Display predicted class and confidence
        cv2.putText(frame, f'Predicted: {predicted_class_name}, Confidence: {confidence*100:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Update elapsed time
        elapsed_time = time.time() - start_time
    
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

# Call the function to predict from webcam input
predict_from_webcam(model)
