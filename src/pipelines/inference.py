import cv2
import torch
import mediapipe as mp
import os
from src.models.sign_model import ASLClassifier
from src.data.preprocess import extract_landmarks

def run_inference(model_path="models/best_model.pth"):
    # Initialize Model
    model = ASLClassifier()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model {model_path} not found. Using untrained weights.")
    
    model.eval()
    
    # ASL Alphabet A-Z
    classes = [chr(i) for i in range(ord('A'), ord('Z')+1)]
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=1, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    print("Starting webcam... Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for a selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Extract landmarks
        landmarks = extract_landmarks(frame, hands)
        
        if landmarks is not None:
            # Draw landmarks on frame
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            
            # Prepare tensor and predict
            tensor = torch.FloatTensor(landmarks).unsqueeze(0)
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                conf, predicted = torch.max(probs, 1)
                
            label = classes[predicted.item()]
            confidence = conf.item()
            
            # Display prediction
            text = f'{label} ({confidence:.2%})'
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            
            cv2.rectangle(frame, (10, 10), (300, 70), (0, 0, 0), -1)
            cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
        cv2.imshow('ASL Real-time Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference()
