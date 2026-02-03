import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import pyautogui
import subprocess
import sys

# Constants
SWIPE_FRACTION = 4  # Fraction of screen width to move for swipe action (1/SWIPE_FRACTION)

# Define the hand connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),  # thumb
    (0,5),(5,6),(6,7),(7,8),  # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),  # ring
    (0,17),(17,18),(18,19),(19,20),  # pinky
    (5,9),(9,13),(13,17)  # palm
]

hand_first_position = 0
hand_last_position = 0

center = None

gesture = None
last_gesture = gesture

initial_x = None
threshold = 0

# Function to calculate distance
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

# Function to get average position of all hand landmarks in pixels
def get_hand_center_pixels(landmarks, w, h):
    if not landmarks:
        return None
    avg_x = sum(lm.x * w for lm in landmarks) / len(landmarks)
    avg_y = sum(lm.y * h for lm in landmarks) / len(landmarks)
    return avg_x, avg_y

# Function to detect hand gesture
def detect_gesture(world_landmarks):
    # MCP joints: thumb 2, index 5, middle 9, ring 13, pinky 17
    mcps = [world_landmarks[2], world_landmarks[5], world_landmarks[9], world_landmarks[13], world_landmarks[17]]
    palm_center_x = sum(m.x for m in mcps) / 5
    palm_center_y = sum(m.y for m in mcps) / 5
    palm_center_z = sum(m.z for m in mcps) / 5
    palm_center = type('Point', (), {'x': palm_center_x, 'y': palm_center_y, 'z': palm_center_z})()
    
    # Fingertips: 4,8,12,16,20
    tips = [world_landmarks[4], world_landmarks[8], world_landmarks[12], world_landmarks[16], world_landmarks[20]]
    distances = [distance(tip, palm_center) for tip in tips]
    
    # Thresholds (adjust as needed)
    threshold_close = 0.08  # fist

    close_count = sum(1 for d in distances if d < threshold_close)
    # print( f"{close_count=}" )
    if close_count == 5:
        return "poing"
    else:
        return "ouverte"

# Initialize MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO)

try:
    landmarker = HandLandmarker.create_from_options(options)
except FileNotFoundError:
    subprocess.run( "curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task".split( " " ) )
    landmarker = HandLandmarker.create_from_options(options)

# Capture video from webcam
cap = cv2.VideoCapture( int( input( "Camera index (default 0) : " ) ) )

timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MP Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Detect
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33  # approx 30 fps
    
    # Draw landmarks
    if result.hand_landmarks and result.hand_world_landmarks:
        for hand, world_hand in zip(result.hand_landmarks, result.hand_world_landmarks):
            # Draw points
            for lm in hand:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            # Draw connections
            for start, end in HAND_CONNECTIONS:
                start_lm = hand[start]
                end_lm = hand[end]
                start_pt = (int(start_lm.x * w), int(start_lm.y * h))
                end_pt = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)
            
            # Draw threshold lines if fist
            if initial_x is not None:
                left_threshold = int(initial_x - threshold)
                right_threshold = int(initial_x + threshold)
                cv2.line(frame, (left_threshold, 0), (left_threshold, h), (255, 0, 0), 2)  # Blue line left
                cv2.line(frame, (right_threshold, 0), (right_threshold, h), (255, 0, 0), 2)  # Blue line right
            
            # Detect gesture
            gesture = detect_gesture( world_hand )
            
            # Get hand center in pixels
            center = get_hand_center_pixels(hand, w, h)
            if center:
                x, y = center
                print( f"{x=} {y=}" )
            
            # Display gesture
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Display the frame
    cv2.imshow('Hand Skeleton', frame)

    if last_gesture != gesture and gesture == "poing":
        hand_first_position = x
        initial_x = x
        threshold = frame.shape[1] // SWIPE_FRACTION
    
    if last_gesture != gesture and gesture == "ouverte":
        hand_last_position = x
        if abs( hand_first_position - hand_last_position ) > frame.shape[1] // SWIPE_FRACTION:
            if hand_first_position - hand_last_position > 0:
                pyautogui.press( "space" )
            else:
                pyautogui.press( "backspace" )
        initial_x = None  # Reset after action
    
    last_gesture = gesture
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()