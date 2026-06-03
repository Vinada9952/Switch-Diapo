import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import pyautogui
import subprocess
import sys

# Constants
SWIPE_FRACTION = 5          # Fraction of screen width to move for swipe action (1/SWIPE_FRACTION)
GESTURE_CONFIRM_FRAMES = 5  # Nombre de frames consécutives pour confirmer un changement de geste

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

gesture = None            # geste brut du frame courant (peut être None)
confirmed_gesture = None  # geste stable après confirmation
last_confirmed_gesture = None

gesture_candidate = None  # geste en cours de confirmation
gesture_candidate_count = 0

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
    running_mode=VisionRunningMode.VIDEO,
    num_hands=4)  # Détecter jusqu'à 4 mains (2 personnes max)

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

    gesture = None
    center = None

    if result.hand_landmarks and result.hand_world_landmarks:
        # Calculer le centre (x, y normalisés) de chaque main détectée
        hand_centers = [
            (
                sum(lm.x for lm in lms) / len(lms),
                sum(lm.y for lm in lms) / len(lms),
            )
            for lms in result.hand_landmarks
        ]

        # Regrouper les mains par personne :
        # deux mains appartiennent à la même personne si leur distance
        # horizontale est < 40 % de la largeur de l'image (coordonnées normalisées).
        SAME_PERSON_X_THRESHOLD = 0.4
        n = len(hand_centers)
        person_groups = []   # liste de listes d'indices de mains
        assigned = [False] * n

        for i in range(n):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            for j in range(i + 1, n):
                if not assigned[j]:
                    if abs(hand_centers[i][0] - hand_centers[j][0]) < SAME_PERSON_X_THRESHOLD:
                        group.append(j)
                        assigned[j] = True
            person_groups.append(group)

        # Choisir la personne dont le centre moyen est le plus haut (y minimal)
        def group_avg_y(group):
            return sum(hand_centers[i][1] for i in group) / len(group)

        best_group = min(person_groups, key=group_avg_y)

        # Parmi les mains de cette personne, prendre celle qui est la plus haute
        best_index = min(best_group, key=lambda i: hand_centers[i][1])

        hand       = result.hand_landmarks[best_index]
        world_hand = result.hand_world_landmarks[best_index]

        # Dessiner toutes les mains en gris (non sélectionnées)
        for i, h_lms in enumerate(result.hand_landmarks):
            if i == best_index:
                continue
            for lm in h_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (128, 128, 128), -1)
            for start, end in HAND_CONNECTIONS:
                s = h_lms[start]; e = h_lms[end]
                cv2.line(frame, (int(s.x*w), int(s.y*h)), (int(e.x*w), int(e.y*h)), (128, 128, 128), 1)

        # Dessiner la main sélectionnée en vert
        for lm in hand:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        for start, end in HAND_CONNECTIONS:
            start_lm = hand[start]
            end_lm   = hand[end]
            start_pt = (int(start_lm.x * w), int(start_lm.y * h))
            end_pt   = (int(end_lm.x   * w), int(end_lm.y   * h))
            cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

        # Draw threshold lines if fist
        if initial_x is not None:
            left_threshold  = int(initial_x - threshold)
            right_threshold = int(initial_x + threshold)
            cv2.line(frame, (left_threshold,  0), (left_threshold,  h), (255, 0, 0), 2)
            cv2.line(frame, (right_threshold, 0), (right_threshold, h), (255, 0, 0), 2)

        # Detect gesture
        gesture = detect_gesture(world_hand)

        # Get hand center in pixels
        center = get_hand_center_pixels(hand, w, h)
        if center:
            x, y = center
            print(f"{x=} {y=}")

        # Display gesture (brut, en rouge pâle — le confirmé s'affiche en vert après)
        cv2.putText(frame, f"? {gesture}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 100, 100), 1)

    # --- Confirmation du geste sur N frames consécutives ---
    # Un frame sans détection (gesture=None) ne réinitialise pas le compteur :
    # on ignore les dropouts isolés et on conserve le candidat en cours.
    if gesture is not None:
        if gesture == gesture_candidate:
            gesture_candidate_count += 1
        else:
            gesture_candidate = gesture
            gesture_candidate_count = 1

        if gesture_candidate_count >= GESTURE_CONFIRM_FRAMES:
            confirmed_gesture = gesture_candidate

    # Afficher le geste confirmé à l'écran (remplace l'affichage brut)
    if confirmed_gesture:
        cv2.putText(frame, confirmed_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

    # Display the frame
    cv2.imshow('Hand Skeleton', frame)

    if last_confirmed_gesture != confirmed_gesture and confirmed_gesture == "poing":
        hand_first_position = x
        initial_x = x
        threshold = frame.shape[1] // SWIPE_FRACTION

    if last_confirmed_gesture != confirmed_gesture and confirmed_gesture == "ouverte":
        hand_last_position = x
        if abs(hand_first_position - hand_last_position) > frame.shape[1] // SWIPE_FRACTION:
            if hand_first_position - hand_last_position > 0:
                pyautogui.press("space")
            else:
                pyautogui.press("backspace")
        initial_x = None  # Reset after action

    last_confirmed_gesture = confirmed_gesture

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()