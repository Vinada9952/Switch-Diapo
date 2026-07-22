import os
import sys
import math
import time
import cv2
import pyautogui
import subprocess
import mediapipe as mp
from mediapipe.tasks.python import vision
import requests

SWIPE_FRACTION = 5
GESTURE_CONFIRM_FRAMES = 5
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

HOOK_CONTENT = '''from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

hiddenimports = collect_submodules('mediapipe.tasks')
hiddenimports += ['mediapipe.tasks.c']
datas = collect_data_files('mediapipe')
binaries = collect_dynamic_libs('mediapipe')
'''


def write_temp_hook(hook_path='hook-mediapipe.py'):
    with open(hook_path, 'w', encoding='utf-8') as f:
        f.write(HOOK_CONTENT)


def build_exe():
    hook_name = 'hook-mediapipe.py'
    write_temp_hook(hook_name)
    cmd = [
        sys.executable, '-m', 'PyInstaller', '--onefile', '--noconfirm', '--clean',
        '--additional-hooks-dir=.', '--hidden-import=mediapipe.tasks.c',
        '--add-data', 'hand_landmarker.task;.', os.path.basename(__file__)
    ]
    subprocess.run(cmd)
    try:
        os.remove(hook_name)
    except OSError:
        pass


def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)


def detect_gesture(world_landmarks):
    mcps = [world_landmarks[i] for i in (2, 5, 9, 13, 17)]
    palm = type('P', (), {
        'x': sum(m.x for m in mcps) / 5,
        'y': sum(m.y for m in mcps) / 5,
        'z': sum(m.z for m in mcps) / 5,
    })()
    tips = [world_landmarks[i] for i in (4, 8, 12, 16, 20)]
    return 'poing' if sum(distance(tip, palm) < 0.08 for tip in tips) == 5 else 'ouverte'


def create_landmarker():
    # Dossier sans accents/espaces pour éviter les problèmes de chemin Unicode
    # avec le moteur C++ de MediaPipe sous Windows (le dossier "École" pose problème)
    model_dir = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser('~')), 'switch_diapo_models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'hand_landmarker.task')
    model_url = (
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    )
    if not os.path.exists(model_path):
        print(f"Model '{model_path}' not found, downloading from {model_url}...")
        try:
            resp = requests.get(model_url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print('Failed to download hand_landmarker.task:', e)
            raise RuntimeError(f'Could not download model from {model_url}') from e

    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=4,
    )
    return vision.HandLandmarker.create_from_options(options)


def hand_center_px(landmarks, w, h):
    if not landmarks:
        return None
    return (
        sum(lm.x for lm in landmarks) / len(landmarks) * w,
        sum(lm.y for lm in landmarks) / len(landmarks) * h,
    )


def draw_hand(frame, hand, w, h, color):
    for lm in hand:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 5, color, -1)
    for start, end in HAND_CONNECTIONS:
        a, b = hand[start], hand[end]
        cv2.line(
            frame,
            (int(a.x * w), int(a.y * h)),
            (int(b.x * w), int(b.y * h)),
            color,
            2,
        )


def show_progress(step_name, index, total, width=30):
    progress = int(index / total * width)
    bar = '#' * progress + '-' * (width - progress)
    print(f'[{bar}] {index}/{total} {step_name}', end='\r', flush=True)


def ask_camera_index():
    raw = input('Camera index (default 0) : ').strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        print('Entrée invalide, utilisation de la caméra 0.')
        return 0


def main():
    show_progress('Initialisation du programme', 1, 3)
    time.sleep(0.2)

    show_progress('Chargement du modèle MediaPipe', 2, 3)
    landmarker = create_landmarker()
    time.sleep(0.2)

    show_progress('Prêt à démarrer', 3, 3)
    time.sleep(0.2)
    print()

    i = ask_camera_index()

    ip_adress = input("IP address (default http://127.0.0.1:9952): ").strip()
    if not ip_adress:
        ip_adress = 'http://127.0.0.1:9952'
    elif not ip_adress.startswith(('http://', 'https://')):
        ip_adress = 'http://' + ip_adress
    ip_adress = ip_adress.rstrip('/')

    print( f"using camera {i}" )

    cap = cv2.VideoCapture(i)
    timestamp_ms = 0
    gesture_candidate = None
    candidate_count = 0
    confirmed_gesture = None
    last_confirmed = None
    initial_x = None
    threshold = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(image, timestamp_ms)
        timestamp_ms += 33
        gesture = None
        center = None

        if result.hand_landmarks and result.hand_world_landmarks:
            best_index = min(
                range(len(result.hand_landmarks)),
                key=lambda i: sum(lm.y for lm in result.hand_landmarks[i]) / len(result.hand_landmarks[i]),
            )
            hand = result.hand_landmarks[best_index]
            world_hand = result.hand_world_landmarks[best_index]
            draw_hand(frame, hand, w, h, (0, 255, 0))
            gesture = detect_gesture(world_hand)
            center = hand_center_px(hand, w, h)
            cv2.putText(frame, f'? {gesture}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 100, 100), 1)

        if gesture is not None:
            if gesture == gesture_candidate:
                candidate_count += 1
            else:
                gesture_candidate = gesture
                candidate_count = 1
            if candidate_count >= GESTURE_CONFIRM_FRAMES:
                confirmed_gesture = gesture_candidate

        if confirmed_gesture:
            cv2.putText(frame, confirmed_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)

        if confirmed_gesture != last_confirmed:
            if confirmed_gesture == 'poing' and center is not None:
                initial_x = center[0]
                threshold = w // SWIPE_FRACTION
            elif confirmed_gesture == 'ouverte' and initial_x is not None and center is not None:
                final_x = center[0]
                if abs(initial_x - final_x) > threshold:
                    if initial_x - final_x > 0:
                        requests.get( f"{ip_adress}/switch-diapo" )
                    else:
                        requests.get( f"{ip_adress}/back-diapo" )
                    # pyautogui.press('space' if initial_x - final_x > 0 else 'backspace')
                initial_x = None
            last_confirmed = confirmed_gesture

        if initial_x is not None:
            lx = int(initial_x - threshold)
            rx = int(initial_x + threshold)
            cv2.line(frame, (lx, 0), (lx, h), (255, 0, 0), 2)
            cv2.line(frame, (rx, 0), (rx, h), (255, 0, 0), 2)

        cv2.imshow('Hand Skeleton', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == '__main__':
    if '--build-exe' in sys.argv:
        build_exe()
    else:
        main()
