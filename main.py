import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from hand import Hand

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
hand = Hand()
def get_command(hand_landmarks, finger_x, finger_y, center_x, center_y):
    command = "STOP"
    threshold = 20
    diagonal_ratio = 0.6

    if hand.is_open_hand(hand_landmarks):
        command = "STOP"
    else:
        dx = finger_x - center_x
        dy = finger_y - center_y

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx < threshold and abs_dy < threshold:
            command = "STOP"

        elif abs_dx > threshold and abs_dy > threshold and min(abs_dx, abs_dy) / max(abs_dx, abs_dy) > diagonal_ratio:
            if dx > 0 and dy < 0:
                command = "RIGHT/UP"
            elif dx > 0 and dy > 0:
                command = "RIGHT/DOWN"
            elif dx < 0 and dy < 0:
                command = "LEFT/UP"
            elif dx < 0 and dy > 0:
                command = "LEFT/DOWN"

        elif abs_dx > abs_dy:
            if dx > threshold:
                command = "RIGHT"
            elif dx < -threshold:
                command = "LEFT"

        else:
            if dy < -threshold:
                command = "FORWARD"
            elif dy > threshold:
                command = "BACKWARD"

    return command

def get_hand_area(hand_landmarks, w, h):
    index_tip = hand_landmarks[8]
    palm = hand_landmarks[0]
    return abs((index_tip.x - palm.x) * w * (index_tip.y - palm.y) * h)


def get_biggest_hand(hands_landmarks, w, h):
    biggest_hand = hands_landmarks[0]
    biggest_area = get_hand_area(biggest_hand, w, h)

    for hand_landmarks in hands_landmarks:
        area = get_hand_area(hand_landmarks, w, h)

        if area > biggest_area:
            biggest_area = area
            biggest_hand = hand_landmarks

    return biggest_hand

def get_hand_position(hand_landmarks, w, h):
    x = int(hand_landmarks[9].x * w)
    y = int(hand_landmarks[9].y * h)
    return x, y


def get_distance(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def choose_active_hand(hands_landmarks, w, h, last_hand_x, last_hand_y):
    if last_hand_x is None or last_hand_y is None:
        return get_biggest_hand(hands_landmarks, w, h)

    closest_hand = hands_landmarks[0]
    closest_x, closest_y = get_hand_position(closest_hand, w, h)
    closest_distance = get_distance(closest_x, closest_y, last_hand_x, last_hand_y)

    for hand_landmarks in hands_landmarks:
        hand_x, hand_y = get_hand_position(hand_landmarks, w, h)
        distance = get_distance(hand_x, hand_y, last_hand_x, last_hand_y)

        if distance < closest_distance:
            closest_distance = distance
            closest_hand = hand_landmarks

    return closest_hand

last_hand_x = None
last_hand_y = None
last_command = None

while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    detection_result = detector.detect(mp_image)

    h, w, _ = img.shape
    command = "STOP"
    if detection_result.hand_landmarks:
        hand_landmarks = choose_active_hand(detection_result.hand_landmarks, w, h, last_hand_x, last_hand_y)
        last_hand_x = int(hand_landmarks[9].x * w)
        last_hand_y = int(hand_landmarks[9].y * h)
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

        index_finger_id = hand.finger_tips[0]
        index_finger = hand_landmarks[index_finger_id]

        finger_x = int(index_finger.x * w)
        finger_y = int(index_finger.y * h)

        center_x = int(hand_landmarks[0].x * w)
        center_y = int(hand_landmarks[0].y * h)

        cv2.circle(img, (finger_x, finger_y), 12, (0, 0, 255), -1)
        cv2.circle(img, (center_x, center_y), 8, (255, 0, 0), -1)

        command = get_command(hand_landmarks, finger_x, finger_y, center_x, center_y)
        if command != last_command:
            last_command = command

    cv2.putText(img, command, (20, 90),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    cv2.imshow("Hand Gesture Control", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("/") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()