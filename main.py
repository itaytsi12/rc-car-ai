import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def is_open_hand(hand_landmarks):
    fingers = []

    fingers.append(hand_landmarks[4].x < hand_landmarks[3].x)

    fingers.append(hand_landmarks[8].y < hand_landmarks[6].y)
    fingers.append(hand_landmarks[12].y < hand_landmarks[10].y)
    fingers.append(hand_landmarks[16].y < hand_landmarks[14].y)
    fingers.append(hand_landmarks[20].y < hand_landmarks[18].y)

    return fingers.count(True) >= 4

cap = cv2.VideoCapture(0)

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
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

            index_finger = hand_landmarks[8]
            finger_x = int(index_finger.x * w)
            finger_y = int(index_finger.y * h)

            center_x = w // 2
            center_y = h // 2

            if is_open_hand(hand_landmarks):
                command = "STOP"
            else:
                dx = finger_x - center_x
                dy = finger_y - center_y

                if abs(dx) > abs(dy):
                    if dx > 50:
                        command = "RIGHT"
                    elif dx < -50:
                        command = "LEFT"
                else:
                    if dy < -50:
                        command = "FORWARD"
                    elif dy > 50:
                        command = "BACKWARD"

            cv2.circle(img, (finger_x, finger_y), 12, (0, 0, 255), -1)
            cv2.circle(img, (center_x, center_y), 8, (255, 0, 0), -1)

    cv2.putText(img, command, (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.imshow("Hand Detection", img)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("/") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()