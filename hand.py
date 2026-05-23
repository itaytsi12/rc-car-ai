class Hand:
    def __init__(self):
        self.finger_tips = [8, 12, 16, 20]
        self.finger_joints = [6, 10, 14, 18]

    def count_open_fingers(self, hand_landmarks):
        count = 0

        for i in range(len(self.finger_tips)):
            tip = self.finger_tips[i]
            joint = self.finger_joints[i]

            if hand_landmarks[tip].y < hand_landmarks[joint].y:
                count += 1

        return count

    def is_open_hand(self, hand_landmarks):
        return self.count_open_fingers(hand_landmarks) >= 4