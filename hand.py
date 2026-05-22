from Finger import Finger
class Hand:
    def __init__(self):
        self.fingers = [
            Finger("thumb"),
            Finger("index"),
            Finger("middle"),
            Finger("ring"),
            Finger("pinky")
        ]

    def open_hand(self):
        for finger in self.fingers:
            finger.open_finger()

    def close_hand(self):
        for finger in self.fingers:
            finger.close_finger()

    def count_open_fingers(self):
        count = 0

        for finger in self.fingers:
            if finger.is_open():
                count += 1

        return count

    def get_finger(self, index):
        return self.fingers[index]