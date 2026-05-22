class Finger:
    def __init__(self, name):
        self.name = name
        self.open = False

    def open_finger(self):
        self.open = True

    def close_finger(self):
        self.open = False

    def is_open(self):
        return self.open