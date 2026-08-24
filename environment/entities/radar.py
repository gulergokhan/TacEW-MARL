class Radar:
    def __init__(self, position, detection_range):
        self.position = position
        self.detection_range = detection_range
        self.state = "SAFE"

    def detect(self, target_position):
        distance = abs(
            self.position[0] - target_position[0]
        ) + abs(
            self.position[1] - target_position[1]
        )

        return distance <= self.detection_range