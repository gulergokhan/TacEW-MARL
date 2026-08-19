class Scout:
    def __init__(self, position):
        self.position = position
        self.alive = True

    def move(self, position):
        self.position = position

    def destroy(self):
        self.alive = False