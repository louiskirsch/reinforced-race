class Action:
    def __init__(self, vertical: int, horizontal: int):
        if vertical > 1 or vertical < -1:
            raise ValueError('Vertical must be -1, 0 or 1')
        if horizontal > 1 or horizontal < -1:
            raise ValueError('Horizontal must be -1, 0 or 1')
        self.vertical = vertical
        self.horizontal = horizontal

    def __str__(self):
        return "horizontal: {}, vertical: {}".format(self.vertical, self.horizontal)