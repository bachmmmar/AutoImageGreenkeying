
class BorderResult:
    def __init__(self, left, top, width, height):
        self._left = left
        self._top = top
        self._width = width
        self._height = height

    def get_size(self):
        return (self._width, self._height)

    def get_x_range(self):
        x1 = self._left
        x2 = x1 + self._width

        return x1, x2

    def get_y_range(self):
        y1 = self._top
        y2 = y1 + self._height

        return y1, y2

    def get_ranges(self):
        x1, x2 = self.get_x_range()
        y1, y2 = self.get_y_range()
        return x1, x2, y1, y2
