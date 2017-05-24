from .border_result import BorderResult

class PreProcessingResult:

    def __init__(self,filename_in, filename_out, center_of_mass, border : BorderResult, rotation: float):
        self.filename_in = filename_in
        self.filename_out = filename_out
        self.center_of_mass = center_of_mass
        self.border = border
        self.rotation = rotation

    @classmethod
    def fromstr(cls, text:str):
        values = text.split(",")
        if len(values) != 9:
            raise Exception("Couldn't initialize PreProcessingResult from string ({})".format(text))
        filename_in = values[0]
        filename_out = values[1]
        center_of_mass = (int(values[2]), int(values[3]))
        border = BorderResult(int(values[4]), int(values[5]), int(values[6]), int(values[7]))
        rotation = float(values[8])
        return cls(filename_in, filename_out, center_of_mass, border, rotation)

    def as_string(self):
        cx, cy = self.center_of_mass
        return '{0},{1},{2},{3},{4},{5},{6},{7},{8}'.format(\
            self.filename_in, self.filename_out, \
            cx, cy, \
            self.border._left, self.border._top, \
            self.border._width, self.border._height, \
            self.rotation)