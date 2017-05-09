import border_result as BorderResult
import numpy as np

class PreProcessingResult:

    def __init__(self):
        self._result = []

    def add_result(self, filename, center_of_mass, border, rotation):
        self._result.append(Result(filename, center_of_mass, border, rotation))

    def add_result(self, result):
        self._result.append(result)

    def print(self):
        print('{} Results found!'.format(len(self._result)))
        for r in self._result:
            print('{}: center-y {}, size {}'.format(r.filename, r.center_of_mass, r.border.get_size()))


class Result:
    def __init__(self,filename, center_of_mass, border, rotation):
        self.filename = filename
        self.center_of_mass = np.round(center_of_mass)
        self.border = border
        self.rotation = rotation