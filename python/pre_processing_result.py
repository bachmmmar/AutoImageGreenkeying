import border_result as BorderResult
import numpy as np

class PreProcessingResult:

    def __init__(self):
        self._result = []

    def add_result(self, filename_in, filename_out, center_of_mass, border, rotation):
        self._result.append(Result(filename_in, filename_out, center_of_mass, border, rotation))

    def add_result(self, result):
        self._result.append(result)

    def print(self):
        print('{} Results found!'.format(len(self._result)))
        for r in self._result:
            print('{}: center-y {}, size {}'.format(r.filename_in, r.center_of_mass, r.border.get_size()))


class Result:
    def __init__(self,filename_in, filename_out, center_of_mass, border, rotation):
        self.filename_in = filename_in
        self.filename_out = filename_out
        self.center_of_mass = np.round(center_of_mass)
        self.border = border
        self.rotation = rotation