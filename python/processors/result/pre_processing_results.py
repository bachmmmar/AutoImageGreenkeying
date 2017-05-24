import os
from .pre_processing_result import PreProcessingResult

class PreProcessingResults:

    def __init__(self):
        self._result = []

    def add_result(self, result :PreProcessingResult):
        #if type(result) is PreProcessingResult:
        self._result.append(result)
        #else:
        #    raise Exception('Result has wrong type! ({} vs. PreProcessingResult)'.format(type(result)))

    def add_result(self, result):
        self._result.append(result)

    def print(self):
        print('{} Results found!'.format(len(self._result)))
        for r in self._result:
            print('{}: center-y {}, size {}'.format(r.filename_in, r.center_of_mass, r.border.get_size()))

    def save(self, filepath):
        file = open(filepath, 'w')
        for r in self._result:
            file.write(r.as_string() + os.linesep)

        file.close()

    def load(self, filepath):
        with open(filepath, 'r') as file:
            for line in file:
                result = PreProcessingResult.fromstr(line)
                self.add_result(result)


