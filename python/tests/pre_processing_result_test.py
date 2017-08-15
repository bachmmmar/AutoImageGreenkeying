import unittest
from processors import PreProcessingResults, PreProcessingResult, BorderResult, FaceResult

class PreProcessingResultTest(unittest.TestCase):
    @staticmethod
    def getComaSeparatedString(nelements):
        ser_cut = "1"
        for i in range(2, nelements+1):
            ser_cut += ",{}".format(i)
        return ser_cut

    @staticmethod
    def getCorrectPreProcessingResult():
        return PreProcessingResultTest.getComaSeparatedString(nelements=15)

    def test_deserialize_inaccurate_string_throws_exception(self):
        ser_cut = PreProcessingResultTest.getComaSeparatedString(nelements=13)
        self.assertRaises(Exception, PreProcessingResult.fromstr,ser_cut)

    def test_deserialize_correct_string(self):
        ser_cut = PreProcessingResultTest.getCorrectPreProcessingResult()
        cut = PreProcessingResult.fromstr(ser_cut)
        self.assertEqual(cut.rotation,9)

    def test_serialize_deserialize_yealds_to_same_result(self):
        cut = PreProcessingResult('fn','fout',(1,2),BorderResult(3,4,5,6),7)
        cut.face = FaceResult((10,11),(12,13),(14,15))

        ser_cut = cut.as_string()
        cut2 = PreProcessingResult.fromstr(ser_cut)

        #check some of the values
        self.assertEqual(cut.center_of_mass,cut2.center_of_mass)
        self.assertEqual(cut.rotation,cut2.rotation)

    def test_write_results_to_files(self):
        ser_res = PreProcessingResultTest.getCorrectPreProcessingResult()
        res = PreProcessingResult.fromstr(ser_res)
        cut = PreProcessingResults()
        cut.add_result(res)
        cut.add_result(res)
        cut.save("test_write.csv")

    def test_read_result_to_files(self):
        cut = PreProcessingResults()
        cut.load("test_write.csv")
        self.assertEqual(len(cut._result), 2)