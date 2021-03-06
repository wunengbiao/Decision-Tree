import DecisionTree
import unittest

training_datafile = "training_symbolic.csv"


class TestBestFeatureCalculation(unittest.TestCase):

    def setUp(self):
        print("Testing best-feature calculation on sample training file")
        self.dt = DecisionTree.DecisionTree(training_datafile = training_datafile,
                                            csv_class_column_index = 1,
                                            csv_columns_for_features = [2,3,4,5])
        self.dt.get_training_data()
        self.dt.calculate_first_order_probabilities()
        self.dt.calculate_class_priors()

    def test_best_feature_calculation(self):
        best = self.dt.best_feature_calculator([], 100)
        self.assertEqual(best[0], 'fatIntake')
        best = self.dt.best_feature_calculator(['smoking=heavy'], 100)
        self.assertEqual(best[0], 'fatIntake')
        best = self.dt.best_feature_calculator(['fatIntake=medium', 'exercising=never'], 100)
        self.assertEqual(best[0], 'smoking')

def getTestSuites(type):
    return unittest.TestSuite([
            unittest.makeSuite(TestBestFeatureCalculation, type)
                             ])                    
if __name__ == '__main__':
    unittest.main()

