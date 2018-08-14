import cProfile

import numpy as np
import pandas as pd
from implementations_bachelor_thesis_vr.svm_predictor import Svm_Predictor
from terminaltables import AsciiTable
import inspect

TRAINING_INDEX_END = 4000
TESTING_INDEX_END = None

class Tester:
    @staticmethod
    def kernel(x, z):
        return np.exp(-20*np.dot(x-z, x-z))

    def __init__(self):
        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=np.nan)

        print("Loading data...")

        df = pd.read_csv('hmeq/hmeq_prepared.csv').sample(frac=1)

        x_data = df.drop('BAD', axis=1).values
        y_data = df['BAD'].values

        self.inputs_train = x_data[ : TRAINING_INDEX_END, ]
        self.outputs_train = y_data[ : TRAINING_INDEX_END, ]

        self.inputs_test = x_data[TRAINING_INDEX_END : TESTING_INDEX_END, ]
        self.outputs_test = y_data[TRAINING_INDEX_END : TESTING_INDEX_END, ]

    def print_parameters(self):
        print("Kernel:")
        print(inspect.getsource(self.kernel))
        print("Training index end:", TRAINING_INDEX_END)
        print("Testing index end:", TESTING_INDEX_END)

    def test_and_print(self):
        self.__calculate_classifier()

        prediction_stats, truth_stats = self.__test()
        self.__print_test_results(prediction_stats, truth_stats)

    def __calculate_classifier(self):
        pr = cProfile.Profile()
        pr.enable()
        self.classifier = Svm_Predictor(self.inputs_train, self.outputs_train)
        self.classifier.train(C=10, max_iterations=200, warmup_iterations=0)
        pr.disable()
        pr.print_stats(sort="tottime")
        #self.classifier.print_diagnostics()

    def __test(self):
        print("Testing...")

        # Format: ["Prediction", "Total amount guessed", "Correct guesses", "Wrong guesses"]
        prediction_stats = [[i, 0, 0, 0] for i in range(2)]

        # Format: ["True value", "Total amount presented", "Correctly identified", "Misidentified"]
        truth_stats = [[i, 0, 0, 0] for i in range(2)]

        for i in range(len(self.inputs_test)):
            prediction = int(0.5*self.classifier.predict_label(self.inputs_test[i]) + 0.5)
            truth = int(0.5*self.outputs_test[i] + 0.5)

            prediction_stats[prediction][1] += 1
            truth_stats[truth][1] += 1

            if prediction == truth:
                prediction_stats[prediction][2] += 1
                truth_stats[truth][2] += 1
            else:
                prediction_stats[prediction][3] += 1
                truth_stats[truth][3] += 1

        return (prediction_stats, truth_stats)

    def __print_test_results(self, prediction_stats, truth_stats):
        for stats_list in (prediction_stats, truth_stats):
            summary = [sum(x) for x in zip(*stats_list)]
            summary[0] = "All"
            stats_list.append(summary)

            for line in stats_list:
                for i in [2, 3]:
                    perc = "??" if line[1] == 0 else str(100*line[i]/line[1])
                    line[i] = str(line[i]) + " (" + perc + "%)"

        prediction_table = [["Prediction", "Total guesses", "Correct guesses", "Wrong guesses"]] + prediction_stats
        truth_table = [["True value", "Total amount presented", "Correctly identified", "Misidentified"]] + truth_stats
        print(AsciiTable(prediction_table).table)
        print(AsciiTable(truth_table).table)

if __name__ == '__main__':
    T = Tester()
    T.print_parameters()
    T.test_and_print()
