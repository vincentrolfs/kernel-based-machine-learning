import numpy as np
import implementations_bachelor_thesis_vr.binary_classifier as binary_classifier
from implementations_bachelor_thesis_vr.tests.mnist import mnist
from terminaltables import AsciiTable
import inspect

# Image size: 28*28 = 784

AMOUNT_TRAIN_INPUTS = 5000
AMOUNT_TESTS = 5000

class Tester:
    @staticmethod
    def kernel(x, y):
        return ((1/28)**3) * (28 + (1/28) * np.dot(x, y))**3

    def __init__(self):
        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=np.nan)

        print("Loading data...")

        x_train, y_train, x_test, y_test = mnist.load()

        indices_train = np.random.choice(len(y_train), AMOUNT_TRAIN_INPUTS, replace=False)
        self.inputs_train = 2*(-0.5 + (1/255) * x_train[indices_train,:])
        self.outputs_train = 2*(0.5 - np.sign(y_train[indices_train]))

        indices_test = np.random.choice(len(y_test), AMOUNT_TESTS, replace=False)
        self.inputs_test = 2*(-0.5 + (1/255) * x_test[indices_test,:])
        self.outputs_test = 2*(0.5 - np.sign(y_test[indices_test]))

    def print_parameters(self):
        print("KERNEL:")
        print(inspect.getsource(self.kernel))
        print("AMOUNT_TRAIN_INPUTS:", AMOUNT_TRAIN_INPUTS)
        print("AMOUNT_TESTS:", AMOUNT_TESTS)

    def test_and_print(self):
        self.__calculate_classifier()
        print("Testing...")
        self.__print_prediction_stats()

    def __calculate_classifier(self):
        self.classifier = binary_classifier.make_classifier(self.inputs_train, self.outputs_train, 10, self.kernel)

    def __print_prediction_stats(self):
        # Format: ["Prediction", "Total amount guessed", "Correct guesses", "Wrong guesses"]
        prediction_stats = [[i, 0, 0, 0] for i in range(2)]

        # Format: ["True value", "Total amount presented", "Correctly identified", "Misidentified"]
        truth_stats = [[i, 0, 0, 0] for i in range(2)]

        for i in range(len(self.inputs_test)):
            prediction = int(0.5*self.classifier(self.inputs_test[i]) + 0.5)
            truth = int(0.5*self.outputs_test[i] + 0.5)

            prediction_stats[prediction][1] += 1
            truth_stats[truth][1] += 1

            if prediction == truth:
                prediction_stats[prediction][2] += 1
                truth_stats[truth][2] += 1
            else:
                prediction_stats[prediction][3] += 1
                truth_stats[truth][3] += 1

        for stats_list in [prediction_stats, truth_stats]:
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
