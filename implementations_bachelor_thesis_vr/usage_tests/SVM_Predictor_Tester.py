import cProfile
import inspect

import numpy as np
from terminaltables import AsciiTable

from implementations_bachelor_thesis_vr.SVM_Predictor import SVM_Predictor


class SVM_Predictor_Tester:
    def __init__(self, x_data, y_data, training_set_size, testing_set_size, kernel, C, max_iterations,
                 warmup_iterations):
        self.x_data = x_data
        self.y_data = y_data

        self.training_set_size = training_set_size
        self.testing_set_size = testing_set_size

        self.kernel = kernel
        self.C = C
        self.max_iterations = max_iterations
        self.warmup_iterations = warmup_iterations

        self._split_train_test()

        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=np.nan)

    def _split_train_test(self):
        if self.training_set_size is None:
            self.training_set_size = len(self.x_data) - self.testing_set_size
        if self.testing_set_size is None:
            self.testing_set_size = len(self.x_data) - self.training_set_size

        assert (self.training_set_size + self.testing_set_size) <= len(self.x_data)

        self.x_train = self.x_data[: self.training_set_size, ]
        self.y_train = self.y_data[: self.training_set_size, ]

        self.x_test = self.x_data[self.training_set_size: (self.training_set_size + self.testing_set_size), ]
        self.y_test = self.y_data[self.training_set_size: (self.training_set_size + self.testing_set_size), ]

    def print_parameters(self):
        print("Amount of training examples:", len(self.x_train))
        print("Amount of testing examples:", len(self.x_test))
        print("Kernel:")
        print(inspect.getsource(self.kernel))
        print("C =", self.C)
        print("Max Iterations:", self.max_iterations)
        print("Warmup Iterations:", self.warmup_iterations)

    def run(self):
        self._calculate_classifier()

        prediction_stats, truth_stats = self._test()
        self._print_test_results(prediction_stats, truth_stats)

    def _calculate_classifier(self):
        print("Training...")
        pr = cProfile.Profile()
        pr.enable()
        self.predictor = SVM_Predictor(self.x_train, self.y_train)
        self.predictor.train(kernel=self.kernel, C=self.C, max_iterations=self.max_iterations,
                             warmup_iterations=self.warmup_iterations)
        pr.disable()
        pr.print_stats(sort="tottime")

    def _test(self):
        print("Testing...")

        # Format: ["Prediction", "Total amount predicted", "Amount predicted correctly", "Amount predicted incorrectly"]
        prediction_stats = [[i, 0, 0, 0] for i in range(2)]

        # Format: ["True value", "Total amount presented", "Amount correctly identified", "Amount misidentified"]
        truth_stats = [[i, 0, 0, 0] for i in range(2)]

        for i in range(len(self.x_test)):
            prediction_index = self._label_to_index(self.predictor.predict_label(self.x_test[i]))
            truth_index = self._label_to_index(self.y_test[i])

            prediction_stats[prediction_index][1] += 1
            truth_stats[truth_index][1] += 1

            if prediction_index == truth_index:
                prediction_stats[prediction_index][2] += 1
                truth_stats[truth_index][2] += 1
            else:
                prediction_stats[prediction_index][3] += 1
                truth_stats[truth_index][3] += 1

        return prediction_stats, truth_stats

    def _label_to_index(self, label):
        return int(0.5 * label + 0.5)

    def _print_test_results(self, prediction_stats, truth_stats):
        for stats_list in (prediction_stats, truth_stats):
            summary = [sum(x) for x in zip(*stats_list)]
            summary[0] = "All"
            stats_list.append(summary)
            self._add_percentages(stats_list)

        prediction_table = [["Prediction", "Total amount predicted", "Amount predicted correctly",
                             "Amount predicted incorrectly"]] + prediction_stats
        truth_table = [["True value", "Total amount presented", "Correctly identified", "Misidentified"]] + truth_stats
        print(AsciiTable(prediction_table).table)
        print(AsciiTable(truth_table).table)

    def _add_percentages(self, stats_list):
        for line in stats_list:
            for i in [2, 3]:
                perc = "??" if line[1] == 0 else str(100 * line[i] / line[1])
                line[i] = str(line[i]) + " (" + perc + "%)"
