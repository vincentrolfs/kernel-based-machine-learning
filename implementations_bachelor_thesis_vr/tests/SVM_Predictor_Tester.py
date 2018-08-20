import cProfile
import inspect
from collections import defaultdict

import numpy as np
from terminaltables import AsciiTable

from implementations_bachelor_thesis_vr.SVM_Predictor import SVM_Predictor


class SVM_Predictor_Tester:
    def __init__(self, x_train, y_train, x_test, y_test, kernel, C, max_iterations,
                 warmup_iterations, label_names=None):
        if label_names is None: label_names = {1: '1', -1: '-1'}

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.kernel = kernel
        self.C = C
        self.max_iterations = max_iterations
        self.warmup_iterations = warmup_iterations

        self.label_names = label_names

        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=np.nan)

    # def _split_train_test(self):
    #     if self.training_set_size is None:
    #         self.training_set_size = len(self.x_data) - self.testing_set_size
    #     if self.testing_set_size is None:
    #         self.testing_set_size = len(self.x_data) - self.training_set_size
    #
    #     assert (self.training_set_size + self.testing_set_size) <= len(self.x_data)
    #
    #     self.x_train = self.x_data[: self.training_set_size, ]
    #     self.y_train = self.y_data[: self.training_set_size, ]
    #
    #     self.x_test = self.x_data[self.training_set_size: (self.training_set_size + self.testing_set_size), ]
    #     self.y_test = self.y_data[self.training_set_size: (self.training_set_size + self.testing_set_size), ]

    def print_parameters(self):
        print('Amount of training examples:', len(self.x_train))
        print('Amount of testing examples:', len(self.x_test))
        print('Kernel:')
        print(inspect.getsource(self.kernel))
        print('C =', self.C)
        print('Max Iterations:', self.max_iterations)
        print('Warmup Iterations:', self.warmup_iterations)

    def run(self):
        self._calculate_classifier()

        prediction_data = self._test()
        self._print_test_results(prediction_data)

    def _calculate_classifier(self):
        print('Training...')
        pr = cProfile.Profile()
        pr.enable()
        self.predictor = SVM_Predictor(self.x_train, self.y_train)
        self.predictor.train(kernel=self.kernel, C=self.C, max_iterations=self.max_iterations,
                             warmup_iterations=self.warmup_iterations)
        pr.disable()
        pr.print_stats(sort='tottime')

    def _label_to_index(self, label):
        return int(0.5 * label + 0.5)

    def _test(self):
        print('Testing...')

        prediction_data = defaultdict(lambda: defaultdict(lambda: 0))

        for i in range(len(self.x_test)):
            prediction = self.predictor.predict_label(self.x_test[i])
            truth = self.y_test[i]
            prediction_data[truth][prediction] += 1

        return prediction_data

    def _print_test_results(self, prediction_data):
        table_heading = ['Truth \ Prediction', self.label_names[1], self.label_names[-1], 'Total']
        table_data = [table_heading]

        for truth_label in [1, -1]:
            label_name = self.label_names[truth_label]
            pred_positive = prediction_data[truth_label][1]
            pred_negative = prediction_data[truth_label][-1]
            table_row = [label_name, pred_positive, pred_negative, pred_positive + pred_negative]
            table_data.append(table_row)

        last_row = ['Total']
        for i in [1, 2, 3]:
            last_row.append(table_data[1][i] + table_data[2][i])

        table_data.append(last_row)
        table = AsciiTable(table_data).table

        print(table)

        sensitivity = 100 * table_data[1][1] / table_data[1][3] if table_data[1][3] != 0 else '??'
        specificity = 100 * table_data[2][2] / table_data[2][3] if table_data[2][3] != 0 else '??'
        ppv = 100 * table_data[1][1] / table_data[3][1] if table_data[3][1] != 0 else '??'
        npv = 100 * table_data[2][2] / table_data[3][2] if table_data[3][2] != 0 else '??'

        print('Sensitivity (probability that "' + self.label_names[1] + '" value is identified correctly):',
              sensitivity, '%')
        print('Specificity (probability that "' + self.label_names[-1] + '" value is identified correctly):',
              specificity, '%')
        print('Positive predictive value (probability that a prediction of "' + self.label_names[1] + '" is correct):',
              ppv, '%')
        print('Negative predictive value (probability that a prediction of "' + self.label_names[-1] + '" is correct):',
              npv, '%')
        print()

        accuracy = 100 * (table_data[1][1] + table_data[2][2]) / table_data[3][3]
        mcc_numerator = table_data[1][1] * table_data[2][2] - table_data[1][2] * table_data[2][1]
        mcc_denominator = np.sqrt((table_data[1][1] + table_data[1][2]) * (table_data[1][1] + table_data[2][1]) * (
                table_data[2][2] + table_data[1][2]) * (table_data[2][2] + table_data[2][1]))
        mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0

        print('Accuracy:', accuracy, '%')
        print('Matthew correlation coefficient:', mcc)

    # def _test(self):
    #     print('Testing...')
    #
    #     # Format: ['Prediction', 'Total amount predicted', 'Amount predicted correctly', 'Amount predicted incorrectly']
    #     prediction_stats = [[i, 0, 0, 0] for i in range(2)]
    #
    #     # Format: ['True value', 'Total amount presented', 'Amount correctly identified', 'Amount misidentified']
    #     truth_stats = [[i, 0, 0, 0] for i in range(2)]
    #
    #     for i in range(len(self.x_test)):
    #         prediction_index = self._label_to_index(self.predictor.predict_label(self.x_test[i]))
    #         truth_index = self._label_to_index(self.y_test[i])
    #
    #         prediction_stats[prediction_index][1] += 1
    #         truth_stats[truth_index][1] += 1
    #
    #         if prediction_index == truth_index:
    #             prediction_stats[prediction_index][2] += 1
    #             truth_stats[truth_index][2] += 1
    #         else:
    #             prediction_stats[prediction_index][3] += 1
    #             truth_stats[truth_index][3] += 1
    #
    #     return prediction_stats, truth_stats

    # def _print_test_results(self, prediction_stats, truth_stats):
    #     for stats_list in (prediction_stats, truth_stats):
    #         summary = [sum(x) for x in zip(*stats_list)]
    #         summary[0] = 'All'
    #         stats_list.append(summary)
    #         self._add_percentages(stats_list)
    #
    #     prediction_table = [['Prediction', 'Total amount predicted', 'Amount predicted correctly',
    #                          'Amount predicted incorrectly']] + prediction_stats
    #     truth_table = [['True value', 'Total amount presented', 'Correctly identified', 'Misidentified']] + truth_stats
    #     print(AsciiTable(prediction_table).table)
    #     print(AsciiTable(truth_table).table)
    #
    # def _add_percentages(self, stats_list):
    #     for line in stats_list:
    #         for i in [2, 3]:
    #             perc = '??' if line[1] == 0 else str(100 * line[i] / line[1])
    #             line[i] = str(line[i]) + ' (' + perc + '%)'
