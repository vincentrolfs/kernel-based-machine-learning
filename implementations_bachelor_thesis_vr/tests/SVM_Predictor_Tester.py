import inspect
import time
from collections import defaultdict

import numpy as np
from terminaltables import AsciiTable

from implementations_bachelor_thesis_vr.SVM_Predictor import SVM_Predictor


class SVM_Predictor_Tester:
    def __init__(self):
        self.predictor = None

        np.set_printoptions(suppress=True)
        np.set_printoptions(threshold=np.nan)

    def calculate_predictor(self, x_train, y_train, kernel, C, max_iterations, warmup_iterations):
        print('>> Amount of training examples:', len(x_train))
        print('>> Kernel:')
        print(inspect.getsource(kernel))
        print('>> C =', C)
        print('>> Max Iterations:', max_iterations)
        print('>> Warmup Iterations:', warmup_iterations)
        print('>> Training...')

        start = time.time()
        self.predictor = SVM_Predictor(x_train, y_train)
        self.predictor.train(kernel, C, max_iterations, warmup_iterations)
        stop = time.time()

        print('>> Training completed. Duration:', stop - start, 'seconds.')

    def perform_test(self, x_test, y_test, label_names=None):
        if label_names is None:
            label_names = {1: '1', -1: '-1'}

        print('>> Amount of testing examples:', len(x_test))
        print('>> Testing...')

        prediction_data = defaultdict(lambda: defaultdict(lambda: 0))

        for i in range(len(x_test)):
            prediction = self.predictor.predict_label(x_test[i])
            truth = y_test[i]
            prediction_data[truth][prediction] += 1

        print('Hi', prediction_data)
        print('>> Test results:')
        self._print_test_results(prediction_data, label_names)

    @staticmethod
    def _print_test_results(prediction_data, label_names):
        table_heading = ['Truth \ Prediction', label_names[1], label_names[-1], 'Total']
        table_data = [table_heading]

        for truth_label in [1, -1]:
            label_name = label_names[truth_label]
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

        print('>> Sensitivity (probability that "' + label_names[1] + '" value is identified correctly):',
              sensitivity, '%')
        print('>> Specificity (probability that "' + label_names[-1] + '" value is identified correctly):',
              specificity, '%')
        print('>> Positive predictive value (probability that a prediction of "' + label_names[1] + '" is correct):',
              ppv, '%')
        print('>> Negative predictive value (probability that a prediction of "' + label_names[-1] + '" is correct):',
              npv, '%')
        print()

        accuracy = 100 * (table_data[1][1] + table_data[2][2]) / table_data[3][3]
        mcc_numerator = table_data[1][1] * table_data[2][2] - table_data[1][2] * table_data[2][1]
        mcc_denominator = np.sqrt((table_data[1][1] + table_data[1][2]) * (table_data[1][1] + table_data[2][1]) * (
                table_data[2][2] + table_data[1][2]) * (table_data[2][2] + table_data[2][1]))
        mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0

        print('>> Accuracy:', accuracy, '%')
        print('>> Matthews correlation coefficient:', mcc)
