import unittest
import numpy as np
from implementations_bachelor_thesis_vr.omh_predictor import Omh_Predictor

class TestStringMethods(unittest.TestCase):
    def test_calculate_nu_test1(self):
        x = np.array([[1, 2, 3, 4], [-7, 2, 4, 1]])
        y = np.array([1, -1])
        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)

        expected_nu = 2*np.dot(x[0], x[1]) - np.dot(x[0], x[0]) - np.dot(x[1], x[1])
        outcome_nu = predictor._calculate_nu(0, 1)
        self.assertAlmostEqual(outcome_nu, expected_nu)

    def test_calculate_nu_test2(self):
        x = np.array([[2.5, 4], [3.5, 0]])
        y = np.array([1, -1])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)

        expected_nu = -17
        outcome_nu = predictor._calculate_nu(0, 1)
        self.assertAlmostEqual(outcome_nu, expected_nu)

    def test_calculate_nu_test3(self):
        x = np.array([[2.5, 4. ], [2.5, 2.5]])
        y = np.array([1, 1])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)

        expected_nu = -2.25
        outcome_nu = predictor._calculate_nu(0, 1)
        self.assertAlmostEqual(outcome_nu, expected_nu, places=5)

    def test_calculate_e_test1(self):
        x = np.array([[1, 2, 3, 4], [-7, 2, 4, 1]])
        y = np.array([1, -1])
        v = np.array([-1, 5, 7, 3])
        s = 9

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s

        expected_e_0 = np.dot(x[0], v) + s - y[0]
        outcome_e_0 = predictor._calculate_e(0)
        self.assertAlmostEqual(outcome_e_0, expected_e_0)

    def test_calculate_e_test2(self):
        x = np.array([[2.5, 4], [3.5, 0]])
        y = np.array([1, -1])
        v = np.array([0.172, 1.276])
        s = -2.619

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s

        expected_e_0 = 1.915
        outcome_e_0 = predictor._calculate_e(0)
        self.assertAlmostEqual(outcome_e_0, expected_e_0)

        expected_e_1 = -1.017
        outcome_e_1 = predictor._calculate_e(1)
        self.assertAlmostEqual(outcome_e_1, expected_e_1)

    def test_calculate_e_test3(self):
        x = np.array([[2.5, 4. ], [2.5, 2.5]])
        y = np.array([1, 1])
        v = np.array([-0.83470769, 1.61156923])
        s = -0.9421538461538466

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s

        expected_e_0 = 2.41735
        outcome_e_0 = predictor._calculate_e(0)
        self.assertAlmostEqual(outcome_e_0, expected_e_0, places=5)

        expected_e_1 = 0
        outcome_e_1 = predictor._calculate_e(1)
        self.assertAlmostEqual(outcome_e_1, expected_e_1, places=5)

    def test_calculate_new_alpha_j_no_clipping(self):
        x = np.array([[1, 2, 3, 4], [-7, 2, 4, 1]])
        y = np.array([1, -1])
        v = np.array([-1, 5, 7, 3])
        s = 4
        alpha = np.array([7, 4])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s
        predictor.alpha = alpha

        e_0 = predictor._calculate_e(0)
        e_1 = predictor._calculate_e(1)
        nu = predictor._calculate_nu(0, 1)

        expected_alpha_1 = 4 - y[1] * (e_0 - e_1)/nu
        outcome_alpha_1 = predictor._calculate_new_alpha_j(0, 1)
        self.assertAlmostEqual(outcome_alpha_1, expected_alpha_1)

    def test_calculate_new_alpha_j_lower_clipping(self):
        x = np.array([[2.5, 4], [3.5, 0]])
        y = np.array([1, -1])
        v = np.array([0.172, 1.276])
        s = -2.619
        alpha = np.array([0, 0])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s
        predictor.alpha = alpha

        expected_alpha_1 = 0
        outcome_alpha_1 = predictor._calculate_new_alpha_j(0, 1)
        self.assertAlmostEqual(outcome_alpha_1, expected_alpha_1)

    def test_calculate_new_alpha_j_upper_clipping(self):
        x = np.array([[2.5, 4. ], [2.5, 2.5]])
        y = np.array([1, 1])
        v = np.array([-0.83470769, 1.61156923])
        s = -0.9421538461538466
        alpha = np.array([0.11815385, 0.62350769])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s
        predictor.alpha = alpha

        expected_alpha_1 = 0.74166154
        outcome_alpha_1 = predictor._calculate_new_alpha_j(0, 1)
        self.assertAlmostEqual(outcome_alpha_1, expected_alpha_1)

    def test_calculate_new_alpha_values_test1(self):
        x = np.array([[1, 2, 3, 4], [-7, 2, 4, 1]])
        y = np.array([1, -1])
        v = np.array([-1, 5, 7, 3])
        s = 4
        alpha = np.array([7, 4])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s
        predictor.alpha = alpha

        e_0 = predictor._calculate_e(0)
        e_1 = predictor._calculate_e(1)
        nu = predictor._calculate_nu(0, 1)

        expected_alpha_1 = 4 - (-1) * (e_0 - e_1)/nu
        expected_alpha_0 = 7 + 1*(-1)*(4 - expected_alpha_1)

        expected_alpha_values = np.array([expected_alpha_0, expected_alpha_1])
        outcome_alpha_values = predictor._calculate_new_alpha_values(0, 1)
        np.testing.assert_allclose(outcome_alpha_values, expected_alpha_values)

    def test_calculate_new_alpha_values_test2(self):
        x = np.array([[2.5, 4], [3.5, 0]])
        y = np.array([1, -1])
        v = np.array([0.172, 1.276])
        s = -2.619
        alpha = np.array([0, 0])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s
        predictor.alpha = alpha

        expected_alpha_values = np.array([0, 0])
        outcome_alpha_values = predictor._calculate_new_alpha_values(0, 1)
        np.testing.assert_allclose(outcome_alpha_values, expected_alpha_values)

    def test_calculate_new_alpha_values_test3(self):
        x = np.array([[2.5, 4. ], [2.5, 2.5]])
        y = np.array([1, 1])
        v = np.array([-0.83470769, 1.61156923])
        s = -0.9421538461538466
        alpha = np.array([0.11815385, 0.62350769])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s
        predictor.alpha = alpha

        expected_alpha_values = np.array([0, 0.74166154])
        outcome_alpha_values = predictor._calculate_new_alpha_values(0, 1)
        np.testing.assert_allclose(outcome_alpha_values, expected_alpha_values)

    def test_update_s_case1(self):
        x = np.array([[1, 2, 3, 4], [-7, 2, 4, 1]])
        y = np.array([1, -1])
        v = np.array([-1, 5, 7, 3])
        s = 5
        alpha = np.array([4, 7])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s
        predictor.alpha = alpha

        new_alpha_values = predictor._calculate_new_alpha_values(0, 1)
        e_0 = predictor._calculate_e(0)

        expected_s = s - e_0 - y[0]*(new_alpha_values[0] - alpha[0])*np.dot(x[0], x[0]) - y[1]*(new_alpha_values[1] - alpha[1])*np.dot(x[0], x[1])
        predictor._update_s(0, 1, new_alpha_values)

        self.assertAlmostEqual(predictor.s, expected_s)

    def test_update_s_case2(self):
        x = np.array([[2.5, 4. ], [2.5, 2.5]])
        y = np.array([1, 1])
        v = np.array([-0.83470769, 1.61156923])
        s = -0.9421538461538466
        alpha = np.array([0.11815385, 0.62350769])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s
        predictor.alpha = alpha

        new_alpha_values = predictor._calculate_new_alpha_values(0, 1)
        e_1 = predictor._calculate_e(1)

        expected_s = s - e_1 - y[0]*(new_alpha_values[0] - alpha[0])*np.dot(x[0], x[1]) - y[1]*(new_alpha_values[1] - alpha[1])*np.dot(x[1], x[1])
        predictor._update_s(0, 1, new_alpha_values)

        self.assertAlmostEqual(predictor.s, expected_s)

    def test_update_s_case3(self):
        x = np.array([[2.5, 4], [3.5, 0]])
        y = np.array([1, -1])
        v = np.array([0.17153846, 1.27615385])
        s = -2.6192307692307693
        alpha = np.array([0, 0])

        predictor = Omh_Predictor(x, y)
        predictor.tol = 10**(-3)
        predictor.v = v
        predictor.s = s
        predictor.alpha = alpha

        new_alpha_values = predictor._calculate_new_alpha_values(0, 1)
        e_0 = predictor._calculate_e(0)
        e_1 = predictor._calculate_e(1)

        s_0 = s - e_0 - y[0]*(new_alpha_values[0] - alpha[0])*np.dot(x[0], x[0]) - y[1]*(new_alpha_values[1] - alpha[1])*np.dot(x[0], x[1])
        s_1 = s - e_1 - y[0]*(new_alpha_values[0] - alpha[0])*np.dot(x[0], x[1]) - y[1]*(new_alpha_values[1] - alpha[1])*np.dot(x[1], x[1])

        # Introduced since KKT conditions are automatically fulfilled
        expected_s = s

        predictor._update_s(0, 1, new_alpha_values)

        self.assertAlmostEqual(predictor.s, expected_s)

if __name__ == '__main__':
    unittest.main()
