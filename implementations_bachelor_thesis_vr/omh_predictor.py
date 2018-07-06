import numpy as np
import cvxopt


class Omh_Predictor:
    def __init__(self, x, y):
        """Initialize the optimal margin hyperplane predictor.

        Parameters
        ----------
        x : array, shape=(n,m)
            The matrix of training inputs. Each row is one training input.
        y : array, shape=(n,)
            The array of class labels. Each label must be either +1 or -1.
        """

        self.x = x
        self.y = y

        self.n, self.m = self.x.shape
        self._setup_trained_parameters()

        self.tol = 0

        assert len(self.y) == self.n, "Invalid arguments: Amount of training inputs and amount of labels is not the same"

    def _setup_trained_parameters(self):
        self.alpha = np.zeros(self.n)
        self.v = np.zeros(self.m)
        self.s = 0

    def predict_raw(self, z):
        """Returns the output of the "inner" decision function, i.e. the decision function without taking the sign.

        Parameters
        ----------
        z : array, shape=(m,)
            The input vector.

        Returns
        ----------
        Value of z applied to the inner decision function."""

        return np.dot(z, self.v) + self.s

    def predict_label(self, z):
        """Returns the output of the decision function.

        Parameters
        ----------
        z : array, shape=(m,)
            The input vector.

        Returns
        ----------
        Value of z applied to the decision function."""

        return np.sign(self.predict_raw(z))

    def train(self, tolerance = 10**(-3)):
        """Train the optimal margin hyperplane predictor.

        Parameters
        ----------
        tolerance : float
            The numerical tolerance."""

        self._setup_trained_parameters()
        self.tol = tolerance

        self._perform_training_iterations()

    def _perform_training_iterations(self):
        change_occured = False
        examine_all_i = True

        while change_occured or examine_all_i:
            change_occured = self._iterate_i(examine_all_i)
            examine_all_i = (not change_occured) and (not examine_all_i)

    def _iterate_i(self, examine_all_i):
        change_occured = False

        for i in self._range_with_random_start(self.n):
            if (examine_all_i or self.alpha[i] > self.tol) and not self._check_kkt_fulfilled(i):
                change_occured_here = self._optimize_i(i)
                if change_occured_here: change_occured = True

        return change_occured

    def _range_with_random_start(self, length):
        i = np.random.randint(0, length)
        for _ in range(length):
            yield i
            i = (i+1) % length

    def _optimize_i(self, i):
        if self._try_j_with_largest_expected_step(i):
            return True
        if self._try_all_j(i, bound_check_must_be=False):
            return True
        if self._try_all_j(i, bound_check_must_be=True):
            return True

        return False

    def _try_j_with_largest_expected_step(self, i):
        e_i = self._calculate_e(i)
        best_j = None
        largest_step = 0

        for j in range(self.n):
            eta = self._calculate_eta(i, j)
            if eta > -self.tol: continue

            e_j = self._calculate_e(j)
            step = np.abs((e_i - e_j) / eta)

            if step >= largest_step:
                best_j = j
                largest_step = step

        return self._try_optimize_pair(i, best_j)

    def _try_all_j(self, i, bound_check_must_be):
        for j in self._range_with_random_start(self.n):
            bound_check = (self.alpha[j] < self.tol)

            if bound_check == bound_check_must_be:
                eta = self._calculate_eta(i, j)
                if eta < -self.tol:
                    success = self._try_optimize_pair(i, j)
                    if success: return True

        return False

    def _try_optimize_pair(self, i, j):
        if j is None:
            return False
        else:
            is_progress_positive = self._optimize_pair(i, j)
            return is_progress_positive

    def _check_kkt_fulfilled(self, p):
        indicator = self.y[p] * self._calculate_e(p)
        if indicator < -self.tol: return False

        return self.alpha[p] < self.tol or indicator < self.tol

    def _optimize_pair(self, i, j):
        new_alpha_values, sign_new_alpha_j_unclipped = self._calculate_new_alpha_values(i, j)
        is_progress_positive = self._is_progress_positive(i, j, new_alpha_values)

        self._update_s(i, j, new_alpha_values, sign_new_alpha_j_unclipped)
        self._update_v(i, j, new_alpha_values)
        self._update_alpha(i, j, new_alpha_values)

        assert self._check_kkt_fulfilled(i)
        assert self._check_kkt_fulfilled(j)

        return is_progress_positive

    def _is_progress_positive(self, i, j, new_alpha_values):
        return np.abs(self.alpha[i] - new_alpha_values[0]) > self.tol or np.abs(self.alpha[j] - new_alpha_values[1]) > self.tol

    def _update_v(self, i, j, new_alpha_values):
        self.v = (
              self.v
            + self.y[i]*self.x[i]*(new_alpha_values[0] - self.alpha[i])
            + self.y[j]*self.x[j]*(new_alpha_values[1] - self.alpha[j]) # Kann noch vereinfacht werden
        )

    def _update_s(self, i, j, new_alpha_values, sign_new_alpha_j_unclipped):
        if new_alpha_values[0] > self.tol:
            self.s = self._calculate_new_s_p(new_alpha_values, i, j, i)
        elif new_alpha_values[1] > self.tol:
            self.s = self._calculate_new_s_p(new_alpha_values, i, j, j)
        elif self.y[i] != self.y[j]:
            self.s = self._calculate_new_s_p(new_alpha_values, i, j, i)
        elif sign_new_alpha_j_unclipped <= 0:
            self.s = self._calculate_new_s_p(new_alpha_values, i, j, i)
        else:
            self.s = self._calculate_new_s_p(new_alpha_values, i, j, j)

    def _calculate_new_s_p(self, new_alpha_values, i, j, p):
        e_p = self._calculate_e(p)
        return self.s - e_p + self.y[j]*(new_alpha_values[1] - self.alpha[j])*np.dot(self.x[p], self.x[i] - self.x[j])

    def _update_alpha(self, i, j, new_alpha_values):
        self.alpha[i] = new_alpha_values[0]
        self.alpha[j] = new_alpha_values[1]

    def _calculate_new_alpha_values(self, i, j):
        new_alpha_j, sign_new_alpha_j_unclipped = self._calculate_new_alpha_j(i, j)

        new_alpha_values = np.empty(2)
        new_alpha_values[0] = self.alpha[i] + self.y[i] * self.y[j] * (self.alpha[j] - new_alpha_j)
        new_alpha_values[1] = new_alpha_j

        return new_alpha_values, sign_new_alpha_j_unclipped

    def _calculate_new_alpha_j(self, i, j):
        eta = self._calculate_eta(i, j)

        e_i = self._calculate_e(i)
        e_j = self._calculate_e(j)

        new_alpha_j_unclipped = self.alpha[j] - self.y[j]*(e_i - e_j)/eta
        new_alpha_j = self._clip_alpha_j(i, j, new_alpha_j_unclipped)

        return (new_alpha_j, np.sign(new_alpha_j_unclipped))

    def _clip_alpha_j(self, i, j, alpha_j_unclipped):
        if self.y[i] != self.y[j]:
            return max(0, self.alpha[j] - self.alpha[i], alpha_j_unclipped)
        else:
            if alpha_j_unclipped <= 0: return 0
            return min(self.alpha[j] + self.alpha[i], alpha_j_unclipped)

    def _calculate_eta(self, p, q):
        d = self.x[p] - self.x[q]
        return -np.dot(d, d)

    def _calculate_e(self, p):
        return self.predict_raw(self.x[p]) - self.y[p]
