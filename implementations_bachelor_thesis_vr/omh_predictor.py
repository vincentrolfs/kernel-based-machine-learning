import numpy as np
import cvxopt


class Omh_Predictor:
    def __init__(self, x, y):
        """Initialise the optimal margin hyperplane predictor.

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

    def train(self, max_unchanging_iterations, tolerance = 10**(-3)):
        """Train the optimal margin hyperplane predictor.

        Parameters
        ----------
        max_unchanging_iterations : int
            Specifies the amount of consecutive iterations in which alpha didn't change after which the training algorithm terminates
        tolerance : float
            The numerical tolerance."""

        self._setup_trained_parameters()
        self.tol = tolerance
        unchanging_iterations = 0
        i = j = 0

        while unchanging_iterations < max_unchanging_iterations:
            i, j = self._select_indices(i)

            if j is None:
                has_alpha_changed = False
            else:
                has_alpha_changed = self._optimize_pair(i, j)

            if has_alpha_changed:
                unchanging_iterations = 0
            else:
                unchanging_iterations += 1

    def _select_indices(self, old_i):
        i = (old_i + 1) % self.n
        check_i = self.y[i]*self._calculate_e(i)

        if check_i > -self.tol and (self.alpha[i] < self.tol or check_i < self.tol):
            return (i, None)

        j = self._randint(upper_bound=self.n, avoid=i)
        return (i, j)

    def _randint(self, upper_bound, avoid):
        return (avoid + np.random.randint(1, upper_bound)) % upper_bound

    def _optimize_pair(self, i, j):
        new_alpha_values = self._calculate_new_alpha_values(i, j)
        has_alpha_changed = (new_alpha_values is not None)

        if has_alpha_changed:
            self._update_s(i, j, new_alpha_values)
            self._update_v(i, j, new_alpha_values)
            self._update_alpha(i, j, new_alpha_values)

        return has_alpha_changed

    def _update_v(self, i, j, new_alpha_values):
        self.v = (
              self.v
            + self.y[i]*self.x[i]*(new_alpha_values[0] - self.alpha[i])
            + self.y[j]*self.x[j]*(new_alpha_values[1] - self.alpha[j]) # Kann noch vereinfacht werden
        )

    def _update_s(self, i, j, new_alpha_values):
        if new_alpha_values[0] > self.tol:
            e_i = self._calculate_e(i)
            self.s = self.s - e_i + self.y[j]*(new_alpha_values[1] - self.alpha[j])*np.dot(self.x[i], self.x[i] - self.x[j])
        elif new_alpha_values[1] > self.tol:
            e_j = self._calculate_e(j)
            self.s = self.s - e_j + self.y[j]*(new_alpha_values[1] - self.alpha[j])*np.dot(self.x[j], self.x[i] - self.x[j])

    def _update_alpha(self, i, j, new_alpha_values):
        self.alpha[i] = new_alpha_values[0]
        self.alpha[j] = new_alpha_values[1]

    def _calculate_new_alpha_values(self, i, j):
        new_alpha_j = self._calculate_new_alpha_j(i, j)
        if new_alpha_j is None: return None

        new_alpha_values = np.empty(2)
        new_alpha_values[0] = self.alpha[i] + self.y[i] * self.y[j] * (self.alpha[j] - new_alpha_j)
        new_alpha_values[1] = new_alpha_j

        return new_alpha_values

    def _calculate_new_alpha_j(self, i, j):
        nu = self._calculate_nu(i, j)
        if nu >= -self.tol: return None

        e_i = self._calculate_e(i)
        e_j = self._calculate_e(j)

        new_alpha_j_unclipped = self.alpha[j] - self.y[j]*(e_i - e_j)/nu
        new_alpha_j = self._clip_alpha_j(i, j, new_alpha_j_unclipped)

        delta = np.abs(new_alpha_j - self.alpha[j])

        if delta <= self.tol: return None
        else: return new_alpha_j

    def _clip_alpha_j(self, i, j, alpha_j_unclipped):
        if self.y[i] != self.y[j]:
            return max(0, self.alpha[j] - self.alpha[i], alpha_j_unclipped)
        else:
            if alpha_j_unclipped <= 0: return 0
            return min(self.alpha[j] + self.alpha[i], alpha_j_unclipped)

    def _calculate_nu(self, i, j):
        d = self.x[i] - self.x[j]
        return -np.dot(d, d)

    def _calculate_e(self, i):
        return self.predict_raw(self.x[i]) - self.y[i]
