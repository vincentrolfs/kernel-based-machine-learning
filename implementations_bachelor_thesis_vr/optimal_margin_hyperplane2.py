import numpy as np
import cvxopt

class Omh_Predictor:
    def __init__(self, x, y, max_unchanging_iterations):
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
        self.__reset_trained_parameters()

        assert len(self.y) == self.n, "Amount of training inputs and amount of labels is not the same"

    def __reset_trained_parameters(self):
        self.n, self.m = self.x.shape

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

        return np.dot(z, v) + s

    def predict_label(self, z):
        """Returns the output of the decision function.

        Parameters
        ----------
        z : array, shape=(m,)
            The input vector.

        Returns
        ----------
        Value of z applied to the decision function."""

        return np.sign(self.predict_margin(z))

    def train(self, max_unchanging_iterations, tolerance = 10**(-3)):
        """Train the optimal margin hyperplane predictor.

        Parameters
        ----------
        max_unchanging_iterations : int
            Specifies the amount of consecutive iterations in which alpha didn't change after which the training algorithm terminates
        tolerance : float
            The numerical tolerance."""

        self.__reset_trained_parameters()
        self.tol = tolerance
        unchanging_iterations = 0
        i = j = 0

        while unchanging_iterations < max_unchanging_iterations:
            i, j = self.__select_indices(i)
            has_alpha_changed = self.__optimize_pair(i, j)

            if has_alpha_changed:
                unchanging_iterations = 0
            else:
                unchanging_iterations += 1

    def __select_indices(self, old_i):
        i = old_i + 1 % self.n
        j = np.random.randint(0, self.n)

        return (i, j)

    def __optimize_pair(self, i, j):
        new_alpha_values = self.__calculate_new_alpha_values(i, j)
        has_alpha_changed = (new_alpha_values is not None)

        if has_alpha_changed:
            self.__update_v(i, j, new_alpha_values)
            self.__update_s(i, j, new_alpha_values)
            self.__update_alpha(i, j, new_alpha_values)

        return has_alpha_changed

    def __calculate_new_alpha_values(self, i, j):
        new_alpha_j = self.__calculate_new_alpha_j(i, j)
        if new_alpha_j is None: return None

        new_alpha_values = np.empty(2)
        new_alpha_values[0] = self.alpha[i] + self.y[i]*self.y[j](self.alpha[j] - new_alpha_j)
        new_alpha_values[1] = new_alpha_j

        return new_alpha_values

    def self.__update_v(self, i, j, new_alpha_values):
        self.v = (
              self.v
            + self.y[i]*self.x[i]*(new_alpha_values[0] - self.alpha[i])
            + self.y[j]*self.x[j]*(new_alpha_values[1] - self.alpha[j])
        )

    def self.__update_s(self, i, j, new_alpha_values):
        if new_alpha_values[0] > self.tol:
            e_i = self.__calculate_raw_prediction_error(i)
            ...

    def __calculate_new_alpha_j(self, i, j):
        nu = self.__calculate_nu(i, j)
        if nu >= -self.tol: return None

        e_i = self.__calculate_raw_prediction_error(i)
        e_j = self.__calculate_raw_prediction_error(j)

        new_alpha_j_unclipped = self.alpha[j] - self.y[j]*(e_i - e_j)/nu

        return self.__clip_alpha_j(i, j, new_alpha_j_unclipped)

    def __clip_alpha_j(self, i, j, alpha_j_unclipped):
        if self.y[i] != self.y[j]:
            return max(0, self.alpha[j] - self.alpha[i], alpha_j_unclipped)
        else:
            return min(self.alpha[j] + self.alpha[i], alpha_j_unclipped)

    def __calculate_nu(self, i, j):
        d = self.x[i] - self.x[j]
        return -np.dot(d, d)

    def __calculate_raw_prediction_error(self, j):
        return self.predict_raw(self.x[j]) - self.y[j]
