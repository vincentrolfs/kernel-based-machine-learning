import numpy as np


class SVM_Predictor:
    def __init__(self, x, y):
        """Initialize the support vector machine predictor.

        Parameters
        ----------
        x : array, shape=(n,m)
            The matrix of training inputs. Each row is one training input.
        y : array, shape=(n,)
            The array of class labels. Each label must be either +1 or -1. At least one example from each class muste be present.
        """

        self.x = x
        self.y = y

        self.n, self.m = self.x.shape

        self.alpha = None
        self.s = None
        self.e_cache = None
        self.kernel = None
        self.kernel_cache = None
        self.C = None
        self.tol = None
        self.max_iterations = None
        self.amount_of_bound_alpha_values = None

        assert len(self.y) == self.n, "Invalid arguments: Amount of training inputs and amount of labels" \
                                      "is not the same"
        assert set(self.y) == {1, -1}, "Invalid arguments: Each label must be either +1 or -1. " \
                                       "At least one example from each class muste be present."

    def _setup_trained_parameters(self):
        self.alpha = np.zeros(self.n)
        self.s = 0
        self.e_cache = np.zeros(self.n)
        self.kernel_cache = {}
        self.amount_of_bound_alpha_values = len(self.alpha)

    def predict_raw(self, z):
        """Returns the output of the "inner" decision function, i.e. the decision function without taking the sign.

        Parameters
        ----------
        z : array, shape=(m,)
            The input vector.

        Returns
        ----------
        Value of z applied to the inner decision function."""

        result = self.s

        for p in range(self.n):
            if self.alpha[p] > self.tol:
                result += self.alpha[p] * self.y[p] * self.kernel(z, self.x[p])

        return result

    def predict_label(self, z):
        """Returns the output of the decision function.

        Parameters
        ----------
        z : array, shape=(m,)
            The input vector.

        Returns
        ----------
        Value of z applied to the decision function."""

        inner_value = self.predict_raw(z)

        if inner_value >= 0:
            return 1.0
        else:
            return -1.0

    def print_diagnostics(self):
        """
        Print diagnostics which can be used to determine if the algorithm works correctly.
        Requires the package 'terminaltables'.
        """

        from terminaltables import AsciiTable

        sum_0 = 0
        table_data = [["p", "alpha_p", "y_p((sum alpha_p y_p k(x_p, v)) + s)", "0 <= alpha_p <= C (within tolerance)?",
                       "KKT-conditions fulfilled (within tolerance)?"]]

        for p in range(self.n):
            sum_0 += self.alpha[p] * self.y[p]
            table_data.append(
                [p, self.alpha[p], self.y[p] * self.predict_raw(self.x[p]),
                 -self.tol <= self.alpha[p] <= self.C + self.tol,
                 self._check_kkt_fulfilled(p)])

        print("Parameters:")
        print("  C =", self.C)
        print("  tolerance =", self.tol)
        print("Computed values:")
        print("  s =", self.s)
        print("  alpha =", self.alpha)
        print("  amount of bound alpha values =", self.amount_of_bound_alpha_values)
        print("Check: This should be zero (within tolerance):")
        print("  sum_p of alpha_p y_p =", sum_0)
        print("  Check", "passed!" if np.abs(sum_0) < self.tol else "failed!")

        table = AsciiTable(table_data)
        print(table.table)

    def train(self, kernel, C, tolerance=10 ** (-3), max_iterations=100):
        """Train the optimal margin hyperplane predictor.

        Parameters
        ---------
        kernel: function
            The kernel function. Must take two arguments which are NumPy arrays of shape (m,) and return a floating point value
        C : float
            The constant bounding the alpha values from above. Must be positive.
        tolerance : float
            The numerical tolerance.
        max_iterations : int or infinity
            The total number of iterations after which the algorithm stops if an optimal solution has not been found up
            to this point. On iteration optimizes each index i at least once."""

        assert C > 0, "Invalid arguments: C must be positive"

        self._setup_trained_parameters()
        self.kernel = kernel
        self.C = C
        self.tol = tolerance
        self.max_iterations = max_iterations

        self._perform_training_iterations()

    def _perform_training_iterations(self):
        made_positive_progress = False
        examine_all_i = True

        iteration_count = 0
        while made_positive_progress or examine_all_i:
            print("Iteration #" + str(iteration_count + 1))

            made_positive_progress = self._iterate_i(examine_all_i)
            examine_all_i = (not made_positive_progress) and (not examine_all_i)

            iteration_count += 1
            if iteration_count == self.max_iterations: break

    def _iterate_i(self, examine_all_i):
        made_positive_progress = False

        for i in self._range_with_random_start(self.n):
            if (examine_all_i or not self._is_at_bounds(self.alpha[i])) and not self._check_kkt_fulfilled(i):
                is_progress_positive = self._optimize_i(i)

                if is_progress_positive:
                    made_positive_progress = True

        return made_positive_progress

    def _is_at_bounds(self, alpha_value):
        return alpha_value < self.tol or self.C - alpha_value < self.tol

    def _check_kkt_fulfilled(self, p):
        indicator = self.y[p] * self._calculate_e(p)

        if self.alpha[p] < self.tol:
            return indicator > - self.tol
        elif self.C - self.alpha[p] < self.tol:
            return indicator < self.tol
        else:
            return np.abs(indicator) < self.tol

    def _optimize_i(self, i):
        if self._try_cached_j_with_largest_expected_step(i):
            return True
        if self._try_all_j(i, bound_check_must_be=False):
            return True
        if self._try_all_j(i, bound_check_must_be=True):
            return True

        return False

    def _try_cached_j_with_largest_expected_step(self, i):
        e_i = self._calculate_e(i)
        best_j = None
        largest_expected_step = 0

        for j in range(self.n):
            if self._is_at_bounds(self.alpha[j]): continue

            eta = self._calculate_eta(i, j)
            if eta > -self.tol: continue

            e_j = self._calculate_e(j)
            expected_step = np.abs((e_i - e_j) / eta)

            if expected_step >= largest_expected_step:
                best_j = j
                largest_expected_step = expected_step

        return self._try_optimize_pair(i, best_j)

    def _try_all_j(self, i, bound_check_must_be):
        for j in self._range_with_random_start(self.n):
            bound_check = self._is_at_bounds(self.alpha[j])

            if bound_check == bound_check_must_be:
                eta = self._calculate_eta(i, j)
                if eta < -self.tol:
                    is_progress_positive = self._try_optimize_pair(i, j)
                    if is_progress_positive: return True

        return False

    def _try_optimize_pair(self, i, j):
        if j is None:
            return False
        else:
            is_progress_positive = self._optimize_pair(i, j)
            return is_progress_positive

    def _optimize_pair(self, i, j):
        new_alpha_values = self._calculate_new_alpha_values(i, j)
        is_progress_positive = self._is_progress_positive(i, new_alpha_values)

        old_s = self.s
        old_alpha_j = self.alpha[j]
        self._update_s(i, j, new_alpha_values)
        self._update_alpha(i, j, new_alpha_values)
        self._update_e_cache(i, j, old_s, old_alpha_j)

        return is_progress_positive

    def _is_progress_positive(self, i, new_alpha_values):
        return np.abs(self.alpha[i] - new_alpha_values[0]) > self.tol ** 2

    def _update_s(self, i, j, new_alpha_values):
        if not self._is_at_bounds(new_alpha_values[0]):
            self.s = self._calculate_new_s_p(new_alpha_values, i, j, i)
        elif not self._is_at_bounds(new_alpha_values[1]):
            self.s = self._calculate_new_s_p(new_alpha_values, i, j, j)
        else:
            self._update_s_boundary_case(i, j, new_alpha_values)

    def _update_s_boundary_case(self, i, j, new_alpha_values):
        delta_i = self._calculate_delta_p(new_alpha_values[0], i)
        delta_j = self._calculate_delta_p(new_alpha_values[1], j)
        new_s_i = self._calculate_new_s_p(new_alpha_values, i, j, i)

        if delta_i != delta_j:
            self.s = new_s_i
        else:
            new_s_j = self._calculate_new_s_p(new_alpha_values, i, j, j)

            if delta_i == 1:
                self.s = max(new_s_i, new_s_j)
            else:
                self.s = min(new_s_i, new_s_j)

    def _calculate_new_s_p(self, new_alpha_values, i, j, p):
        e_p = self._calculate_e(p)
        return self.s - e_p + self.y[j] * (new_alpha_values[1] - self.alpha[j]) * (
                self._kernel_optimized(p, i) - self._kernel_optimized(p, j))

    def _calculate_delta_p(self, new_alpha_value, p):
        lambda_p = 1 if new_alpha_value < self.tol else -1
        return self.y[p] * lambda_p

    def _update_alpha(self, i, j, new_alpha_values):
        self.amount_of_bound_alpha_values += int(self._is_at_bounds(new_alpha_values[0])) + int(
            self._is_at_bounds(new_alpha_values[1])) - int(self._is_at_bounds(self.alpha[i])) - int(
            self._is_at_bounds(self.alpha[j]))

        self.alpha[i] = new_alpha_values[0]
        self.alpha[j] = new_alpha_values[1]

    def _calculate_new_alpha_values(self, i, j):
        new_alpha_j = self._calculate_new_alpha_j(i, j)

        new_alpha_values = np.empty(2)
        new_alpha_values[0] = self.alpha[i] + self.y[i] * self.y[j] * (self.alpha[j] - new_alpha_j)
        new_alpha_values[1] = new_alpha_j

        return new_alpha_values

    def _calculate_new_alpha_j(self, i, j):
        eta = self._calculate_eta(i, j)

        e_i = self._calculate_e(i)
        e_j = self._calculate_e(j)

        new_alpha_j_unclipped = self.alpha[j] - self.y[j] * (e_i - e_j) / eta
        new_alpha_j = self._clip_alpha_j(i, j, new_alpha_j_unclipped)

        return new_alpha_j

    def _clip_alpha_j(self, i, j, alpha_j_unclipped):
        l, r = self._calculate_l_r(i, j)

        if alpha_j_unclipped < l:
            return l
        elif alpha_j_unclipped > r:
            return r
        else:
            return alpha_j_unclipped

    def _calculate_l_r(self, i, j):
        if self.y[i] == self.y[j]:
            l = max(0, self.alpha[i] + self.alpha[j] - self.C)
            r = min(self.C, self.alpha[i] + self.alpha[j])
        else:
            l = max(0, self.alpha[j] - self.alpha[i])
            r = min(self.C, self.alpha[j] - self.alpha[i] + self.C)

        return l, r

    def _update_e_cache(self, i, j, old_s, old_alpha_j):
        for p in range(self.n):
            if self._is_at_bounds(self.alpha[p]):
                continue
            elif p == i or p == j:
                self.e_cache[p] = 0
            else:
                self.e_cache[p] += self.y[j] * (self.alpha[j] - old_alpha_j) * (
                        self._kernel_optimized(p, j) - self._kernel_optimized(p, i)
                ) + self.s - old_s

    def _calculate_e(self, p):
        if self._is_at_bounds(self.alpha[p]):
            return self._predict_raw_optimized(p) - self.y[p]
        else:
            return self.e_cache[p]

    def _predict_raw_optimized(self, p):
        result = self.s

        for q in range(self.n):
            if self.alpha[q] > self.tol:
                result += self.alpha[q] * self.y[q] * self._kernel_optimized(p, q)

        return result

    def _calculate_eta(self, p, q):
        return 2 * self._kernel_optimized(p, q) - self._kernel_optimized(p, p) - self._kernel_optimized(q, q)

    def _kernel_optimized(self, p, q):
        if p >= q:
            key = (p, q)
        else:
            key = (q, p)

        result = self.kernel_cache.get(key)

        if result is None:
            result = self.kernel_cache[key] = self.kernel(self.x[p], self.x[q])

        return result

    @staticmethod
    def _range_with_random_start(length):
        i = np.random.randint(0, length)
        for _ in range(length):
            yield i
            i = (i + 1) % length
