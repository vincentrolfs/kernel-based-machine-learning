import numpy as np
import scipy.linalg as la


class Regressor:
    def __init__(self, x, y):
        """Initialize the regressor.

        Parameters
        ----------
        x : array, shape=(n,m)
            The matrix of training inputs. Each row is one training input.
        y : array, shape=(n,)
            The array of training outputs.
        """

        self.x = x
        self.y = y

        self.n, self.m = x.shape

        self.basis_size = None
        self.kernel = None
        self.mu = None
        self.coefficients = None
        self.x_basis = None
        self.max_iterations_basis_search = None

        assert len(self.y) == self.n, "Invalid arguments: Amount of training inputs and amount of training outputs " \
                                      "is not the same"
        assert self.n >= 1, "Invalid arguments: Amount of training inputs must be at least 1"

    def predict(self, z):
        """Predicts an output value.

        Parameters
        ----------
        z : array, shape=(m,)
            The input vector."""

        if self.m != 1:
            assert z.shape == (self.m,), "Could not make prediction because of wrong shape. Expected: " + str(
                (self.m,)) + ". Actual: " + str(z.shape)

        kernel_values = np.apply_along_axis(self.kernel, 1, self.x_basis, z)
        return np.dot(self.coefficients, kernel_values)

    def train(self, kernel, mu, basis_size, max_iterations_basis_search=100):
        """
        Train the regressor.
        kernel: function
            The kernel function. Must take two arguments which are arrays of shape (m,) and return a floating point
            value. The kernel function used must be strictly positive definite.
        mu : float
            The regularization constant. Must be non-negative.
        basis_size : int
            The amount of inputs that are used for the basis of the regressor. Must be between 1 and n (inclusive).
        max_iterations_basis_search : int
            The maximum amount of iterations that may be used in the k-medoids algorithm that determines the best basis nodes.
        """

        assert 1 <= basis_size <= self.n, "Basis size must be between 1 and n (inclusive)"

        self.kernel = kernel
        self.mu = mu
        self.basis_size = basis_size
        self.max_iterations_basis_search = max_iterations_basis_search

        kernel_matrix = self._calculate_kernel_matrix()
        basis_indices = self._determine_basis_indices(kernel_matrix)

        M_tilde = self._calculate_M_tilde(kernel_matrix, basis_indices)
        y_tilde = self.calculate_y_tilde(basis_indices)
        self.x_basis = self.x[basis_indices]

        del kernel_matrix
        del basis_indices

        Q, R = self._calculate_QR(M_tilde)
        del M_tilde

        S = self._calculate_S(R)
        del R

        w = self._calculate_w(Q, y_tilde)
        del Q

        self._calculate_coefficients(S, w)

    def _calculate_kernel_matrix(self):
        print("Calculating kernel matrix...")
        A = np.empty((self.n, self.n))

        for i in range(self.n):
            for j in range(i, self.n):
                A[i, j] = A[j, i] = self.kernel(self.x[i], self.x[j])

        return A

    def _determine_basis_indices(self, kernel_matrix):
        print("Choosing basis...")
        distance_matrix = self._calculate_distance_matrix(kernel_matrix)
        medoid_indices = np.sort(np.random.choice(self.n, self.basis_size, replace=False))
        cluster_indices = [None for _ in range(self.basis_size)]
        medoid_indices_new = np.empty(self.basis_size, dtype=int)

        for _ in range(self.max_iterations_basis_search):
            cluster_assignment = np.argmin(distance_matrix[:, medoid_indices], axis=1)
            for i in range(self.basis_size):
                cluster_indices[i] = np.where(cluster_assignment == i)[0]

            for i in range(self.basis_size):
                indices_in_cluster = cluster_indices[i]
                distances_in_cluster = distance_matrix[np.ix_(indices_in_cluster, indices_in_cluster)]
                mean_distances = np.mean(distances_in_cluster, axis=1)
                best_mean_distance_index = np.argmin(mean_distances)

                medoid_indices_new[i] = cluster_indices[i][best_mean_distance_index]

            medoid_indices_new.sort()

            if np.array_equal(medoid_indices, medoid_indices_new):
                print('Clustering ended with break.')
                break

            medoid_indices = medoid_indices_new.copy()
        return medoid_indices

    def _calculate_distance_matrix(self, kernel_matrix):
        diag = np.diag(kernel_matrix)

        return -2 * kernel_matrix + diag + diag[:, np.newaxis]

    def _calculate_M_tilde(self, kernel_matrix, basis_indices):
        print("Calculating M_tilde...")
        non_basis_indices_mask = self._get_non_basis_indices_mask(basis_indices)

        M_tilde = np.empty((self.n + self.basis_size, self.basis_size))

        B = M_tilde[:self.basis_size, :] = kernel_matrix[:, basis_indices][basis_indices, :].copy()
        M_tilde[self.basis_size:self.n, :] = kernel_matrix[:, basis_indices][non_basis_indices_mask, :].copy()

        M_tilde[self.n:, :] = np.sqrt(self.mu) * la.sqrtm(B)

        return M_tilde

    def calculate_y_tilde(self, basis_indices):
        print("Calculating y_tilde...")
        non_basis_indices_mask = self._get_non_basis_indices_mask(basis_indices)

        y_tilde = np.zeros(self.n + self.basis_size)
        y_tilde[:self.basis_size] = self.y[basis_indices]
        y_tilde[self.basis_size:self.n] = self.y[non_basis_indices_mask]

        return y_tilde

    def _get_non_basis_indices_mask(self, basis_indices):
        non_basis_indices = np.ones(self.n, dtype=np.bool)
        non_basis_indices[basis_indices] = False

        return non_basis_indices

    def _calculate_QR(self, data_matrix):
        print("Calculating QR...")
        return np.linalg.qr(data_matrix, mode="complete")

    def _calculate_S(self, R):
        print("Calculating S...")
        return R[0:self.basis_size, 0:self.basis_size]

    def _calculate_w(self, Q, y_tilde):
        print("Calculating w...")
        return np.dot(np.transpose(Q), y_tilde)[0:self.basis_size]

    def _calculate_coefficients(self, S, w):
        print("Calculating coefficients...")
        self.coefficients = la.solve_triangular(S, w)
