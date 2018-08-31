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
        self.C = None
        self.coefficients = None
        self.x_basis = None

        assert len(self.y) == self.n, "Invalid arguments: Amount of training inputs and amount of training outputs" \
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

    def train(self, kernel, mu, basis_size):
        """
        Train the regressor.
        kernel: function
            The kernel function. Must take two arguments which are arrays of shape (m,) and return a floating point value
        mu : float
            The regularization constant. Must be non-negative.
        basis_size : int
            The amount of inputs that are used for the basis of the regressor. Must be between 1 and n (inclusive).
        """

        assert 1 <= basis_size <= self.n, "Basis size must be between 1 and n (inclusive)"

        self.kernel = kernel
        self.C = mu
        self.basis_size = basis_size

        kernel_matrix = self._calculate_kernel_matrix()

        basis_indices = self._determine_basis_indices(kernel_matrix)
        data_matrix = self._calculate_data_matrix(kernel_matrix, basis_indices)
        self.x_basis = self.x[basis_indices]

        del kernel_matrix
        del basis_indices

        Q, R = self.calculate_QR(data_matrix)
        S = self.calculate_S(R)
        g = self.calculate_g(Q)

        self.calculate_coefficients(S, g)

    def _calculate_kernel_matrix(self):
        print("Calculating kernel matrix...")
        A = np.empty((self.n, self.n))

        for i in range(self.n):
            for j in range(i, self.n):
                A[i, j] = A[j, i] = self.kernel(self.x[i], self.x[j])

        return A

    def _determine_basis_indices(self, A):
        print("Choosing basis...")
        D = self._calculate_distance_matrix(A)
        medoid_indices = np.sort(np.random.choice(self.n, self.basis_size, replace=False))
        cluster_indices = [None for _ in range(self.basis_size)]
        medoid_indices_new = np.empty(self.basis_size, dtype=int)

        for _ in range(100):
            cluster_assignment = np.argmin(D[:, medoid_indices], axis=1)
            for i in range(self.basis_size):
                cluster_indices[i] = np.where(cluster_assignment == i)[0]

            for i in range(self.basis_size):
                indices_in_cluster = cluster_indices[i]
                distances_in_cluster = D[np.ix_(indices_in_cluster, indices_in_cluster)]
                mean_distances = np.mean(distances_in_cluster, axis=1)
                best_mean_distance_index = np.argmin(mean_distances)

                medoid_indices_new[i] = cluster_indices[i][best_mean_distance_index]

            np.sort(medoid_indices_new)

            if np.array_equal(medoid_indices, medoid_indices_new):
                print('Clustering ended with break.')
                break

            medoid_indices = medoid_indices_new.copy()
        return medoid_indices

    def _calculate_distance_matrix(self, A):
        diag = np.diag(A)

        return -2 * A + diag + diag[:, np.newaxis]

    def _calculate_data_matrix(self, kernel_matrix, basis_indices):
        print("Calculating data matrix...")
        B_c = np.empty((self.n + self.basis_size, self.basis_size))
        B_c[:self.basis_size, :] = kernel_matrix[:, basis_indices][basis_indices, :].copy()
        mask = np.ones(self.n, dtype=np.bool)
        mask[basis_indices] = False
        B_c[self.basis_size: self.n] = kernel_matrix[:, basis_indices][mask, :].copy()

        B_c[self.n:, :] = np.sqrt(self.C) * la.sqrtm(B_c[:self.basis_size, :])

        return B_c

    def calculate_QR(self, data_matrix):
        print("Calculating QR...")
        return np.linalg.qr(data_matrix, mode="complete")

    def calculate_S(self, R):
        print("Calculating S...")
        return R[0:self.basis_size, 0:self.basis_size]

    def calculate_g(self, Q):
        print("Calculating g...")
        y_padded = np.pad(self.y, (0, self.basis_size), "constant", constant_values=0)
        return np.dot(np.transpose(Q), y_padded)[0:self.basis_size]

    def calculate_coefficients(self, S, g):
        print("Calculating alpha...")
        self.coefficients = la.solve_triangular(S, g)
