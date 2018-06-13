import numpy as np
import cvxopt

def calculate_hyperplane(inputs, labels, c):
    """Calculate a soft margin hyperplane for a binary classifier.

    Parameters
    ----------
    inputs : array, shape=(n,m)
             The matrix of training inputs. Each row is one training input.
    labels : array, shape=(n)
             The array of class labels. Each label must be either +1 or -1.
    c : float
             The value of the constant c in the dual problem for the soft margin hyperplane. Must be positive.
    Returns
    -------
    v : array, shape=(m)
        The normal vector of the hyperplane.
    s : float
        The shift of the hyperplane.
    """

    n = labels.size
    G = calculate_gram_matrix(inputs, labels, n)
    h = cvxopt.matrix(np.full(n, -1.0))
    P = calculate_inequality_matrix(n)
    q = calculate_inequality_vector(n, c)
    A = cvxopt.matrix(labels, tc='d').trans() # tc='d' causes conversion to double
    b = cvxopt.matrix(0.0)

    cvxopt_solution = cvxopt.solvers.qp(G, h, P, q, A, b)
    alpha = np.array(cvxopt_solution['x']).reshape( (n, ) )

    v = calculate_normal_vector(inputs, labels, alpha, n)
    s = calculate_shift(inputs, labels, c, alpha, v, n)

    return (v, s)

def calculate_gram_matrix(inputs, labels, n):
    G = np.empty( (n, n) )

    for i in range(0, n):
        for j in range(i, n):
            G[i, j] = G[j, i] = labels[i]*labels[j]*np.dot(inputs[i], inputs[j])

    return cvxopt.matrix(G)

def calculate_inequality_matrix(n):
    np_matrix = np.vstack( (-np.eye(n), np.eye(n)) )

    return cvxopt.matrix(np_matrix)

def calculate_inequality_vector(n, c):
    zero_vector = np.zeros(n)
    constant_vector = np.full(n, c)
    np_vector = np.concatenate( (zero_vector, constant_vector) )

    return cvxopt.matrix(np_vector)

def calculate_normal_vector(inputs, labels, alpha, n):
    m = inputs.shape[1]
    v = np.zeros( (m, ) )

    for i in range(0, n):
        v += alpha[i] * labels[i] * inputs[i]

    return v

def calculate_shift(inputs, labels, c, alpha, v, n):
    sum = 0.0
    summands = 0.0
    precision = np.finfo(np.float32).eps

    for i in range(n):
        if alpha[i] > precision and c - alpha[i] > precision:
            summands += 1
            sum += labels[i] - np.dot(inputs[i], v)

    return sum/summands
