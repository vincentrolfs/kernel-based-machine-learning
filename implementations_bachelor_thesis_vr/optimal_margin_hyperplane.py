import numpy as np
import cvxopt

def calculate_hyperplane(inputs, labels):
    """Calculate an optimal margin hyperplane for a binary classifier.

    Parameters
    ----------
    inputs : array, shape=(n,m)
             The matrix of training inputs. Each row is one training input.
    labels : array, shape=(n,)
             The array of class labels. Each label must be either +1 or -1.
    Returns
    -------
    v : array, shape=(m,)
        The normal vector of the hyperplane.
    s : float
        The shift of the hyperplane.
    """

    n = labels.size
    G = calculate_gram_matrix(inputs, labels, n)
    h = cvxopt.matrix(np.full(n, -1.0))
    P = cvxopt.matrix(-np.eye(n))
    q = cvxopt.matrix(np.zeros(n))
    A = cvxopt.matrix(labels, tc='d').trans() # tc='d' causes conversion to double
    b = cvxopt.matrix(0.0)

    cvxopt_solution = cvxopt.solvers.qp(G, h, P, q, A, b)
    alpha = np.array(cvxopt_solution['x']).reshape( (n, ) )
    v = calculate_normal_vector(inputs, labels, alpha, n)
    s = calculate_shift(inputs, labels, alpha, v, n)

    return (v, s)

def calculate_gram_matrix(inputs, labels, n):
    G = np.empty( (n, n) )

    for i in range(0, n):
        for j in range(i, n):
            G[i, j] = G[j, i] = labels[i]*labels[j]*np.dot(inputs[i], inputs[j])

    return cvxopt.matrix(G)

def calculate_normal_vector(inputs, labels, alpha, n):
    m = inputs.shape[1]
    v = np.zeros( (m, ) )

    for i in range(0, n):
        v += alpha[i] * labels[i] * inputs[i]

    return v

def calculate_shift(inputs, labels, alpha, v, n):
    sum = 0.0
    summands = 0.0

    for i in range(n):
        if alpha[i] <= np.finfo(np.float32).eps: continue

        summands += 1
        sum += labels[i] - np.dot(inputs[i], v)

    return sum/summands
