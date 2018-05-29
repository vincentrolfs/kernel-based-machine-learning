import numpy as np
import cvxopt

def make_classifier(inputs, labels, c, kernel):
    alpha, s = calculate_hyperplane(inputs, labels, c, kernel)

    def classifier(x):
        sigma = sum([
            labels[i] * alpha[i] * kernel(x, inputs[i])
            for i in range(labels.size)
        ])

        return np.sign(sigma + s)

    return classifier

def calculate_hyperplane(inputs, labels, c, kernel):
    n = labels.size
    G = calculate_gram_matrix(inputs, labels, kernel, n)
    h = cvxopt.matrix(np.full(n, -1.0))
    P = calculate_inequality_matrix(n)
    q = calculate_inequality_vector(n, c)
    A = cvxopt.matrix(labels, tc='d').trans() # tc='d' causes conversion to double
    b = cvxopt.matrix(0.0)

    sol = cvxopt.solvers.qp(G, h, P, q, A, b)
    alpha = np.array(sol['x']).reshape( (n, ) )
    s = calculate_shift(inputs, labels, c, alpha, G, n)

    return (alpha, s)

def calculate_gram_matrix(inputs, labels, kernel, n):
    G = np.empty( (n, n) )

    for i in range(0, n):
        for j in range(i, n):
            G[i, j] = G[j, i] = labels[i]*labels[j]*kernel(inputs[i], inputs[j])

    return cvxopt.matrix(G)

def calculate_inequality_matrix(n):
    np_matrix = np.vstack( (-np.eye(n), np.eye(n)) )

    return cvxopt.matrix(np_matrix)

def calculate_inequality_vector(n, c):
    zero_vector = np.zeros(n)
    constant_vector = np.full(n, c)
    np_vector = np.concatenate( (zero_vector, constant_vector) )

    return cvxopt.matrix(np_vector)

def calculate_shift(inputs, labels, c, alpha, G, n):
    sum = 0.0
    summands = 0.0
    precision = 10**(-6)*c

    for i in range(n):
        if alpha[i] > precision and c - alpha[i] > precision:
            summands += 1
            gram_row = np.array(G[i, :]).reshape((n,))
            sum += labels[i] * (1 - np.dot(alpha, gram_row))

    return sum/summands
