import numpy as np
import scipy.linalg

def generate_random_qp_problem(n, p, condition_number, m):
    """
    Generates a random QP problem.
    
    Parameters:
    n (int): Number of variables.
    p (int): Number of observations.
    condition_number (float): Desired condition number of C.
    m (int): Number of constraints.
    
    Returns:
    Q (ndarray): nxn matrix for the quadratic term.
    q (ndarray): n-vector for the linear term.
    A (ndarray): mxn matrix for constraints coefficients.
    b (ndarray): m-vector for constraints bounds.
    """
    # Generate random orthogonal matrices P and Q using QR decomposition
    P = scipy.linalg.qr(np.random.randn(p, p))[0]
    Q = scipy.linalg.qr(np.random.randn(n, n))[0]
    
    # Generate the diagonal matrix D for condition number
    D_values = np.linspace(1.0 / condition_number, condition_number, min(p, n))
    D = np.diag(D_values)
    if p > n:
        D = np.vstack([D, np.zeros((p-n, n))])
    elif n > p:
        D = np.hstack([D, np.zeros((p, n-p))])
    
    # Compute C using P, D, and Q
    C = np.dot(P, np.dot(D, Q))
    
    # Generate the Hessian matrix Q for the quadratic term
    Q_mat = np.dot(C.T, C)
    
    # Generate linear term q randomly
    q = np.random.randn(n)
    
    # Generate constraints A and b
    A = np.random.randn(m, n)
    b = np.random.randn(m)
    
    return Q_mat, q, A, b


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1])
    p = 15
    condition_number = 10
    m = 2 * n 
    q_mat, q, A, b = generate_random_qp_problem(n, p, condition_number, m)

    print(*q_mat.flatten(), sep=',', end=';')
    print(*q.flatten(), sep=',', end=';')
    print(*A.flatten(), sep=',', end=';')
    print(*b.flatten(), sep=',')
