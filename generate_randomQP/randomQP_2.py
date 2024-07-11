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

# Example usage
n = 500  # Number of variables
p = 15  # Number of observations
condition_number = 10  # Desired condition number
# m = 5   # Number of constraints

m = 2*n

Q, q, A, b = generate_random_qp_problem(n, p, condition_number, m)

# Display generated matrices and vectors
# print("Q (Hessian matrix):\n", Q)
# print("\nq (Linear term vector):\n", q)
# print("\nA (Constraints coefficients):\n", A)
# print("\nb (Constraints bounds):\n", b)

# Save variables to a text file
Q_flat = Q.flatten()
A_flat = A.flatten()

text_file = "qp_variables_n{}.txt".format(n)
with open(text_file, 'w') as f:
    # Matrix Q
    f.write('double Q_arr[%d]= {\n' %(len(Q_flat)))
    for i in range(0, Q.shape[0]):
        for j in range(0, Q.shape[1]):
            f.write('%.4f' %(Q[i,j]))
            if ((i+1) * (j+1) < len(Q_flat) - 1):
                f.write(',')
        f.write('\n')
    f.write('};\n')
    f.write('matrix Q(%d, %d, Q_arr);\n' %(Q.shape[0], Q.shape[1]))
    
    # Vector c (here c = q)
    f.write('\n')
    f.write('double c_arr[%d] = {\n' %len(q))
    for i in range(0, len(q)):
        f.write('%.4f' %(q[i]))
        if(i < len(q)-1):
            f.write(',')
        f.write('\n')
    f.write('};\n')
    f.write('vector c(%d, c_arr);\n' %(len(q)))

    # Matrix J (here J = A)
    f.write('\n')
    f.write('double J_arr[%d]= {\n' %(len(A_flat)))
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            f.write('%.4f' %(A[i,j]))
            if ((i+1) * (j+1) < len(A_flat)):
                f.write(',')
        f.write('\n')
    f.write('};\n')
    f.write('matrix J(%d, %d, J_arr);\n' %(A.shape[0], A.shape[1]))

    # Vector g (here g = b)
    f.write('\n')
    f.write('double g_arr[%d] = {\n' %len(b))
    for i in range(0, len(b)):
        f.write('%.4f' %(b[i]))
        if(i < len(b)-1):
            f.write(',')
        f.write('\n')
    f.write('};\n')
    f.write('vector g(%d, g_arr);\n' %(len(b)))
    
