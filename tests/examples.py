# examples.py
import numpy as np

# ---- Quadratic Program ----
def get_qp_example():
    def func(x):
        val = x[0]**2 + x[1]**2 + (x[2]+1)**2
        grad = np.array([2*x[0], 2*x[1], 2*(x[2]+1)])
        hess = 2*np.eye(3)
        return val, grad, hess

    def make_ineq(i):
        return lambda x: ( -x[i],   # value
                           np.eye(3)[i],  # gradient
                           np.zeros((3, 3)))  # hessian is 0

    ineqs = [make_ineq(i) for i in range(3)]  # x >= 0, y >= 0, z >= 0 => -x <= 0

    A = np.ones((1, 3))  # x + y + z = 1
    b = np.array([1.0])
    x0 = np.array([0.1, 0.2, 0.7])

    return func, ineqs, A, b, x0

# ---- Linear Program ----
def get_lp_example():
    def func(x):
        val = -(x[0] + x[1])  # maximize x + y = minimize -(x + y)
        grad = np.array([-1.0, -1.0])
        hess = np.zeros((2, 2))
        return val, grad, hess

    # Constraints:
    # y >= -x + 1 => -y - x + 1 <= 0
    # y <= 1     => y - 1 <= 0
    # x <= 2     => x - 2 <= 0
    # y >= 0     => -y <= 0
    def c1(x): return (-x[1] - x[0] + 1, np.array([-1.0, -1.0]), np.zeros((2,2)))
    def c2(x): return ( x[1] - 1,       np.array([ 0.0,  1.0]), np.zeros((2,2)))
    def c3(x): return ( x[0] - 2,       np.array([ 1.0,  0.0]), np.zeros((2,2)))
    def c4(x): return (-x[1],           np.array([ 0.0, -1.0]), np.zeros((2,2)))

    ineqs = [c1, c2, c3, c4]
    A = np.zeros((0, 2))  # no equality constraints
    b = np.zeros(0)
    x0 = np.array([0.5, 0.75])

    return func, ineqs, A, b, x0
