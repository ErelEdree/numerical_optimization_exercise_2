import numpy as np


def interior_pt(func, ineq_constraints, eq_constraints_mat,
eq_constraints_rhs, x0):
    """
    interior point method for constrained minimization using 
    the log barrier method.
    """

def log_barrier_obj_func(func,ineq_constraints,x,t):
    """
    the log barrier objective function. returns as a three-tuple the value, 
    the gradient, and the Hessian.
    """
    return (func(x)[0] + phi(ineq_constraints,x)/t,
            func(x)[1] + grad_phi(ineq_constraints,x)/t,
            func(x)[2] + hess_phi(ineq_constraints,x)/t)

def phi(ineq_constraints,x):
    return -sum(np.log(-ineq_constraints(x)[0]) for ineq_constraints in ineq_constraints)

def grad_phi(ineq_constraints,x):
    return sum(ineq_constraints(x)[1] / (-ineq_constraints(x)[0]) for ineq_constraints in ineq_constraints)

def hess_phi(ineq_constraints,x):
    first_sum = sum(ineq_constraint(x)[1]@ineq_constraint(x)[1].T / (ineq_constraint(x)[0])**2 for ineq_constraint in ineq_constraints)
    second_sum = sum(ineq_constraint(x)[2]/(-ineq_constraint(x)[0]) for ineq_constraint in ineq_constraints)
    return first_sum + second_sum

def newton_step(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x, t):
    """
    the Newton step for the log barrier objective function.
    """
    obj_func = log_barrier_obj_func(func,ineq_constraints,x,t)
    grad = obj_func[1]
    hess = obj_func[2]
    #use KKT and eq_constraints_mat
    A = eq_constraints_mat
    b = eq_constraints_rhs
    KKT_top = np.hstack([hess, A.T])
    KKT_bottom = np.hstack([A, np.zeros((A.shape[0], A.shape[0]))])
    KKT = np.vstack([KKT_top, KKT_bottom])
    rhs = -np.concatenate([grad, A @ x - b])
    sol = np.linalg.solve(KKT, rhs)
    delta_x = sol[:x.shape[0]]
    return delta_x


