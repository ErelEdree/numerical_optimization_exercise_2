import numpy as np


def interior_pt(func, ineq_constraints, eq_constraints_mat,
                 eq_constraints_rhs, x0, t0=1.0, mu=10.0, tol=1e-6, max_iter=20):
    x = x0.copy()
    t = t0
    m = len(ineq_constraints)
    path = [x.copy()]
    obj_vals = [func(x)[0]]

    for _ in range(max_iter):
        for _ in range(50):  # Newton iterations
            delta_x = newton_step(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x, t)
            if np.linalg.norm(delta_x) < 1e-8:
                break
            step_size = backtracking(x, delta_x, func, ineq_constraints, t)
            x += step_size * delta_x

        path.append(x.copy())
        obj_vals.append(func(x)[0])

        if m / t < tol:
            break
        t *= mu

    return x, path, obj_vals


def log_barrier_obj_func(func, ineq_constraints, x, t):
    return (func(x)[0] + phi(ineq_constraints, x) / t,
            func(x)[1] + grad_phi(ineq_constraints, x) / t,
            func(x)[2] + hess_phi(ineq_constraints, x) / t)


def phi(ineq_constraints, x):
    return -sum(np.log(-g(x)[0]) for g in ineq_constraints)


def grad_phi(ineq_constraints, x):
    grad = np.zeros_like(x)
    for g in ineq_constraints:
        val, grad_g, _ = g(x)
        grad += grad_g / (-val)
    return grad


def hess_phi(ineq_constraints, x):
    n = x.shape[0]
    hess = np.zeros((n, n))
    for g in ineq_constraints:
        val, grad_g, hess_g = g(x)
        grad_g = grad_g.reshape((-1, 1))
        hess += (grad_g @ grad_g.T) / (val ** 2) - hess_g / val
    return hess


def newton_step(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x, t):
    obj_val, grad, hess = log_barrier_obj_func(func, ineq_constraints, x, t)
    A = eq_constraints_mat
    b = eq_constraints_rhs
    KKT_top = np.hstack([hess, A.T])
    KKT_bottom = np.hstack([A, np.zeros((A.shape[0], A.shape[0]))])
    KKT = np.vstack([KKT_top, KKT_bottom])
    rhs = -np.concatenate([grad, A @ x - b])
    sol = np.linalg.solve(KKT, rhs)
    delta_x = sol[:x.shape[0]]
    return delta_x


def backtracking(x, dx, func, ineq_constraints, t, alpha=0.01, beta=0.5):
    s = 1.0
    while True:
        x_new = x + s * dx
        if any(g(x_new)[0] >= 0 for g in ineq_constraints):
            s *= beta
            continue
        phi_new = log_barrier_obj_func(func, ineq_constraints, x_new, t)[0]
        phi_curr, grad, _ = log_barrier_obj_func(func, ineq_constraints, x, t)
        if phi_new <= phi_curr + alpha * s * grad @ dx:
            break
        s *= beta
    return s
