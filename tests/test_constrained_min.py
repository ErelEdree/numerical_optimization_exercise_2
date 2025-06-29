# test_constrained_min.py
import numpy as np
import matplotlib.pyplot as plt
import unittest
from src.constrained_min import interior_pt
from tests.examples import get_qp_example, get_lp_example
from src.utils import plot_3d_central_path, plot_2d_central_path, plot_objective_values


class TestConstrainedMin(unittest.TestCase):
    def test_qp(self):
        func, ineqs, A, b, x0 = get_qp_example()
        sol, path, obj_vals = interior_pt(func, ineqs, A, b, x0)

        # Plot 3D central path with feasible region
        plot_3d_central_path(path, title="QP Central Path", xlabel="x", ylabel="y", zlabel="z", 
                           ineq_constraints=ineqs, eq_constraints_mat=A, eq_constraints_rhs=b,
                           save=True, filename="qp_central_path.png")
        
        # Plot objective values
        plot_objective_values(obj_vals, title="Objective vs Iteration (QP)", save=True,filename="qp_obj_vals.png" )

        print("QP solution:", sol)
        print("Objective:", obj_vals[-1])
        print("Equality constraint (Ax = b):", A @ sol - b)
        for i, c in enumerate(ineqs):
            print(f"Ineq {i}:", c(sol)[0])

    def test_lp(self):
        func, ineqs, A, b, x0 = get_lp_example()
        sol, path, obj_vals = interior_pt(func, ineqs, A, b, x0)

        # Plot 2D central path with feasible region
        plot_2d_central_path(path, title="LP Central Path", xlabel="x", ylabel="y", 
                           ineq_constraints=ineqs, save=True, filename="lp_central_path.png")
        
        # Plot objective values
        negative_obj_vals = [-obj for obj in obj_vals]
        plot_objective_values(negative_obj_vals, title="Objective vs Iteration (LP)", save=True,filename="lp_obj_vals.png")

        print("LP solution:", sol)
        print("Objective:", obj_vals[-1])
        for i, c in enumerate(ineqs):
            print(f"Ineq {i}:", c(sol)[0])


if __name__ == '__main__':
    unittest.main()