# Generating states that have a high Schmidt rank but are 2-unfaithful (discussion above Eq.(12))

import numpy as np
import cvxpy as cp
import utils

np.set_printoptions(suppress=True, linewidth=150, precision=5)


class Trial:

    def __init__(self, d, D, reps):
        self.d = d
        self.D = D  # The Schmidt rank we want
        self.reps = reps
        self.state = self.make_init_state()

    def make_init_state(self):
        vector = utils.random_orthogonal(self.d**2)[0]
        return np.outer(vector, vector)

    def optimize(self):
        for iter_num in range(self.reps):

            value = self.see()
            to_exit = self.check_see(value)
            if to_exit is True:
                break

            value = self.saw()
            to_exit = self.check_saw(value)
            if to_exit is True:
                break

            print("")

    # Find the closest state with a lower scmidt rank, and move away from it
    def see(self):
        constraints, rho = utils.variable_max_rank_D_state(self.d, self.d, self.D-1, False)
        target = cp.norm(rho - self.state)
        pr = cp.Problem(cp.Minimize(target), constraints)
        pr.solve(solver=cp.MOSEK)
        print("see:", pr.value)
        self.make_valid_state_away_from(0.1*rho.value)
        return pr.value

    def make_valid_state_away_from(self, rho):
        self.state -= rho
        vals, vecs = np.linalg.eigh(self.state)
        vals = np.minimum(vals, np.zeros(len(vals)))
        self.state = vecs @ np.diag(vals) @ vecs.conj().T
        self.state /= np.trace(self.state)

    def check_see(self, value):
        is_npt = utils.is_NPT(self.state, self.d, self.d)
        is_rank = value > 1e-4
        is_unf = utils.is_UNF(self.state, self.d, self.d)
        return self.print_results_and_check(is_npt, is_rank, is_unf)

    # Make the state unfaithful, but as close to what it is now
    def saw(self):
        constraints, rho = utils.variable_UNF_state(self.d, self.d, False)
        target = cp.norm(rho - self.state)
        pr = cp.Problem(cp.Minimize(target), constraints)
        pr.solve(solver=cp.MOSEK)
        self.state = 0.9*self.state + 0.1*rho.value
        return pr.value

    def check_saw(self, value):
        is_npt = utils.is_NPT(self.state, self.d, self.d)
        is_rank = utils.is_schmidt_rank_greater_than_D(self.state, self.d, self.d, 2)
        is_unf = value < 1e-4
        return self.print_results_and_check(is_npt, is_rank, is_unf)

    def print_results_and_check(self, is_npt, is_rank, is_unf):
        print("rank > 1: {}    rank > 2: {}    unfaithful: {}".format(is_npt, is_rank, is_unf))
        if is_unf and is_rank:
            print(self.state)
            return True
        elif not is_unf and not is_rank:
            print("outside interesting region")
            return True
        else:
            return False


def check_previously_found_example():
    state = np.array([[0.0283, -0.00342, 0.00267, -0.03152, 0.01907, 0.00241, 0.07041, -0.03374, 0.02087],  # noqa
                     [-0.00342, 0.00754, -0.0007, -0.01227, 0.00895, 0.00489, -0.02732, 0.0147, 0.00538],  # noqa
                     [0.00267, -0.0007, 0.01653, -0.03384, 0.00742, -0.03055, 0.00028, 0.01439, 0.04042],  # noqa
                     [-0.03152, -0.01227, -0.03384, 0.1343, -0.05436, 0.06032, -0.02112, -0.0196, -0.1095],  # noqa
                     [0.01907, 0.00895, 0.00742, -0.05436, 0.05178, 0.06264, 0.01708, 0.0101, 0.07279],  # noqa
                     [0.00241, 0.00489, -0.03055, 0.06032, 0.06264, 0.28285, 0.01353, 0.00219, 0.04244],  # noqa
                     [0.07041, -0.02732, 0.00028, -0.02112, 0.01708, 0.01353, 0.22841, -0.11915, 0.01705],  # noqa
                     [-0.03374, 0.0147, 0.01439, -0.0196, 0.0101, 0.00219, -0.11915, 0.08073, 0.04553],  # noqa
                     [0.02087, 0.00538, 0.04042, -0.1095, 0.07279, 0.04244,  0.01705, 0.04553, 0.16955]])  # noqa

    print(state)
    is_rank = utils.is_schmidt_rank_greater_than_D(state, 3, 3, 2)
    is_unf = utils.is_UNF(state, 3, 3)
    print("Schmidt rank > 2: {} \t is 2-unfaithful: {}".format(is_rank, is_unf))


# Trial(3, 3, 3).optimize()

check_previously_found_example()
