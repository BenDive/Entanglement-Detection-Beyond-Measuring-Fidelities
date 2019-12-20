# Show, by example, that there exists self-activatable states for faithfulness Eq.(16)
import numpy as np
import cvxpy as cp
import utils

np.set_printoptions(linewidth=120, suppress=True)

# The state as written in Eq.(16)
vec1 = np.array([0, 0, 0, 0, 0.628, 0, 0, 0, -0.778])
vec2 = np.array([0, 0.807, -0.185, -0.102, -0.0270, 0.0111, 0.555, -0.0236, -0.0218])
state = 0.501746 * np.outer(vec1, vec1) + 0.498174 * np.outer(vec2, vec2)
state /= np.trace(state)
state = (1-0.001) * state + 0.001 * np.eye(9)/9

# Show that a single copy of it as unfaithful
unfaithful = utils.is_UNF(state, 3, 3)
print("State is unfaithful: {}".format(unfaithful))


# Checking that the state is self-activating

def pure_witness_ansatz(rho):
    constraints, F, Lam = utils._UNF_witness(9, 9, hermitian=False)
    target = F - cp.real(cp.trace(rho@Lam))
    pr = cp.Problem(cp.Minimize(target), constraints)
    pr.solve(solver=cp.MOSEK)
    eig_vals, eig_vecs = np.linalg.eig(Lam.value)
    return eig_vecs[:, 0]


def find_F_for_Lam(Lam):
    F = cp.Variable(1)
    rho = utils._make_cp_matrix((81, 81), hermitian=False)
    constraints = [cp.trace(rho) == 1, rho >> 0]
    constraints += [utils.partial_transpose(rho, [0], [9, 9]) >> 0]
    constraints += [cp.trace(rho @ Lam) == F]
    pr = cp.Problem(cp.Maximize(F), constraints)
    pr.solve(solver=cp.MOSEK)
    return F.value


# Double the state
reorder_swap = np.kron(np.kron(np.eye(3), utils.swap(3, 3)), np.eye(3))
doubled_state = reorder_swap @ np.kron(state, state) @ reorder_swap.T

# Find a pure state (\ket{\psi}_W) from which we can construct a witness for it
witness_vec = pure_witness_ansatz(doubled_state)
print("pure state as base of fidelity\n", witness_vec)
witness = np.outer(witness_vec, witness_vec)
witness /= np.trace(witness)

# Find 'F' in order to have a valid witness
F = find_F_for_Lam(witness)
print("F=", F)

activated = ((F - np.trace(witness @ doubled_state)) < 0)
print("State is activated: {}".format(activated))
