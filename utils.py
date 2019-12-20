# Various utility functions used by the scripts

import functools
import itertools
import numbers
import collections.abc
import cvxpy as cp
import numpy as np

# Caches the matrices used to compute swaps, partial traces and partial transposes
# Switch to "True" to maximize speed, switch to "False" to minimize RAM usage
cache = True

# Used as tolerance when testing inequalities about 0
epsilon = 1e-6


########################################
""" PUBLIC FUNCTIONS USED IN SCRIPTS """
########################################


""" Random operator creation """


def random_real_state(d):
    """Creates a random d dim, real quantum state"""
    rho = np.random.normal(0, 1, size=(d, d))
    rho = rho @ rho.T
    rho = rho / np.trace(rho)
    return rho


def random_HS_state(d):
    """Creates a random d dim quantum state distributed according to the Hilbert-Schmidt measure"""
    rho = np.random.normal(0, 1, size=(d, d)) + np.random.normal(0, 1, size=(d, d))*1j
    rho = rho @ rho.conj().T
    rho = rho / np.trace(rho)
    return rho


def random_bures_state(d):
    """Creates a random d dim quantum state distributed according to the Bures measure"""
    gin = np.random.normal(0, 1, size=(d, d)) + np.random.normal(0, 1, size=(d, d))*1j
    uni = random_unitary(d)
    iden = np.eye(d)
    rho = (iden + uni) @ gin @ gin.conj().T @ (iden + uni.conj().T)
    rho /= np.trace(rho)
    return rho


def max_entangled_ket(d):
    """Maximally entangled state between two systems of dimension d. Normalised"""
    ket = np.zeros(d**2)
    val = 1/np.sqrt(d)
    for k in range(d):
        ket[k*d + k] = val
    return ket


def random_unitary(d):
    """Creates a random d dim unitary matrix distributed according to the Haar measure"""
    gin = np.random.normal(0, 1, size=(d, d)) + np.random.normal(0, 1, size=(d, d))*1j
    q, r = np.linalg.qr(gin)
    u = q @ np.sign(np.diag(np.diag(r)))
    return u


def random_orthogonal(d):
    """Creates a random d dim orthogonal matrix distributed according to the Haar measure"""
    gin = np.random.normal(0, 1, size=(d, d))
    q, r = np.linalg.qr(gin)
    u = q @ np.sign(np.diag(np.diag(r)))
    return u


def variable_max_rank_D_state(da, db, D, hermitian=True):
    """
    SDP1 FROM THE PAPER
    Creates a bipartite state as a variable, with the constraints for it to have Schmidt Rank <= D
    da, db:         Dimensions of the two subsystems
    D:              Maximum Schmidt rank imposed by the constrints
    hermitian       If True, rho and constraints are complex, if false, only real
    OUTPUT:
    constraints:    list of SDP constraints for cvxpy
    rho:            cvxpy variable matrix that, when optimised over the constraints, is the state
    """
    dT = da*D*D*db
    proj = tensor(np.eye(da), np.sqrt(D)*max_entangled_ket(D), np.eye(db))
    rho = _make_cp_matrix((da*db, da*db), hermitian)
    sigma = _make_cp_matrix((dT, dT), hermitian)

    constraints = [rho >> 0, sigma >> 0]
    constraints += [cp.trace(rho) == 1, cp.trace(sigma) == D]
    constraints += [proj @ sigma @ proj.T == rho]
    constraints += [partial_transpose(sigma, [0], [da*D, D*db]) >> 0]
    return constraints, rho


def variable_UNF_state(da, db, hermitian=True):
    """
    SDP2 FROM THE PAPER - D=2 CASE
    Creates a bipartite state as a variable, with the constraints for it to be unfaithful
    da, db:         Dimensions of the two subsystems
    hermitian       If True, rho and constraints are complex, if false, only real
    OUTPUT:
    constraints:    list of SDP constraints for cvxpy
    rho:            cvxpy variable matrix that, when optimised over the constraints, is the state
    This is equivalent to variable_D_UNF_state(da, db, D, hermitian), but faster.
    """
    rho = _make_cp_matrix((da*db, da*db), hermitian)
    Na = _make_cp_matrix((da, da), hermitian)
    Nb = _make_cp_matrix((db, db), hermitian)

    constraints = [rho >> 0, cp.trace(rho) == 1, Na >> 0, Nb >> 0]
    constraints += [tensor(Na, np.eye(db)) + tensor(np.eye(da), Nb) >> rho]
    constraints += [cp.trace(Na) + cp.trace(Nb) == 1]

    return constraints, rho


def variable_D_UNF_state(da, db, D, hermitian=True):
    """
    SDP2 FROM THE PAPER - GENERAL D CASE
    creates a bipartite state as a variable, with the constraints for it to be D-unfaithful
    da, db:         Dimensions of the two subsystems
    D:              Dimension of unfaithfulness to satisfy
    hermitian       If True, rho and constraints are complex, if false, only real
    OUTPUT:
    constraints:    list of SDP constraints for cvxpy
    rho:            cvxpy variable matrix that, when optimised over the constraints, is the state
    """
    rho = _make_cp_matrix((da*db, da*db), hermitian)
    Na = _make_cp_matrix((da, da), hermitian)
    Nb = _make_cp_matrix((db, db), hermitian)

    constraints = [rho >> 0, cp.trace(rho) == 1, Na >> 0, Nb >> 0]
    constraints += [tensor(Na, np.eye(db)) + tensor(np.eye(da), Nb) >> rho]
    constraints += [cp.trace(Na)*np.eye(da) >> (D-1)*Na]
    constraints += [cp.trace(Nb)*np.eye(db) >> (D-1)*Nb]
    constraints += [cp.trace(Na) + cp.trace(Nb) == D-1]

    return constraints, rho


""" Determining properties of quantum states """


def is_NPT(rho, da, db):
    """
    Does the partial transpose of the state have at least 1 negative eigenvalue?
    rho:        bipartite matrix whose NPT status is to be determined
    da, db:     Dimensions of the two subsystems of the state, in that order
    """
    minPT = min(np.real(np.linalg.eigvals(partial_transpose(rho, [0], [da, db]))))
    if minPT < -epsilon:
        return True  # State is NPT (definitely entangled)
    else:
        return False  # State is PPT (could be entangled or not)


def is_PPT(rho, da, db):
    """Inverse of is_NPT()"""
    return not is_NPT(rho, da, db)


def is_reducible(rho, da, db):
    """
    Is the state of the form of one of:
        id x rho_b - rho_ab > 0
        rho_a x id - rho_ab > 0?
    rho:        bipartite matrix whose reducible status is to be determined
    da, db:     Dimensions of the two subsystems of the state, in that order
    """

    rho_B = partial_trace(rho, [0], [da, db])
    reduce_B = np.kron(np.eye(da), rho_B) - rho
    evals_B = min(np.real(np.linalg.eigvals(reduce_B)))
    if evals_B > 0:
        return True

    rho_A = partial_trace(rho, [1], [da, db])
    reduce_A = np.kron(rho_A, np.eye(db)) - rho
    evals_A = min(np.real(np.linalg.eigvals(reduce_A)))
    if evals_A > 0:
        return True

    return False


def is_UNF(rho, da, db):
    """
    SDP2 FROM THE PAPER - D=2 CASE
    Is the state in \tilde(U), an inner approximation of U
    rho:        bipartite matrix whose faithfulness is to be determined
    da, db:     Dimensions of the two subsystems of the state, in that order
    This is equivalent to is_UNF(rho, da, db, D), but faster.
    """
    constraints, F, Lam = _UNF_witness(da, db, hermitian=np.iscomplexobj(rho))

    target = F - cp.real(cp.trace(rho@Lam))
    pr = cp.Problem(cp.Minimize(target), constraints)
    pr.solve(solver=cp.MOSEK)

    if pr.value < -epsilon:
        return False  # IT IS FAITHFUL - entanglement detectable with pure state fidelity
    else:
        return True  # IT IS UNFAITHFUL - but not necessarily entangled


def is_D_UNF(rho, da, db, D):
    """
    SDP2 FROM THE PAPER - GENERAL D CASE
    Is the state in \tilde(U)_d, an inner approximation of U_d
    If it is in that set, there is no rank 1 witness that can certify it has Schmidt rank >= d.
    rho:        bipartite matrix whose faithfulness is to be determined
    da, db:     Dimensions of the two subsystems of the state, in that order
    D:          Schmidt rank
    """
    constraints, F, Lam = _UNF_D_witness(da, db, D, hermitian=np.iscomplexobj(rho))

    target = F - cp.real(cp.trace(rho@Lam))
    pr = cp.Problem(cp.Minimize(target), constraints)
    pr.solve(solver=cp.MOSEK)

    if pr.value < -epsilon:
        return False  # IT IS d-FAITHFUL - Schmidt rank d detectable with pure state fidelities
    else:
        return True  # IT IS d-UNFAITHFUL - but not necessarily Schmidt rank d


def is_schmidt_rank_greater_than_D(rho, da, db, D):
    """
    SDP1 FROM THE PAPER
    Does the state have Schmidt greater than D, as witnessed by decomposable witnesses
    rho:        bipartite matrix whose rank is to be determined
    da, db:     Dimensions of the two subsystems of the state, in that order
    D:          Schmidt rank to witness
    """
    constraints, W = _schmidt_rank_witness(da, db, D, hermitian=np.iscomplexobj(rho))
    obj = cp.Minimize(cp.real(cp.trace(rho @ W)))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK)
    if prob.value < -epsilon:
        return True  # rho has Schmidt rank GREATER than D
    else:
        return False  # rho has Schmidt rank LESS than or equal to D


""" Common transformations on quantum states """


def tensor(*args):
    """Tensors any number of np.ndarray and up to one cp.variable array together"""
    return functools.reduce(_custom_kron, args)


def partial_transpose(rho, part_ind, d):
    """
    computes the partial transpose of an np/cp array
    rho:        matrix to be partially transposed
    part_ind:   indices of the systems to transpose, list of integers between 0 to len(d)-1
    d:          list of dimensions of each system
    """
    assert all(x >= 0 and isinstance(x, numbers.Integral) for x in part_ind+d),\
        "Input vectors must be non-negative integers."
    assert len(part_ind) == len(set(part_ind)) and set(part_ind).issubset(set(range(len(d)))),\
        "The list of systems to transpose must contain unique numbers in 0, ..., len(d)."
    assert np.prod(d) == rho.shape[0] and np.prod(d) == rho.shape[1],\
        "Mismatch between reported dimensions and matrix size."

    indexes = _index_tuples(part_ind, d)

    p_trans = np.zeros((np.prod(d), np.prod(d)), dtype=complex)
    for i in itertools.product(*indexes):
        for j in itertools.product(*indexes):
            ope = _partial_transpose_index_target(i, j, part_ind, d)
            p_trans += ope @ rho @ ope
    return p_trans


def partial_trace(rho, part_ind, d):
    """
    computes the partial trace of an np/cp array
    rho:        matrix to be partially traced
    part_ind:   indices of the systems to trace out, list of integers between 0 to len(d)-1
    d:          list of dimensions of each system
    """
    assert all(x >= 0 and isinstance(x, numbers.Integral) for x in part_ind+d),\
        "Input vectors must be non-negative integers."
    assert len(part_ind) == len(set(part_ind)) and set(part_ind).issubset(set(range(len(d)))),\
        "The list of systems to traced out must contain unique numbers in 0, ..., len(d)."
    assert np.prod(d) == rho.shape[0] and np.prod(d) == rho.shape[1],\
        "Mismatch between reported dimensions and matrix size."

    indexes = _index_tuples(part_ind, d)

    dim_fin = int(np.prod(d) / np.prod([d[j] for j in part_ind]))
    p_trace = np.zeros((dim_fin, dim_fin), dtype=complex)
    for i in itertools.product(*indexes):
        ope = _partial_trace_index_target(i, part_ind, d)
        p_trace += ope @ rho @ ope.T
    return p_trace


def swap(da, db):
    """
    Creates the matrix that performs a swap on a bipartite system such that:
        S|v_a> |v_b> = |v_b> |v_a>
    da, db:     Dimensions of the two subsystem, in there original order
    """
    return matrices(_Swap, da, db)


###########################################
""" PRIVATE FUNCTIONS USED IN THE ABOVE """
###########################################


def _custom_kron(A, B):
    """ cp.kron only works if the cp.variable array is the second argument """
    # Second argument is not constant, can do cp.kron as normal
    if not isinstance(B, np.ndarray) and not isinstance(B, numbers.Number):
        return cp.kron(A, B)
    # First argument is not constant, swap the order
    elif not isinstance(A, np.ndarray) and not isinstance(A, numbers.Number):
        # find dimensions of both matrices
        dAL, dAR = A.shape
        # if B is not an np array, it must be a number, in which case its dimension is 1
        if isinstance(B, np.ndarray):
            dBL, dBR = B.shape
        else:
            dBL = 1
            dBR = 1
        swap_left = swap(int(dBL), int(dAL))
        swap_right = swap(int(dBR), int(dAR)).T
        return swap_left @ cp.kron(B, A) @ swap_right
    # Both elements are constant, can do np.kron as normal
    else:
        return np.kron(A, B)


def _index_tuples(len_list, index_list):
    """ Indices to iterate over when computing partial traces or partial transposes """
    indexes = []
    for j in len_list:
        indexes += [(range(index_list[j]))]
    return tuple(indexes)


def _ket(k, d):
    """generates the basis state |k> in dimension d"""
    v = np.zeros((d, 1))
    v[k, 0] = 1
    return v


def _bra(k, d):
    """generates the basis state <k| in dimension d"""
    v = np.zeros((1, d))
    v[0, k] = 1
    return v


def _UNF_witness(da, db, hermitian=False):
    """
    Generates an ansatz for a pair (F, Lam) which together form a witness for unfaithul states.
    da, db:     Dimensions of the two hilbert spaces we want a witness for
    hermitian:  If True, produces complex witness/constraints; if False, only real
    OUTPUTS
    constrains: list of SDP constraints for cvxpy
    F:          Scalar component of the unfaithfulness witness
    Lam:        Matrix component of the unfaithlfulness witness
    Equivalent to _UNF_witness(da, db, 2, hermitian), but faster
    """
    F = cp.Variable(1)
    Lam = _make_cp_matrix((da*db, da*db), hermitian=hermitian)
    constraints = [Lam >> 0, cp.trace(Lam) == 1, F >= 0]
    constraints += [- partial_trace(Lam, [0], [da, db]) + np.eye(da)*F >> 0]
    constraints += [- partial_trace(Lam, [1], [da, db]) + np.eye(db)*F >> 0]
    return constraints, F, Lam


def _UNF_D_witness(da, db, D, hermitian=False):
    """
    Generates an ansatz for a pair (F, Lam) which together form a witness for D-unfaithul states.
    da, db:     Dimensions of the two hilbert spaces we want a witness for
    D:          Unfaithfulness dimension to be witnesses
    hermitian:  If True, produces complex witness/constraints; if False, only real
    OUTPUTS
    constrains: list of SDP constraints for cvxpy
    F:          Scalar component of the unfaithfulness witness
    Lam:        Matrix component of the unfaithlfulness witness
    """
    # Define needed variables, including latent ones
    F, ta, tb = cp.Variable(1), cp.Variable(1), cp.Variable(1)
    Xa = _make_cp_matrix((da, da), hermitian=hermitian)
    Xb = _make_cp_matrix((db, db), hermitian=hermitian)
    Lam = _make_cp_matrix((da*db, da*db), hermitian=hermitian)

    # Constraints to be satisfied
    constraints = [Lam >> 0, cp.trace(Lam) == 1, Xa >> 0, Xb >> 0, ta >= 0, tb >= 0, F >= 0]
    constraints += [ta*np.eye(da) + Xa - partial_trace(Lam, [1], [da, db]) >> 0]
    constraints += [tb*np.eye(db) + Xb - partial_trace(Lam, [0], [da, db]) >> 0]

    # Because cp.real() of a real variable is inexplicably not implemented
    if hermitian:
        constraints += [F - (D-1)*ta - cp.real(cp.trace(Xa)) >= 0]
        constraints += [F - (D-1)*tb - cp.real(cp.trace(Xb)) >= 0]
    else:
        constraints += [F - (D-1)*ta - cp.trace(Xa) >= 0]
        constraints += [F - (D-1)*tb - cp.trace(Xb) >= 0]
    return constraints, F, Lam


def _schmidt_rank_witness(da, db, D, hermitian=False):
    """
    Generates an ansatz of the dual cone of bipartite states with Schmidt rank D
    (namely, bipartite MPSs with bond dimension D=2). Those correspond to operators M such that
    P^T MP=W+mu(D.PT^P-1), where W is an entanglement witness and P=1_A x <Psi_D^+| x 1_B.
    We approximate the set of witnesses W by the set of decomposable witnesses.
    The systems are in the order A B D D
    da, db:     Dimensions of the two hilbert spaces we want a witness for
    D:          Schmidt rank dimension to witness
    hermitian:  If True, produces complex witness/constraints; if False, only real
    OUTPUTS
    constrains: list of SDP constraints for cvxpy
    W:          cvxpy variable matrix that, when optimised over the constraints, is the witness
    """
    dT = da*db*D*D

    # Create Witness and slack variables used in constraints
    W = _make_cp_matrix((da*db, da*db), hermitian)
    slack1 = _make_cp_matrix((dT, dT), hermitian)
    slack2 = _make_cp_matrix((dT, dT), hermitian)

    # Create useful operators for constraints
    P = tensor(np.eye(da*db), np.sqrt(D)*max_entangled_ket(D))
    proj_M = P.T @ W @ P
    p_trans_slack = partial_transpose(slack2, [0, 2], [da, db, D, D])
    scalar_slack = cp.Variable(1) * (D * P.T @ P - np.eye(dT))

    # Assemble constraints and return
    constraints = [slack1 >> 0, slack2 >> 0,  W << np.eye(da*db)]
    constraints += [proj_M == slack1 + p_trans_slack + scalar_slack]
    return constraints, W


def _make_cp_matrix(dims, hermitian=False):
    if hermitian is False:
        return cp.Variable(dims, symmetric=True)
    else:
        return cp.Variable(dims, hermitian=True)


##################################
""" HANDLING OF MATRIX CACHING """
##################################


def _partial_transpose_index_target(i, j, part_ind, d):
    return matrices(_PartialTransposeIndexTarget, i, j, part_ind, d)


def _partial_trace_index_target(i, part_ind, d):
    return matrices(_PartialTraceIndexTarget, i, part_ind, d)


class _MatrixStorage:
    """
    Takes a matrix creation class and its arguments.
    If it has already called that creator with those arguments before, returns the matrix
    Otherwise, it calls that creator and saves the matrix in a dictionary before returning it
    """
    def __init__(self):
        self.matrix_dict = {}

    def __call__(self, class_ref, *args):
        key = self.make_key(class_ref, *args)

        if key in self.matrix_dict:
            return self.matrix_dict[key]
        else:
            matrix = class_ref(*args).make_matrix()
            if cache:
                self.matrix_dict[key] = matrix
            return matrix

    def make_key(class_ref, *args):
        list_args = []
        for arg in args:
            if isinstance(arg, collections.abc.Iterable):
                list_args.append(tuple(arg))
            else:
                list_args.append(arg)
        return tuple((class_ref, tuple(list_args)))


class _Swap:
    """
    Matrix that performs a swap operation on a bipartite space of dimensions da x db
    """
    def __init__(self, da, db):
        self.da = da
        self.db = db
        self.dim = self.da * self.db

    def make_matrix(self):
        aux = np.zeros((self.dim, self.dim))
        for i in range(self.da):
            for j in range(self.db):
                right = _ket(i*self.db + j, self.dim)
                left = _bra(j*self.da + i, self.dim)
                aux += np.outer(left, right)
        return aux


class _PartialTransposeIndexTarget:
    """
    auxiliary function to prepare a matrix of the form
    # id otimes..otimes |j_0><i_0|otimes id.. otimes |j_len(part_ind)-1><i_len(part_ind)-1|otimes
    """
    def __init__(self, i, j, part_ind, d):
        self.i = i
        self.j = j
        self.part_ind = part_ind
        self.d = d

    def make_matrix(self):
        aux = np.array([[1]], dtype=complex)
        pos = 0
        for k in range(len(self.d)):
            if k in self.part_ind:
                aux = tensor(aux, _ket(self.j[pos], self.d[k]) @ _bra(self.i[pos], self.d[k]))
                pos = pos+1
            else:
                aux = tensor(aux, np.eye(self.d[k]))
        return aux


class _PartialTraceIndexTarget:
    """
    auxiliary function to prepare a matrix of the form
    # id otimes..otimes <i_0|otimes id.. otimes <i_len(part_ind)-1|otimes
    """
    def __init__(self, i, part_ind, d):
        self.i = i
        self.part_ind = part_ind
        self.d = d

    def make_matrix(self):
        aux = np.array([[1]], dtype=complex)
        pos = 0
        for k in range(len(self.d)):
            if k in self.part_ind:
                aux = tensor(aux, _bra(self.i[pos], self.d[k]))
                pos = pos+1
            else:
                aux = tensor(aux, np.eye(self.d[k]))
        return aux


matrices = _MatrixStorage()
