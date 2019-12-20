# Explores how pure states become unfaithul as white noise is added to them
# (Fig.II and discussion afterwards)

import numpy as np
import cvxpy as cp
import utils

np.set_printoptions(suppress=True)


class BoundariesForState:
    """ Gives how much noise needs to be added for a pure state to become
    unfaithful, reducible and PPT"""

    def __init__(self, dim, hermitian):
        self.dim = dim
        self.iden = np.eye(dim**2) / dim**2
        self.hermitian = hermitian

    def make_state(self, noise):
        raise NotImplementedError

    def max_noise_entangled(self):
        sigma = self.make_state(0)
        p_trans = utils.partial_transpose(sigma, [0], [self.dim, self.dim])
        lambda_min = min(np.real(np.linalg.eigvals(p_trans)))
        return lambda_min / (lambda_min - 1/self.dim**2)

    def min_noise_unfaithful(self):
        noise = cp.Variable(1)
        state = self.make_state(noise)
        constraints, rho = utils.variable_UNF_state(self.dim, self.dim, hermitian=self.hermitian)
        constraints += [noise >= 0, noise <= 1, state == rho]
        pr = cp.Problem(cp.Minimize(noise), constraints)
        pr.solve(solver=cp.MOSEK)
        return noise.value[0]

    def min_noise_reducible(self):
        sigma = self.make_state(0)
        sigmaB = utils.partial_trace(sigma, [0], [self.dim, self.dim])
        sigmaB_ext = utils.tensor(np.eye(self.dim), sigmaB)
        eigval = min(np.real(np.linalg.eigvals(sigmaB_ext - sigma)))
        return eigval / (eigval - (self.dim - 1)/self.dim**2)

    def min_noise_D_unfaithful(self, D):
        noise = cp.Variable(1)
        iden = np.eye(self.dim)
        state = self.make_state(noise)
        Na = utils._make_cp_matrix((self.dim, self.dim), False)
        Nb = utils._make_cp_matrix((self.dim, self.dim), False)

        constraints = [noise >= 0, noise <= 1, Na >> 0, Nb >> 0]
        constraints += [utils.tensor(Na, iden) + utils.tensor(iden, Nb) >> state]
        constraints += [cp.trace(Na)*iden >> (D-1)*Na]
        constraints += [cp.trace(Nb)*iden >> (D-1)*Nb]
        constraints += [cp.trace(Na) + cp.trace(Nb) == D-1]
        pr = cp.Problem(cp.Minimize(noise), constraints)
        pr.solve(solver=cp.MOSEK)
        return noise.value[0]

    def min_noise_to_have_rank_D_or_less(self, D):
        # Saves some memory by not redefining a cvxpy state here, and rewriting the constraints
        dT = self.dim**2 * D**2
        iden = np.eye(self.dim)
        proj = utils.tensor(iden, np.sqrt(D)*utils.max_entangled_ket(D), iden)

        noise = cp.Variable(1)
        state = self.make_state(noise)
        sigma = utils._make_cp_matrix((dT, dT), False)

        constraints = [noise >= 0, noise <= 1]
        constraints += [sigma >> 0, cp.trace(sigma) == D]
        constraints += [proj @ sigma @ proj.T == state]
        constraints += [utils.partial_transpose(sigma, [0], [self.dim*D, self.dim*D]) >> 0]

        pr = cp.Problem(cp.Minimize(noise), constraints)
        pr.solve(solver=cp.MOSEK)
        return noise.value[0]


class BoundariesForRank2States(BoundariesForState):
    """For the states p id + (1-p) (|00> + |11>)(<00| + <11|) / 2"""

    def make_state(self, noise):
        pure_vec = np.zeros(self.dim**2)
        pure_vec[0*self.dim + 0] = 1
        pure_vec[1*self.dim + 1] = 1
        pure_mat = np.outer(pure_vec, pure_vec) / 2
        return noise*self.iden + (1-noise)*pure_mat


class BoundariesForRank3States(BoundariesForState):
    """For the states p id + (1-p) (|00> + |11> + |22>)(<00| + <11| + <22|) / 3"""

    def make_state(self, noise):
        pure_vec = np.zeros(self.dim**2)
        pure_vec[0*self.dim + 0] = 1
        pure_vec[1*self.dim + 1] = 1
        pure_vec[2*self.dim + 2] = 1
        pure_mat = np.outer(pure_vec, pure_vec) / 3
        return noise*self.iden + (1-noise)*pure_mat


class BoundariesForRandomStates(BoundariesForState):
    """ For random pure states calculated according to the Haar measure """

    def __init__(self, dim, hermitian):
        super().__init__(dim, hermitian)
        self.random_unitary = utils.random_unitary(self.dim**2)

    def make_state(self, noise):
        pure_mat = np.outer(self.random_unitary[0].conj(), self.random_unitary[0])
        return noise*self.iden + (1-noise)*pure_mat


def entangled_and_unfaithful():
    print("Amount of noise for state to have rank 2 entanglement and be 2-unfaithful")
    print("states = (1-p) (|00> + |11>)(<00| + <11|) / 2  +  p/d Id_d")
    for d in range(2, 13):
        trial = BoundariesForRank2States(d, hermitian=False)
        print("d={}:\t {:.3f}-{:.3f}-{:.3f}".format(
            d, trial.min_noise_unfaithful(), trial.min_noise_reducible(),
            trial.max_noise_entangled()))


def rank_3_entangled_and_rank_3_unfaithful():
    print("Amount of noise for state to have rank 3 entanglement and be 3-unfaithful")
    print("states = (1-p) (|00> + |11> + |22>)(<00| + <11| + <22|) / 3  +  p/d Id_d")
    for d in range(3, 6):
        trial = BoundariesForRank3States(d, hermitian=False)
        min_noise = trial.min_noise_D_unfaithful(3)
        max_noise = trial.min_noise_to_have_rank_D_or_less(3-1)
        print("d={}\t {:.3f}-{:.3f}".format(d, min_noise, max_noise))


def almost_all_pure_states_become_unfaithful_while_entangled():
    repeats = int(10e6)
    ds = [3, 4, 5]

    print("Mixing random pure states with varying white noise in different dimensions")

    for d in ds:
        become_unfaithful = 0
        for _ in range(repeats):
            trial = BoundariesForRandomStates(d, hermitian=True)

            # Check this first, reducibility much quicker than unfaithfulness to evaluate
            if trial.max_noise_entangled() - trial.min_noise_reducible() > utils.epsilon:
                become_unfaithful += 1

            elif trial.max_noise_entangled() - trial.min_noise_unfaithful() > utils.epsilon:
                become_unfaithful += 1

        print("d={}:\t states entangled and unfaithful for some amount of noise: {}/{}".format(
            d, become_unfaithful, repeats))


# Fig.II
entangled_and_unfaithful()

# Paragraph just below Eq.(13)
almost_all_pure_states_become_unfaithful_while_entangled()

# Last paragraph of "Noisy pure states are unfaithful"
rank_3_entangled_and_rank_3_unfaithful()
