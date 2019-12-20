# Show, by example, that there exist states that have Schmidt Rank 3 and are in UNF 3 Eq.(12)

import numpy as np
import utils

np.set_printoptions(linewidth=120, suppress=True)


vec1 = np.zeros(16)
vec1[0*4 + 0] = 1
vec1[1*4 + 1] = 1
vec1[2*4 + 2] = 1
state1 = np.outer(vec1, vec1) / 3

vec2 = np.zeros(16)
vec2[2*4 + 3] = 1
vec2[3*4 + 2] = 1
state2 = np.outer(vec2, vec2) / 2

state = 0.5*state1 + 0.5*state2

unfaithful_3 = utils.is_D_UNF(state, 4, 4, 3)
print("State is 3-unfaithful: {}".format(unfaithful_3))

rank_over_2 = utils.is_schmidt_rank_greater_than_D(state, 4, 4, 2)
print("State has Schmidt rank > 2: {}".format(rank_over_2))
