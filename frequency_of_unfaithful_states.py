# Code to generate the data for Table 1 in paper

import datetime
import numpy as np
import utils

np.set_printoptions(suppress=True)


def frequency_of_states_PPT_and_UNF(metric):

    ds = [2, 3, 4, 5]
    num_PPT = []  # number of PPT states for each dimension
    num_eUNF = []  # number of states in UNF but NOT PPT for each dimension

    repeats = 10**2

    for d in ds:
        print("looking at d={} ({})".format(d, datetime.datetime.now()))
        num_PPT.append(0)
        num_eUNF.append(0)

        for _ in range(repeats):
            state = metric(d**2)
            if utils.is_PPT(state, d, d):
                num_PPT[-1] += 1
            else:
                # reducible => UNF, much faster to just check that first
                if utils.is_reducible(state, d, d):
                    num_eUNF[-1] += 1
                elif utils.is_UNF(state, d, d):
                    num_eUNF[-1] += 1

    print("Finished {}, with {} repeats ({})".format(metric.__name__, repeats,
                                                     datetime.datetime.now()))
    print("ds:", ds)
    print("num_PPT:", num_PPT)
    print("num_eUNF:", num_eUNF)


def frequency_of_states_schmid_rank_D_or_lower(D, ds, repeats, metric):

    num_low_rank = []
    for d in ds:
        print("looking at d={} ({})".format(d, datetime.datetime.now()))
        num_low_rank.append(0)

        for _ in range(repeats):
            state = metric(d**2)
            if not utils.is_schmidt_rank_greater_than_D(state, d, d, D):
                num_low_rank[-1] += 1

    print("Finished {}, with {} repeats ({})".format(metric.__name__, repeats,
                                                     datetime.datetime.now()))
    print("ds:", ds)
    print("num_low_rank:", num_low_rank)


# First two columns of Table 1
frequency_of_states_PPT_and_UNF(utils.random_HS_state)
# last two columns of Table 1
frequency_of_states_PPT_and_UNF(utils.random_bures_state)

# Discussion immediately below table 1
frequency_of_states_schmid_rank_D_or_lower(2, [3], 1000, utils.random_HS_state)
frequency_of_states_schmid_rank_D_or_lower(2, [4], 1000, utils.random_real_state)
frequency_of_states_schmid_rank_D_or_lower(2, [3], 1000, utils.random_bures_state)
