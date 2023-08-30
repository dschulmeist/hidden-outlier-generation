import random
from itertools import chain, combinations

import numpy as np
from scipy.stats import norm


def random_unif_on_sphere(number, dimensions, r, random_state=5):
    normal_deviates = norm.rvs(size=(number, dimensions), random_state=random_state)
    radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
    points = normal_deviates / radius
    return points * r


def gen_powerset(dims):
    return set(chain.from_iterable(combinations(range(dims), r) for r in range(1, dims)))


def subspace_grab(indices, data):
    return data[:, np.array(indices)]


def gen_rand_subspaces(dims, upper_limit, include_all_attr=True, seed=5):
    rd = random.Random(seed)
    features = list(range(0, dims))
    rd.shuffle(features, )
    subspaces = set()
    # includes every attribute singleton and for every attribute a random subspace with more than 1 feature
    # containing it
    if include_all_attr:
        for i in features:
            r = rd.randint(2, dims - 1)
            fts = rd.sample(range(dims), r)
            fts.append(i)
            subspace1 = tuple(fts)
            subspace2 = tuple([i])
            if subspace1 not in subspaces:  # ensure it's a new subspace
                subspaces.add(subspace1)
            if subspace2 not in subspaces:
                subspaces.add(subspace2)

    # avoid sampling singletons, because they are already included
    if include_all_attr:
        lower_limit = 2
    else:
        lower_limit = 1

    while len(subspaces) < (2 ** upper_limit) - 2:
        r = rd.randint(lower_limit, dims - 1)
        random_comb = tuple(rd.sample(range(dims), r))
        if random_comb not in subspaces:
            subspaces.add(random_comb)
    return subspaces


list0 = gen_powerset(4)
print(list0)

list1 = gen_rand_subspaces(4, 4, include_all_attr=True, seed=5)

print(list1)
