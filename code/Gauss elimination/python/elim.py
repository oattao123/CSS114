import os
import sympy as sp
import numpy as np


def matrix_representation(system, syms):
    a, b = sp.linear_eq_to_matrix(system, syms)
    return np.asarray(a.col_insert(len(syms), b), dtype=np.float32)



def upper_triangular(M):
    M = np.concatenate((M[np.any(M != 0, axis=1)], M[np.all(M == 0, axis=1)]), axis=0)
    for i in range(0, M.shape[0]):
        j = 1
        pivot = M[i][i]
        while pivot == 0 and i + j < M.shape[0]:
            M[[i, i + j]] = M[[i + j, i]]
            j += 1
            pivot = M[i][i]
        if pivot == 0:
            return M
        row = M[i]
        M[i] = row / pivot

        for j in range(i + 1, M.shape[0]):
            M[j] = M[j] - M[i] * M[j][i]

    return M


def backsubstitution(M, syms):
    for i, row in reversed(list(enumerate(M))): 
        eqn = -M[i][-1]
        for j in range(len(syms)):
            eqn += syms[j] * row[j]

        syms[i] = sp.solve(eqn, syms[i])[0]

    return syms


def validate_solution(system, solutions, tolerance=1e-6):
    for eqn in system:
        assert eqn.subs(solutions) < tolerance


def linalg_solve(system, syms):
    M, c = sp.linear_eq_to_matrix(system, syms)
    M, c = np.asarray(M, dtype=np.float32), np.asarray(c, dtype=np.float32)

    return np.linalg.solve(M, c)




