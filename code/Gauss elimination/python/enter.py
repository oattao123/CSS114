from elim import matrix_representation, upper_triangular, backsubstitution
import numpy as np
import sympy as sp

x1, x2, x3 = sp.symbols('x1 x2 x3')
symbolic_vars = [x1, x2, x3]

equations = [x1 + 2 * x2 + 3 * x3 - 1, 2 * x1 + 3 * x2 + 4 * x3 - 2, 3 * x1 + 4 * x2 + 5 * x3 - 3]
[print(eq) for eq in equations]
augmented_matrix = matrix_representation(system=equations, syms=symbolic_vars)
print('\naugmented matrix:\n', augmented_matrix)

upper_triangular_matrix = upper_triangular(augmented_matrix)
print('\nupper triangular matrix:\n', upper_triangular_matrix)

backsub_matrix = upper_triangular_matrix[np.any(upper_triangular_matrix != 0, axis=1)]
numeric_solution = np.array([0., 0., 0.])

if backsub_matrix.shape[0] != len(symbolic_vars):
    print('dependent system. infinite number of solutions')
elif not np.any(backsub_matrix[-1][:len(symbolic_vars)]):
    print('inconsistent system. no solution..')
else:
    numeric_solution = backsubstitution(backsub_matrix, symbolic_vars)
    print(f'\nsolutions:\n{numeric_solution}')
