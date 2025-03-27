import numpy as np
from itertools import combinations
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import linear_sum_assignment
from time import time

# Observed and theoretical transitions
observed = np.array([28947.80952381, 31794.76190476, 40011.71428572, 42057.42857143, 47802.47619048])
theoretical = np.array([3728.48, 4862.691, 4960.295, 4996.25375, 6564.632, 9533.2, 10833, 12821.576, 16447.9555, 18756.096])


# Residual computation function
def compute_residual(Z, observed_subset, theoretical):
    adjusted_observed = observed_subset / (1 + Z)
    cost_matrix = np.abs(adjusted_observed[:, None] - theoretical[None, :]) / theoretical[None, :]

    # Find the best matching subset using linear sum assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    residual = np.sum(cost_matrix[row_ind, col_ind])

    return residual


# Wrapper to optimize Z and extract best matching subset
def optimize_redshift_and_subset_with_exclusions(observed, theoretical, max_exclusions=1):
    best_Z = None
    best_residual = float('inf')
    best_matching_subset = None
    best_observed_subset = None

    # Generate all subsets of observed values with up to max_exclusions exclusions
    for k in range(len(observed) - max_exclusions, len(observed) + 1):
        for observed_subset in combinations(observed, k):
            observed_subset = np.array(observed_subset)

            # Optimize Z using differential evolution
            result = differential_evolution(
                lambda Z: compute_residual(Z, observed_subset, theoretical),
                bounds=[(0, 10)],
                strategy='best1bin',
                popsize=15,
                tol=0.0001,
            )
            Z = result.x[0]
            residual = compute_residual(Z, observed_subset, theoretical)

            # Check if this is the best fit
            if residual < best_residual:
                best_residual = residual
                best_Z = Z

                # Recompute cost matrix to get the best matching subset
                adjusted_observed = observed_subset / (1 + Z)
                cost_matrix = np.abs(adjusted_observed[:, None] - theoretical[None, :]) / theoretical[None, :]
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                best_matching_subset = theoretical[col_ind]
                best_observed_subset = observed_subset

    return best_Z, best_matching_subset, best_observed_subset


def get_max_exclusions(observed, exclusion_fraction=0.3, min_line_number=2):

    total_lines = len(observed)

    if exclusion_fraction is None or exclusion_fraction == 0:
        num_exclusions = 0
    else:
        if total_lines > min_line_number:
            num_exclusions = max(1, np.floor(total_lines * exclusion_fraction))
        else:
            num_exclusions = 0

    return num_exclusions

ex_frac, min_n_lines = 0.3, 3
obs_arr = np.array([1,2,3,4])
print(get_max_exclusions(obs_arr, exclusion_fraction=ex_frac, min_line_number=min_n_lines))

obs_arr = np.array([1,2,3])
print(get_max_exclusions(obs_arr, exclusion_fraction=ex_frac, min_line_number=min_n_lines))

obs_arr = np.array([1,2])
print(get_max_exclusions(obs_arr, exclusion_fraction=ex_frac, min_line_number=min_n_lines))



# Run the optimization
start_time = time()
best_Z, best_matching_subset, best_observed_subset = optimize_redshift_and_subset_with_exclusions(observed, theoretical, max_exclusions=2)
end_time = np.round(time() - start_time, 4)

print(f'Computation time: {end_time}')
print("Best-fit Z:", best_Z)
print("Best matching subset of R:", best_matching_subset)
print("Best subset of O:", best_observed_subset)

# import numpy as np
# from scipy.optimize import minimize
# from scipy.optimize import linear_sum_assignment
#
# # Example observed and theoretical transitions
# # observed = np.array([400, 450, 500])  # Observed transitions (O)
# # theoretical = np.array([380, 430, 490, 520])  # Theoretical transitions (R)
#
# observed = np.array([28947.80952381, 31794.76190476, 40011.71428572, 42057.42857143, 47802.47619048])
# theoretical = np.array([ 3728.48,  4862.691,  4960.295,  4996.25375, 6564.632, 9533.2, 10833, 12821.576,  16447.9555, 18756.096  ])
#
#
# # Residual computation function
# def compute_residual(Z, observed, theoretical):
#     """
#     Computes the residual for a given redshift Z.
#
#     Parameters:
#     - Z: Redshift value.
#     - observed: Observed transitions (array-like).
#     - theoretical: Theoretical transitions (array-like).
#
#     Returns:
#     - Residual value (float).
#     """
#     adjusted_observed = observed / (1 + Z)
#     cost_matrix = np.abs(adjusted_observed[:, None] - theoretical[None, :])
#
#     # Find the best matching subset using linear sum assignment
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     residual = np.sum(cost_matrix[row_ind, col_ind])
#
#     return residual
#
#
# # Wrapper to optimize Z and extract best matching subset
# def optimize_redshift_and_subset(observed, theoretical):
#     """
#     Optimizes the redshift Z and identifies the best matching subset of theoretical transitions.
#
#     Parameters:
#     - observed: Observed transitions (array-like).
#     - theoretical: Theoretical transitions (array-like).
#
#     Returns:
#     - best_Z: Best-fit redshift (float).
#     - best_matching_subset: Best matching subset of theoretical transitions (array).
#     """
#     # Minimize residual to find the best Z
#     result = minimize(lambda Z: compute_residual(Z, observed, theoretical),
#                       x0=[5], bounds=[(0, 10)])
#     best_Z = result.x[0]
#
#     # Recompute cost matrix and find the best matching subset
#     adjusted_observed = observed / (1 + best_Z)
#     cost_matrix = np.abs(adjusted_observed[:, None] - theoretical[None, :])
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#
#     # Extract the best matching subset of theoretical transitions
#     best_matching_subset = theoretical[col_ind]
#
#     return best_Z, best_matching_subset
#
#
# # Run the optimization
# best_Z, best_matching_subset = optimize_redshift_and_subset(observed, theoretical)
#
# print("Best-fit Z:", best_Z)
# print("Best matching subset of R:", best_matching_subset)

# import numpy as np
# from scipy.optimize import minimize
# from scipy.optimize import linear_sum_assignment
#
# # Example observed and theoretical transitions
# observed = np.array([400, 450, 500])  # Observed transitions (O)
# theoretical = np.array([380, 430, 490, 520])  # Theoretical transitions (R)
#
#
# # Objective function to minimize
# def objective_function(Z, observed, theoretical):
#     adjusted_observed = observed / (1 + Z)
#     cost_matrix = np.abs(adjusted_observed[:, None] - theoretical[None, :])
#
#     # Find the best matching subset using linear sum assignment
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     residual = np.sum(cost_matrix[row_ind, col_ind])
#
#     return residual
#
#
# # Wrapper to optimize Z and extract best matching subset
# def optimize_redshift_and_subset(observed, theoretical):
#
#     def residual(Z):
#         return objective_function(Z, observed, theoretical)
#
#     # Optimize Z
#     result = minimize(residual, x0=[0.1], bounds=[(0, 10)])
#     best_Z = result.x[0]
#
#     # Recompute cost matrix and find the best matching subset
#     adjusted_observed = observed / (1 + best_Z)
#     cost_matrix = np.abs(adjusted_observed[:, None] - theoretical[None, :])
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#
#     # Extract the best matching subset of theoretical transitions
#     best_matching_subset = theoretical[col_ind]
#
#     return best_Z, best_matching_subset
#
#
# # Run the optimization
# best_Z, best_matching_subset = optimize_redshift_and_subset(observed, theoretical)
#
# print("Best-fit Z:", best_Z)
# print("Best matching subset of R:", best_matching_subset)
