import cma
import logging
from cma_cost_function import curvature_obj_fct
import numpy as np
import warnings

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

if __name__ == "__main__":
    starting_coefficients = 6 * [0]
    opts = {"maxfevals": 10000, "popsize": 40}
    n_jobs = 8
    es = cma.CMAEvolutionStrategy(starting_coefficients, 5.0, opts)
    es.optimize(objective_fct=curvature_obj_fct, iterations=250, n_jobs=n_jobs)
    optimized_muscle_torque_coefficients = es.result.xbest
    # Save the optimized coefficients to a file
    filename_data = "optimized_coefficients.txt"
    np.savetxt(filename_data, optimized_muscle_torque_coefficients, delimiter=",")
