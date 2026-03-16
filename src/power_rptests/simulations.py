import pandas as pd
import numpy as np

from power_rptests.data_generation import simulate

# ----------------------------------------------------------------------------------------------------------------------
# 1) DEFINE THE DETAILS OF THE SIMULATION
# ----------------------------------------------------------------------------------------------------------------------
andreoni_miller_budgets = [(120, 40), (40, 120), (120, 60), (60, 120), (150, 75),
                           (75, 150), (60, 60), (100, 100), (80, 80), (160, 40), (40, 160)]

andreoni_miller_budgets = [(10 * ms, 10 * mo) for ms, mo in andreoni_miller_budgets]

param_distributions = {
    "ces": {
        "alpha": lambda rng: rng.uniform(0.5, 1),
        "rho": lambda rng: rng.uniform(-0.5, 0.95),
    }
}

# ----------------------------------------------------------------------------------------------------------------------
# 2) SIMULATE
# ----------------------------------------------------------------------------------------------------------------------
df = simulate(
    budgets=andreoni_miller_budgets,
    utility_name="ces",
    param_distributions=param_distributions,
    n_samples=100,
    std=[5, 10, 15],
    seed=123
)
