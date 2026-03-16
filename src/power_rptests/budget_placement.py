from dataclasses import dataclass
import numpy as np
import pandas as pd
from data_generation import UTILITY_REGISTRY, optimise, apply_noise

# --------------------------------------------------------------------------------------
# Budget utils
# --------------------------------------------------------------------------------------
def _budget_slope(x_intercept, y_intercept):
    """"Return the slope of the budget constraint"""
    return float(y_intercept)/(x_intercept)

def _intersection_x(b1, b2, atol=1e-12):
    """
    x-coordinates of the intersection of two downward sloping budgets (return None if numerically parellel)
    """
    # Retrieve the two budget constraints and their slopes
    x1, y1 = b1
    x2, y2 = b2
    s1 = _budget_slope(x1, y1)
    s2 = _budget_slope(x2, y2)
    denom = (s2 - s1)
    # If the difference between the slopes of the budget constraints is too close to 0, return None
    if np.isclose(denom, 0.0, atol=atol):
        return None
    # Solve y1 - s1*x = y2 - s_2*x
    return (y2-y1)/denom

def _intersection_in_overlap(xA, b1, b2):
    """Require intersection to lie within the overlapping x-range of the two budgets."""
    x1, _ = b1
    x2, _ = b2
    return (xA is not None) and (0.0 < xA < min(x1, x2))

# --------------------------------------------------------------------------------------
# Utils for the population
# --------------------------------------------------------------------------------------
def draw_subject_params(utility_name, param_distributions, n_subjects, rng):
    """
    Draw only once the subject-level parameters (shared across all budgets).
    Returns a DataFrame with one row per id and columns: ['id', 'utility', <param1>, <param2>, ...]
    """
    if utility_name not in UTILITY_REGISTRY:
        raise ValueError(f"Unkown utility: {utility_name}")
    if utility_name not in param_distributions:
        raise ValueError(f"param_distribution missing entry for '{utility_name}'")

    params = UTILITY_REGISTRY[utility_name]["params"]
    dist = param_distributions[utility_name]
    for p in params:
        if p not in dist:
            raise ValueError(f"param_distributions['{utility_name}'] missing parameter '{p}'")
    ids = np.arange(1, n_subjects+1, dytpe = int)
    out = {"id": ids, "utility": np.repeat(utility_name, n_subjects)}

    for p in params:
        out[p] = np.array([dist[p](rng) for _ in range(n_subjects)], dtype=float)

    return pd.DataFrame(out)

def expand_subjects_over_budgets(subjects, budgets):
    """
    Cartesian product of subject rows with budgets, producing one row per (id, budget).

    Adds:
        x_intercept, y_intercept
    """
    budgets = list(budgets)
    B = len(budgets)
    if B == 0:
        raise ValueError("budgets cannot be empty")

    x_ints = np.array([b[0] for b in budgets], dtype=float)
    y_ints = np.array([b[1] for b in budgets], dtype=float)

    n = len(subjects)

    df = pd.DataFrame({
        "id": np.repeat(subjects["id"].to_numpy(), B),
        "x_intercept": np.tile(x_ints, n),
        "y_intercept": np.tile(y_ints, n),
        "utility": np.repeat(subjects["utility"].to_numpy(), B),
    })

    # Copy parameter columns
    for c in subjects.columns:
        if c in ("id", "utility"):
            continue
        df[c] = np.repeat(subjects[c].to_numpy(), B)

    return df


# --------------------------------------------------------------------------------------
# Prices -> budgets
# --------------------------------------------------------------------------------------

def budget_from_price_income(price, income):
    """
    Convert (price, income) to intercept-form budget.

    Convention:
        x_intercept = income
        y_intercept = income / price
    """
    if price <= 0:
        raise ValueError("price must be > 0")
    if income <= 0:
        raise ValueError("income must be > 0")
    return float(income), float(income) / float(price)


# --------------------------------------------------------------------------------------
# Power metric (baseline): count unique "crossing" violators
# --------------------------------------------------------------------------------------

def violators_against_previous(previous_budgets, previous_noisy, candidate_budget, candidate_noisy):
    """
    Return IDs that "cross" an intersection point xA between the candidate budget and
    any previous budget.

    Crossing criterion (symmetric):
        (x_prev - xA) and (x_cand - xA) have opposite signs.

    Requires:
      candidate_noisy and each previous_noisy[j] contain columns: ['id','noisy_x'].
    """
    violators = set()

    if len(previous_budgets) != len(previous_noisy):
        raise ValueError("previous_budgets and previous_noisy must have the same length")

    for b_j, noisy_j in zip(previous_budgets, previous_noisy):
        xA = _intersection_x(b_j, candidate_budget)
        if xA is None:
            continue
        if not _intersection_in_overlap(xA, b_j, candidate_budget):
            continue

        merged = noisy_j[["id", "noisy_x"]].merge(
            candidate_noisy[["id", "noisy_x"]],
            on="id",
            suffixes=("_prev", "_cand"),
            how="inner",
        )

        x_prev = merged["noisy_x_prev"].to_numpy()
        x_cand = merged["noisy_x_cand"].to_numpy()

        cross = (x_prev - xA) * (x_cand - xA) < 0.0
        if np.any(cross):
            ids = merged.loc[cross, "id"].astype(int).tolist()
            violators.update(ids)

    return violators


# --------------------------------------------------------------------------------------
# Greedy optimiser
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GreedyPlacementResult:
    prices: list
    incomes: list
    budgets: list
    optima: list
    noisy_optima: list
    violator_counts: list


def greedy_place_incomes(
    prices,
    first_income,
    income_grid = np.linspace(5, 50, 40),
    *,
    utility_name,
    param_distributions,
    n_subjects=200,
    maximiser="exact",
    noise_type=None,
    seed=123,
    **noise_kwargs,
):
    """
    Greedy algorithm that places the incomes given a set of prices:
      1) Fix budget 1 at (price_1, first_income).
      2) For t=2,...,T: choose income_t in income_grid that maximises the number of unique violators vs all previous budgets.
      3) Return the set of incomes.

    To ensure reproducibility:
    - Subject parameters are drawn once using `seed`;
    - For candidate evaluation at step t, candidate noise is generated using a step-specific seed so that comparisons
        across incomes use the same RNG stream.
    """
    # Check consistency of the inputs
    prices = list(prices)
    if len(prices) == 0:
        raise ValueError("prices cannot be empty")

    if first_income <= 0:
        raise ValueError("first_income must be > 0")

    income_grid = list(income_grid)
    if len(income_grid) == 0 and len(prices) > 1:
        raise ValueError("income_grid cannot be empty if more than one price is provided")

    # RNG for subject parameter draws (fixed across everything)
    rng_subjects = np.random.default_rng(seed)

    # Draw once parameters for each subject
    subjects = draw_subject_params(
        utility_name=utility_name,
        param_distributions=param_distributions,
        n_subjects=n_subjects,
        rng=rng_subjects,
    )

    # Prepare a list for what we need
    placed_incomes = []
    placed_budgets = []
    optima_list = []
    noisy_list = []
    viol_counts = []

    # First budget is set: find optima on this budget
    inc1 = float(first_income)
    b1 = budget_from_price_income(prices[0], inc1)
    df1 = expand_subjects_over_budgets(subjects, [b1])
    df1 = optimise(df1, maximiser=maximiser)

    # Compute the noisy points
    if noise_type is not None:
        rng_noise_1 = np.random.default_rng(None if seed is None else seed + 10_000)
        df1n = apply_noise(df1, noise_type=noise_type, rng=rng_noise_1, **noise_kwargs)
    else:
        df1n = df1.copy()

    # Append
    placed_incomes.append(inc1)
    placed_budgets.append(b1)
    optima_list.append(df1)
    noisy_list.append(df1n)
    viol_counts.append(0)

    # Greedy algorithm for subsequent prices
    for t in range(1, len(prices)):
        price_t = prices[t]

        best_income = None
        best_cnt = -1
        best_opt = None
        best_noisy = None

        # Step-specific seed so all candidates share same noise RNG stream
        base_step_seed = None if seed is None else (seed + 20_000 + 1_000 * t)

        # Go over each income in the income grid
        for income_t in income_grid:
            income_t = float(income_t)
            if income_t <= 0:
                continue

            b_t = budget_from_price_income(price_t, income_t)

            # Optimise candidate budget for all subjects
            df_t = expand_subjects_over_budgets(subjects, [b_t])
            df_t = optimise(df_t, maximiser=maximiser)

            # Noise: reset RNG per candidate -> fair comparisons across incomes
            if noise_type is not None:
                rng_noise_t = np.random.default_rng(base_step_seed)
                df_tn = apply_noise(df_t, noise_type=noise_type, rng=rng_noise_t, **noise_kwargs)
            else:
                df_tn = df_t.copy()

            viol = violators_against_previous(
                previous_budgets=placed_budgets,
                previous_noisy=noisy_list,
                candidate_budget=b_t,
                candidate_noisy=df_tn,
            )
            cnt = len(viol)

            if cnt > best_cnt:
                best_cnt = cnt
                best_income = income_t
                best_opt = df_t
                best_noisy = df_tn

        if best_income is None or best_opt is None or best_noisy is None:
            raise RuntimeError(f"No valid income found at step t={t+1}. Check income_grid / constraints.")

        placed_incomes.append(best_income)
        placed_budgets.append(budget_from_price_income(price_t, best_income))
        optima_list.append(best_opt)
        noisy_list.append(best_noisy)
        viol_counts.append(best_cnt)

    return GreedyPlacementResult(
        prices=prices,
        incomes=placed_incomes,
        budgets=placed_budgets,
        optima=optima_list,
        noisy_optima=noisy_list,
        violator_counts=viol_counts,
    )


__all__ = [
    "GreedyPlacementResult",
    "budget_from_price_income",
    "greedy_place_incomes",
    "violators_against_previous",
]