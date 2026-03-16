"""
Generates simulated data under different noise specification.

Pipeline:
1) Draw subject-level parameters.
2) Compute optimal bundles (exact or numeric) by maximising utility subject to budget constraint
3) Add noise under a certain noise specification
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from itertools import product


# ------------------------------------------------------------------------------------------
# 1) Budget constraints and utility function
# ------------------------------------------------------------------------------------------
def budget_y(x_intercept, y_intercept, x):
    """Budget line written as y = y_int - (y_int/x_int)*x."""
    return y_intercept - (y_intercept / x_intercept) * x


def u_ces(x, y, alpha, rho):
    """CES utility."""
    return (alpha * (x ** rho) + (1.0 - alpha) * (y ** rho)) ** (1.0 / rho)


def solve_exact_ces(x_intercept, y_intercept, alpha, rho, rho_eps=1e-8):
    """
    Closed-form interior demand for CES.
    Handles rho ≈ 0 via Cobb–Douglas limit.
    """
    p = x_intercept / y_intercept
    m = x_intercept

    # Cobb-Douglas limit
    if abs(rho) < rho_eps:
        opt_x = alpha * m
        opt_y = (m - opt_x) / p
        return opt_x, opt_y

    # Compute numerator and denominator
    sigma = 1.0 / (1.0 - rho)
    num = alpha ** sigma
    den = num + (1.0 - alpha) ** sigma * p ** (1.0 - sigma)

    # Return the point of optimum
    opt_x = m * num / den
    opt_y = (m - opt_x) / p
    return opt_x, opt_y


"""
Define a utility registry as a dictionary of dictionaries. Each entry is a utility function with entries: (1) 
"params", (2) "u" and (3) "solve exact"
"""
UTILITY_REGISTRY = {
    "ces": {
        "params": ("alpha", "rho"),
        "u": u_ces,
        "solve_exact": solve_exact_ces,
    },

    # Add other utility function here
}


# ------------------------------------------------------------------------------------------
# 2) Generate a population of individuals with sampled parameters
# ------------------------------------------------------------------------------------------

def generate_population(budgets, utility_name, param_distributions, n_samples=50, rng=None):
    """
    Construct long dataframe with one row per (subject, budget).
    Inputs:
    - budgets: either a list with iterables of (x_intercept, y_intercept), or a dict with keys:
        {"budgets": [(x_1, y_1),...], "label": str}
    - utility_name
    - param_distribution
    - n_samples
    - rng
    """
    rng = np.random.default_rng() if rng is None else rng

    if utility_name not in UTILITY_REGISTRY:
        raise ValueError("Unknown utility")

    # If 'budgets' is a dictionary, first check that it contains the key 'budgets', then retrieve the label and the
    # budgets themselves
    budget_label = None
    if isinstance(budgets, dict):
        if "budgets" not in budgets:
            raise KeyError("If budgets is a dict, it must contain a key 'budgets'.")
        budget_label = budgets.get("label", None)
        budgets = budgets["budgets"]

    # Retrieve all the x and y intercepts and the number of budgets
    budgets = list(budgets)
    x_ints = np.array([b[0] for b in budgets], dtype=float)
    y_ints = np.array([b[1] for b in budgets], dtype=float)
    B = len(budgets)

    # Create an id for each subject
    ids = np.repeat(np.arange(1, n_samples + 1), B)
    x_rep = np.tile(x_ints, n_samples)
    y_rep = np.tile(y_ints, n_samples)

    # Blueprint for the dataframe
    out = {"id": ids, "x_intercept": x_rep, "y_intercept": y_rep, "utility": np.repeat(utility_name, n_samples * B),
           "budget_label": np.repeat(budget_label, n_samples * B)}

    # Retrieve the parameters
    dist = param_distributions[utility_name]
    for pname in UTILITY_REGISTRY[utility_name]["params"]:
        draws = np.array([dist[pname](rng) for _ in range(n_samples)], dtype=float)
        out[pname] = np.repeat(draws, B)

    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------
# 3) Optimisation
# ------------------------------------------------------------------------------------------

def optimise(df, maximiser="exact"):
    """
    Compute optimal bundles for each row.
    """
    out = df.copy()
    n = len(out)

    opt_x = np.empty(n)
    opt_y = np.empty(n)

    # Exact maximiser (use if a closed-form solution exists)
    if maximiser == "exact":
        for i, row in enumerate(out.itertuples(index=False)):
            if row.utility == "ces":
                opt_x[i], opt_y[i] = solve_exact_ces(
                    row.x_intercept,
                    row.y_intercept,
                    row.alpha,
                    row.rho,
                )
            else:
                raise ValueError("Exact solver not implemented for this utility.")
    # SLSQP maximiser (if a closed form solution does not exist)
    elif maximiser == "slsqp":

        for i, row in enumerate(out.itertuples(index=False)):
            reg = UTILITY_REGISTRY[row.utility]
            u = reg["u"]
            params = {p: getattr(row, p) for p in reg["params"]}

            xI = row.x_intercept
            yI = row.y_intercept

            def neg_u(z):
                return -u(z[0], z[1], **params)

            bounds = [(0.0, xI), (0.0, yI)]

            def bc(x):
                return budget_y(xI, yI, x)

            cons = ({"type": "ineq", "fun": lambda z: bc(z[0]) - z[1]},)

            x0 = np.array([0.5 * xI, 0.5 * yI])

            res = minimize(
                neg_u,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
                options={"ftol": 1e-6, "maxiter": 2000, "disp": False},
            )

            if not res.success:
                raise RuntimeError(res.message)

            opt_x[i] = res.x[0]
            opt_y[i] = res.x[1]

    else:
        raise ValueError("maximiser must be 'exact' or 'slsqp'")

    out["opt_x"] = opt_x
    out["opt_y"] = opt_y
    return out


# ------------------------------------------------------------------------------------------
# 4) Noise specification
# ------------------------------------------------------------------------------------------
# Helper function that converts iterables to list
def _as_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    return [x]

def add_noise_jittering(df, std=5.0, free_disposal=True, rng=None):
    """
    Adds jittering noise to the point of optimum. The jittered point is either along the budget constraint, or in the semi-
    circle around the optimum and within the feasible region. Prevent points from falling outside the feasible region.
    """

    # ---------- 1. define geometry parameters ---------------------------------------------------------------------
    rng = np.random.default_rng() if rng is None else rng
    out = df.copy()
    n = len(out)
    noisy_x = np.empty(n)
    noisy_y = np.empty(n)

    # Go over the rows of the dataframe received as input
    for i, row in enumerate(out.itertuples(index=False)):

        # Retrieve the point of optimum, the x and y intercept of the budget constraint
        p0 = np.array([row.opt_x, row.opt_y])
        xI = row.x_intercept
        yI = row.y_intercept

        # Compute the relative price
        p2 = xI / yI

        # Compute the unit-vector orthogonal to the budget constraint and pointing inward
        n_in = np.array([-1.0, -p2])
        n_in /= np.linalg.norm(n_in)

        # ---------- 2. choose the jittering direction -------------------------------------------------------------
        # If free disposal, sample uniformly at random an angle between -90 and 90 wrt to the orthogonal direction
        if free_disposal:
            angle_deg = rng.uniform(-90, 90)
        # If no free disposal, choose at random either -90 or +90 degrees wrt to the orthogonal direction
        else:
            angle_deg = rng.choice([-90, 90])

        # Convert the angle to radiants, compute the rotation matrix, multiply it by the orthogonal vector to compute
        # the direction "d" used for jittering
        theta = np.deg2rad(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        d = R @ n_in

        # ---------- 3. prevent points from falling outside the feasible region ----------------------------------
        # The jittering direction is stored in the two dimensional array d, where the first component represents the
        # horizontal direction and the second the vertical one. We check whether the direction of jittering intersects
        # the x axis or the y axis first. We then compute the maximum distance allowable so that the jittered point
        # does not jump outside the feasible region. If the jittered point is outside the feasible region, cap the jittering
        # to the maximum distance.
        # If the direction vector points to the left, solve the equation p0[0]+t*d[0]=0, i.e. t = -p0[0]/d[0]. In other
        # words, we measure how many times the x component of the direction vector needs to be repeated in order to reach
        # the y axis
        candidates = []
        if d[0] < 0:
            candidates.append(-p0[0] / d[0])
        # Same applies if the direction vector points downward. In this case, compute t = -p0[1]/d[1], i.e. the number
        # of times the y component of the direction vector needs to be "repeated" in order to reach the x axis
        if d[1] < 0:
            candidates.append(-p0[1] / d[1])

        # Take the maximum of these two distances
        t_max = max(0.0, min(candidates)) if candidates else np.inf

        # ---------- 4. Displace the points ------------------------------------------------------------------------
        # Draw a random number from a (half) normal distribution (only positive values)
        length = abs(rng.normal(scale=std))
        length = min(length, t_max)

        # Displace p0 (ensuring it is non-negative)
        noisy = np.maximum(p0 + length * d, 0.0)
        noisy_x[i] = noisy[0]
        noisy_y[i] = noisy[1]

    out["noisy_x"] = noisy_x
    out["noisy_y"] = noisy_y

    # Stack all standard deviation results
    return out

def add_noise_misperception(df, price_noise_sd=0.05, rng=None):
    """
    Perturb the slope of the budget constraint (i.e. price) and recompute the perceived optimum
    :param price_noise_sd:
    :return:
    """
    rng = np.random.default_rng() if rng is None else rng
    out = df.copy()
    n = len(df)

    noisy_x = np.empty(n)
    noisy_y = np.empty(n)

    for i, row in enumerate(out.itertuples(index=False)):
        if getattr(row, "utility", "ces") != "ces":
            raise ValueError("misperception currently implemented only for CES + exact optimisation")

        xI = float(row.x_intercept)
        yI = float(row.y_intercept)
        alpha = float(row.alpha)
        rho = float(row.rho)

        p_true = xI/yI
        p_perc = p_true * np.exp(rng.normal(scale=price_noise_sd))

        yI_perc = xI/p_perc
        opt_x, opt_y = solve_exact_ces(xI, yI_perc, alpha, rho)

        y_cap = yI - (yI/xI)*opt_x
        opt_y = min(opt_y, y_cap)
        opt_x = max(opt_x, 0.0)
        opt_y = max(opt_y, 0.0)

        noisy_x[i], noisy_y[i] = opt_x, opt_y

    out["noisy_x"] = noisy_x
    out["noisy_y"] = noisy_y
    return out


def add_noise_lapses(df, lapse_prob=0.5, rng=None):
    """
    With probability "lapse_prob" choose a random feasible point.

    :param df: input dataframe
    :param lapse_prob: probability with wich a random feasible point is picked.
    :param rng: random number generator (for reproducibility)
    :return: the original dataframe with additional columns: noisy_x, noisy_t, noise_type and noise_param
    """
    rng = np.random.default_rng() if rng is None else rng
    out = df.copy()
    n = len(out)

    # Create two empty arrays
    noisy_x = np.empty(n)
    noisy_y = np.empty(n)

    # Iterate over the rows of the dataframe
    for i, row in enumerate(out.itertuples(index=False)):
        xI = row.x_intercept
        yI = row.y_intercept

        # With probability "lapse_prob" choose a random point on the budget set
        if rng.random() < lapse_prob:
            x = rng.uniform(0.0, xI)
            # TODO: check - one random draw is enough? Retrieve then the y from the x
            y = rng.uniform(0.0, yI - (yI/xI) * x)
            noisy_x[i], noisy_y[i] = x,y
        else:
            noisy_x[i], noisy_y[i] = row.opt_x, row.opt_y

    out["noisy_x"] = noisy_x
    out["noisy_y"] = noisy_y
    return out


def add_noise_quantal_response(df, lambda_qr=0.1, grid_size=51, rng=None):
    """
    Adds noise through quantal response:
        1. Build a grid in the feasible region according to the parameter "grid size"
        2. Assign probability of being sampled proportional to exp(lambda * U(x,y))
        3. Sample one point for this distribution

    :param df: input dataframe
    :param lambda_qr: noise/rationality parameter
    :param grid_size: horizontal and vertical size of the grid used to compute the utility. The grid will have
                        grid_size**2 cells
    :param rng: for reproducibility
    :return: the original dataframe with additional columns: noisy_x, noisy_y, noise_type and noise_param
    """
    rng = np.random.default_rng() if rng is None else rng
    out = df.copy()

    # Number of subjects
    n = len(out)

    # Prepare two vectors for the noisy points
    noisy_x = np.empty(n)
    noisy_y = np.empty(n)

    # Go over each row of the dataframe
    for i, row in enumerate(out.itertuples(index=False)):
        # Retrieve the utility function and parameters: row of the df --> utility registry --> corresponding function
        util_name = getattr(row, "utility", "ces")
        # This returns a dictionary with all the entries corresponding to the utility function specified
        reg = UTILITY_REGISTRY[util_name]
        # Retrieve the utility function
        u = reg["u"]
        # Retrieve the parameters. Example: {'alpha': 0.53} {'rho': -0.11}
        params = {p: getattr(row, p) for p in reg["params"]}
        # Retrieve the intercepts
        xI = float(row.x_intercept)
        yI = float(row.y_intercept)

        # Create two empty lists: one for storing the points on the budget constraints, the other for storing the
        # corresponding utility values
        points = []
        utils = []

        # Create a linear space over the x (from 0 to the x-intercept) for a given grid size
        xs = np.linspace(0.0, xI, grid_size)

        # Create an horizontal and a vertical linear space, that in combination form a grid. For each point in the grid
        # compute the utility and append it to the list
        for x in xs:
            y_max = budget_y(xI, yI, x)
            ys = np.linspace(0.0, y_max, grid_size)
            for y in ys:
                points.append((x, y))
                utils.append(u(x, y, **params))

        utils = np.array(utils, dtype=float)
        utils = utils - np.max(utils)

        # Create the sampling distribution
        w = np.exp(lambda_qr * utils)
        w = w / w.sum()

        # Sample
        j = rng.choice(len(points), p=w)
        noisy_x[i], noisy_y[i] = points[j]

    # Add the relevant columns
    out["noisy_x"] = noisy_x
    out["noisy_y"] = noisy_y
    return out


# Tells which are the params associated with each noise type and which is the function that implements that noise type
NOISE_REGISTRY = {
    "jittering": {
        "params": ("std", "free_disposal"),
        "fn": add_noise_jittering,
    },
    "lapses": {
        "params": ("lapse_prob",),
        "fn": add_noise_lapses,
    },
    "misperception": {
        "params": ("price_noise_sd",),
        "fn": add_noise_misperception,
    },
    "quantal_response": {
        "params": ("lambda_qr", "grid_size"),
        "fn": add_noise_quantal_response,
    },
}


def apply_noise(df, noise_type, rng=None, **noise_kwargs):
    """
    Apply a noise model to the dataframe.

    Convention:
    - the first parameter listed in NOISE_REGISTRY[noise_type]["params"] is stored in the standardised column
        'noise_param' (this is common across noise types)
    - any additional parameters are stored in columns named 'noise_<parameter_name>' (this is the case for example for
        the jittering parameter "free_disposal")
    """
    if noise_type not in NOISE_REGISTRY:
        raise ValueError(f"Unknown noise_type '{noise_type}'")

    # Retrive the noise type, the noise function and the expected parameters
    entry = NOISE_REGISTRY[noise_type]
    fn = entry["fn"]
    expected_params = entry["params"]

    # Check whether there are unexpected parameters specified by the users but not present in the noise registry
    unexpected = set(noise_kwargs) - set(expected_params)
    if unexpected:
        raise ValueError(
            f"Unexpected parameter(s) for noise '{noise_type}': {sorted(unexpected)}. "
            f"Expected: {expected_params}. "
            "Please check the user inputs or the 'NOISE_REGISTRY'"
        )

    # Check for missing parameters
    missing = [p for p in expected_params if p not in noise_kwargs]
    if missing:
        raise ValueError(
            f"Missing parameter(s) for noise '{noise_type}': {missing}"
        )

    # Convert all parameters to lists
    param_lists = {p: _as_list(noise_kwargs[p]) for p in expected_params}

    frames = []

    # Create a Cartesian product of all the noise parameters. Then loop over them
    for values in product(*(param_lists[p] for p in expected_params)):
        param_dict = dict(zip(expected_params, values))

        # Apply the noise (add column "noisy_x", "noisy_y")
        out = fn(df, rng=rng, **param_dict).copy()

        # Add the two columns: "noise_type" and "noise_param"
        out["noise_type"] = noise_type
        out["noise_param"] = param_dict[expected_params[0]]

        # Add additional columns for additional parameters
        for p in expected_params[1:]:
            out[f"noise_{p}"] = param_dict[p]

        frames.append(out)

    return pd.concat(frames, ignore_index=True)

# ------------------------------------------------------------------------------------------
# 5) Full simulation
# ------------------------------------------------------------------------------------------
def simulate(budgets,
             utility_name,
             param_distributions,
             noise_type=None,
             n_samples=50,
             maximiser="exact",
             seed=None,
             **noise_kwargs):
    """
    Full simulation pipeline:

    1) Draw parameters
    2) Compute optimal bundles
    3) Apply noise model (if specified)
    """

    rng = np.random.default_rng(seed)

    if noise_type is None and noise_kwargs:
        raise ValueError("noise_kwargs were provided but noise_type is None")

    # 1) Generate population
    df = generate_population(
        budgets,
        utility_name,
        param_distributions,
        n_samples=n_samples,
        rng=rng
    )

    # 2) Optimise
    df = optimise(df, maximiser=maximiser)

    # 3) Apply noise (optional)
    if noise_type is not None:
        df = apply_noise(
            df,
            noise_type=noise_type,
            rng=rng,
            **noise_kwargs
        )

    return df
