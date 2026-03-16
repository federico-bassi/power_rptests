# rp_tests/test_rev_pref.py
import numpy as np
import pandas as pd
import revpref as rp


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _pick_choice_columns(df: pd.DataFrame, choice: str = "auto") -> tuple[str, str]:
    """
    Decide which bundle columns to use as quantities:
      - "noisy":  use (noisy_x, noisy_y)
      - "opt":    use (opt_x, opt_y)
      - "auto":   prefer noisy if present & not all-NaN, else opt
    """
    choice = choice.lower()
    if choice not in {"auto", "noisy", "opt"}:
        raise ValueError("choice must be one of: 'auto', 'noisy', 'opt'")

    has_noisy = {"noisy_x", "noisy_y"}.issubset(df.columns) and not df[["noisy_x", "noisy_y"]].isna().all().all()
    has_opt = {"opt_x", "opt_y"}.issubset(df.columns) and not df[["opt_x", "opt_y"]].isna().all().all()

    if choice == "noisy":
        if not {"noisy_x", "noisy_y"}.issubset(df.columns):
            raise KeyError("Requested choice='noisy' but df lacks columns noisy_x/noisy_y.")
        return "noisy_x", "noisy_y"

    if choice == "opt":
        if not {"opt_x", "opt_y"}.issubset(df.columns):
            raise KeyError("Requested choice='opt' but df lacks columns opt_x/opt_y.")
        return "opt_x", "opt_y"

    # auto
    if has_noisy:
        return "noisy_x", "noisy_y"
    if has_opt:
        return "opt_x", "opt_y"

    raise KeyError("Could not find usable choice columns: need noisy_x/noisy_y or opt_x/opt_y.")


def _generate_arrays(data: pd.DataFrame, *, choice: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """
    Return price and quantity arrays for *revpref*.

    Prices are normalised so that p_x = 1 and p_y = x_intercept / y_intercept
    (consistent with budgets: x + (xI/yI) y <= xI).
    """
    # Check whether the columns "x_intercept" and "y_intercept" are there in the dataset
    required = {"x_intercept", "y_intercept"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing columns for prices: {sorted(missing)}")

    # Retrieve the columns "noisy_x" and "noisy_y" if noise is present
    xcol, ycol = _pick_choice_columns(data, choice=choice)

    # Create the price vector of length "len(data)" with first entry always 1 and second entry the relative price
    p = np.column_stack((np.ones(len(data), dtype=float),
                         (data["x_intercept"].to_numpy(dtype=float) / data["y_intercept"].to_numpy(dtype=float)),))
    # Create the quantity vector with "noisy_x" as first entry and "noisy_y" as second entry
    q = np.column_stack((data[xcol].to_numpy(dtype=float), data[ycol].to_numpy(dtype=float),))
    return p, q


def _grouping_columns(df: pd.DataFrame) -> list[str]:
    """
    Identifies the columns on which to do the grouping. Returns a list with the columns "id" (required) and -if present-
    other columns that might be informative: "noise_type", "noise_param" and "noise_free_disposal"
    """
    if "id" not in df.columns:
        raise KeyError("Data must contain an 'id' column.")

    cols = ["id"]
    for c in ["noise_type", "noise_param", "noise_free_disposal"]:
        if c in df.columns:
            # include if at least one non-null value exists
            if not df[c].isna().all():
                cols.append(c)
    return cols


# ---------------------------------------------------------------------
# Index calculation
# ---------------------------------------------------------------------
def compute_index(
    data: pd.DataFrame,
    index_type: str = "CCEI",
    simple_check: bool = False,
    *,
    choice: str = "auto",
) -> pd.DataFrame:
    """
    Compute the chosen revealed-preference statistic for each group.

    Inputs:
    data : pd.DataFrame
        Must contain:
          - id
          - x_intercept, y_intercept
          - either (noisy_x,noisy_y) and/or (opt_x,opt_y)
        Optionally contains:
          - noise_type, noise_param, noise_free_disposal
    index_type : str
        Either "CCEI" (default) or "HMI". Ignored if simple_check=True.
    simple_check : bool - Simply checks how many individual satisfy GARP
        If True, compute pref.check_garp() and return it under column "GARP".
    choice : {"auto","noisy","opt"}
        Which bundle columns to treat as observed choices.

    Returns: a dataframe with, for each id and for each noise type and noise level (value of "noise_param"),
            returns the corresponding index value.
    """
    df = data.copy()
    gcols = _grouping_columns(df)

    results = []

    # Group by id and, if present, noise columns. We obtain a dataframe with individual-level data for a given noise
    # specification and noise parameter
    for keys, grp in df.groupby(gcols, observed=True):
        # keys can be scalar if gcols=["id"]
        if not isinstance(keys, tuple):
            keys = (keys,)

        # Generate price and quantity vectors
        p, q = _generate_arrays(grp, choice=choice)

        # Compute the "preference" object (from Revealed Preference package)
        pref = rp.RevealedPreference(p, q)

        # If simple_check is true, simply check whether the individual satisfies GARP. If it is false
        if simple_check:
            value = bool(pref.check_garp())
            out_col = "GARP"
        else:
            out_col = index_type.upper()
            try:
                if out_col == "HMI":
                    value = pref.hmi()
                else:
                    value = pref.ccei()
            except Exception:
                value = np.nan

        # Associates to each column (used to form groups) the corresponding value (key), then adds one column (GARP) and
        # associates to it the result of the GARP test or the value of the CCEI
        row = dict(zip(gcols, keys))
        row[out_col] = value
        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------
# Binning utilities
# ---------------------------------------------------------------------
def bin_index(
    data: pd.DataFrame,
    *,
    index_type: str = "CCEI",
    group_by: str = "noise_param",
    bin_width: float = 0.05,
) -> pd.DataFrame:
    """
    Bin an index (e.g. CCEI, HMI) and compute the share of observations
    in each bin within every level of group_by.

    Returns a DataFrame with columns ["bin", group_by, "fraction"].
    """
    df = data.copy()
    index_type = index_type.upper()

    # backward-compat alias: "noise" -> "noise_param"
    if group_by == "noise" and "noise" not in df.columns and "noise_param" in df.columns:
        group_by = "noise_param"

    if index_type not in df.columns:
        raise KeyError(f"'{index_type}' not found in supplied DataFrame.")
    if group_by not in df.columns:
        raise KeyError(f"'{group_by}' not found in supplied DataFrame.")

    edges = np.arange(0, 1.0 + bin_width + 1e-9, bin_width)
    labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(len(edges) - 2)] + ["1"]

    df["bin"] = pd.cut(
        df[index_type],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    df["bin"] = (
        pd.Categorical(df["bin"], categories=labels, ordered=True)
        .remove_unused_categories()
    )

    # nicer ordering for numeric-ish group_by
    try:
        cats = sorted(df[group_by].dropna().unique(), key=lambda x: float(x))
    except Exception:
        cats = sorted(df[group_by].dropna().unique(), key=lambda x: str(x))
    df[group_by] = pd.Categorical(df[group_by], categories=cats, ordered=True)

    binned = (
        df.groupby(["bin", group_by], observed=True)
        .size()
        .reset_index(name="count")
    )
    totals = df.groupby(group_by, observed=True).size()
    binned["fraction"] = binned.apply(lambda r: r["count"] / totals.loc[r[group_by]], axis=1)

    return binned
