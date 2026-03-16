# data_plotting.py
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
from .rp_tests.test_rev_pref import compute_index, bin_index

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
# Check whether cols are in df ('where' is only needed for the error message)
def _ensure_cols(df, cols, where=""):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = f"Missing columns{(' in ' + where) if where else ''}: {', '.join(missing)}"
        raise KeyError(msg)


# filter the rows/column if one of these parameter is specified
def _filter_df(
        df,
        ids=None,
        utility=None,
        alpha=None,
        rho=None,
        noise_type=None,
        noise_param=None,
        noise_free_disposal=None,
):
    out = df

    if ids is not None:
        if np.isscalar(ids):
            ids = [ids]
        out = out[out["id"].isin(ids)]

    if utility is not None:
        out = out[out["utility"] == utility]

    if alpha is not None:
        out = out[out["alpha"] == alpha]

    if rho is not None:
        out = out[out["rho"] == rho]

    if noise_type is not None:
        out = out[out["noise_type"] == noise_type]

    if noise_param is not None:
        out = out[out["noise_param"] == noise_param]

    if noise_free_disposal is not None:
        out = out[out["noise_free_disposal"] == noise_free_disposal]

    return out


def _budget_segment(row):
    """
    Downward budget line from (0, y_intercept) to (x_intercept, 0).
    """
    xs = [0.0, float(row["x_intercept"])]
    ys = [float(row["y_intercept"]), 0.0]
    return xs, ys


def _axis_limits(sub, xlim=None, ylim=None, pad=0.0):
    max_x = float(sub["x_intercept"].max())
    max_y = float(sub["y_intercept"].max())
    lim = max(max_x, max_y)
    if xlim is None:
        xlim = (0.0, lim + pad)
    if ylim is None:
        ylim = (0.0, lim + pad)
    return xlim, ylim


def _group_color_key(sub, color_by):
    """
    Build a categorical key used for coloring in plotly/matplotlib.
    color_by can be:
      - None
      - 'utility'
      - 'alpha'
      - 'rho'
      - 'noise_type'
      - 'noise_param'
      - ('alpha','rho')
      - ('utility','alpha','rho')
      - any list/tuple of existing columns
    """
    if color_by is None:
        return None

    if isinstance(color_by, str):
        cols = [color_by]
    else:
        cols = list(color_by)

    for c in cols:
        if c not in sub.columns:
            raise KeyError(f"color_by column '{c}' not in dataframe.")

    if len(cols) == 1:
        return sub[cols[0]].astype(str)

    key = sub[cols[0]].astype(str)
    for c in cols[1:]:
        key = key + ", " + c + "=" + sub[c].astype(str)
    return key


# --------------------------------------------------------------------------------------
# 1) plot(): budgets only / + optimum / + noisy
# --------------------------------------------------------------------------------------

def plot(
        df,
        what="budgets",
        ids=None,
        utility=None,
        alpha=None,
        rho=None,
        noise_type=None,
        noise_param=None,
        noise_free_disposal=None,
        color_by="utility",
        equal_axes=True,
        title=None,
        save=False,
        savepath=None,
        domain="social_preferences",
        figsize=(7, 7),
        xlim=None,
        ylim=None,
):
    """
    Plot budget constraints and (optionally) optimal / noisy points.
    Receives as input a pandas dataframe as outputted by "data_generation.py".
    """
    # "what" specifies what to plot: either only budgets, or budgets + optimum points, or budgets + optimum points
    # +noisy point
    if what not in ("budgets", "opt", "noisy"):
        raise ValueError("what must be one of: 'budgets', 'opt', 'noisy'")

    # Put in a list the required geometric element to create the plot
    required = ["x_intercept", "y_intercept"]
    if ids is not None:
        required += ["id"]
    if what in ("opt", "noisy"):
        if what == "opt":
            required += ["opt_x", "opt_y"]
        else:
            required += ["noisy_x", "noisy_y"]

    # Ensure that df has the required columns
    _ensure_cols(df, required, where="plot()")

    # Filter the df, create a filtered dataframe called "sub"
    sub = _filter_df(
        df,
        ids=ids,
        utility=utility,
        alpha=alpha,
        rho=rho,
        noise_type=noise_type,
        noise_param=noise_param,
        noise_free_disposal=noise_free_disposal,
    )
    if sub.empty:
        raise ValueError("No rows match the requested filters.")

    # Retrieve the axis limits
    xlim, ylim = _axis_limits(sub, xlim=xlim, ylim=ylim, pad=0.0)

    # Retrieve the budget sets to plot (dropping duplicates)
    budgets_df = sub.drop_duplicates(subset=["x_intercept", "y_intercept"])

    # color showing
    color_key = _group_color_key(sub, color_by)
    sub = sub.copy()
    if color_key is not None:
        sub["_color_key"] = color_key

    fig, ax = plt.subplots(figsize=figsize)

    # Plotting the budgets
    for _, row in budgets_df.iterrows():
        xs, ys = _budget_segment(row)
        ax.plot(xs, ys, color="lightgrey", linewidth=1, zorder=1)

    # Plotting the points
    def _scatter_points(xcol, ycol, label_prefix, zorder, marker=None):
        if xcol not in sub.columns or ycol not in sub.columns:
            return
        if color_key is None:
            ax.scatter(sub[xcol], sub[ycol], s=30, edgecolors="black", zorder=zorder, label=label_prefix)
            return

        groups = sub["_color_key"].unique().tolist()
        for g in groups:
            gg = sub[sub["_color_key"] == g]
            ax.scatter(
                gg[xcol], gg[ycol],
                s=30, edgecolors="black", zorder=zorder, label=f"{label_prefix}"
            )

    if what in ("opt", "noisy") and ("opt_x" in sub.columns) and ("opt_y" in sub.columns):
        _scatter_points("opt_x", "opt_y", "Optimal Choice", zorder=2)

    display_noise = ""
    if what == "noisy":
        if noise_type == "jittering":
            display_noise = "σ"
        # else if noise_type = ...
        #    display_noise = "λ"
        elif noise_type is None:
            if df["noise_type"].nunique(dropna=False) == 1:
                display_noise = df["noise_type"].iloc[0]
            else:
                raise ValueError("Column 'noiset_ype' does not contain a unique value. Not clear what noise type "
                                 "to plot")
        _scatter_points("noisy_x", "noisy_y", f"Noisy Choice ({display_noise}={noise_param})", zorder=3)

    # Set axis label depending on "domain". Default is "social_preferences"
    if domain == "social_preferences":
        x_label, y_label = "Self", "Other"
    elif domain == "risk_preferences":
        x_label, y_label = "Asset 1", "Asset 2"
    elif domain == "consumer_choice":
        x_label, y_label = "Bundle 1", "Bundle 2"
    else:
        raise ValueError("'domain' not recognised: either 'social_preferences' or 'risk_preferences' or "
                         "'consumer_choice'")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if equal_axes:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title or "Budgets / choices")

    if what != "budgets":
        ax.legend(fontsize=8)

    fig.tight_layout()

    # Save
    if save:
        if not savepath:
            raise ValueError("savepath must be provided when save=True")
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    return fig


# --------------------------------------------------------------------------------------
# 2) dashboard(): interactive filtering by id / showing opt / noisy
# --------------------------------------------------------------------------------------

def dashboard(df):
    """
    Jupyter dashboard:
      - filter by id (single) or show all (pooled)
      - choose what: budgets / opt / noisy
      - filter utility, alpha, rho
      - filter noise_type, noise_param, noise_free_disposal
      - choose coloring
    """
    _ensure_cols(df, ["x_intercept", "y_intercept"], where="dashboard()")

    work = df.copy()

    # widgets
    id_w = widgets.Dropdown(options=sorted(work["id"].dropna().unique().tolist()), description="id")
    all_w = widgets.Checkbox(value=False, description="All ids")
    what_w = widgets.ToggleButtons(options=["budgets", "opt", "noisy"], value="noisy", description="Show")
    color_w = widgets.Dropdown(
        options=[
            ("utility", "utility"),
            ("alpha", "alpha"),
            ("rho", "rho"),
            ("alpha,rho", ("alpha", "rho")),
            ("utility,alpha,rho", ("utility", "alpha", "rho")),
            ("noise_type", "noise_type"),
            ("noise_param", "noise_param"),
        ],
        value="utility",
        description="Color",
    )

    # optional filters (only if present, but your schema says they are)
    util_w = widgets.Dropdown(options=[None] + sorted(work["utility"].dropna().unique().tolist()), description="util")
    alpha_w = widgets.Dropdown(options=[None] + sorted(work["alpha"].dropna().unique().tolist()), description="α")
    rho_w = widgets.Dropdown(options=[None] + sorted(work["rho"].dropna().unique().tolist()), description="ρ")
    nt_w = widgets.Dropdown(options=[None] + sorted(work["noise_type"].dropna().unique().tolist()),
                            description="n_type")
    np_w = widgets.Dropdown(options=[None] + sorted(work["noise_param"].dropna().unique().tolist()),
                            description="n_par")
    fd_w = widgets.Dropdown(options=[None, True, False], value=None, description="free")

    out = widgets.Output()

    def _subset():
        sub = work
        if not all_w.value:
            sub = sub[sub["id"] == id_w.value]
        if util_w.value is not None:
            sub = sub[sub["utility"] == util_w.value]
        if alpha_w.value is not None:
            sub = sub[sub["alpha"] == alpha_w.value]
        if rho_w.value is not None:
            sub = sub[sub["rho"] == rho_w.value]
        if nt_w.value is not None:
            sub = sub[sub["noise_type"] == nt_w.value]
        if np_w.value is not None:
            sub = sub[sub["noise_param"] == np_w.value]
        if fd_w.value is not None:
            sub = sub[sub["noise_free_disposal"] == fd_w.value]
        return sub

    def _redraw(*_):
        out.clear_output(wait=True)
        sub = _subset()
        with out:
            if sub.empty:
                print("No data for this combination.")
                return
            ttl = "Dashboard"
            if not all_w.value:
                ttl += f" — id={id_w.value}"
            plot(
                sub,
                what=what_w.value,
                interactive=True,
                color_by=color_w.value,
                title=ttl,
            )

    for w in [id_w, all_w, what_w, color_w, util_w, alpha_w, rho_w, nt_w, np_w, fd_w]:
        w.observe(_redraw, names="value")

    ui = widgets.VBox([
        widgets.HBox([what_w, all_w, id_w]),
        widgets.HBox([util_w, alpha_w, rho_w]),
        widgets.HBox([nt_w, np_w, fd_w, color_w]),
        out
    ])
    display(ui)
    _redraw()


# --------------------------------------------------------------------------------------
# 3) plot_distribution_index(): distribution by noise_type / noise_param (bar or line) / rp test
# --------------------------------------------------------------------------------------

def plot_distribution_index(
        raw_data,
        index_type="CCEI",
        noise_type=None,
        noise_params=None,
        noise_free_disposal=None,
        group_by="noise_param",
        bin_width=0.05,
        title=None,
        save=False,
        savepath=None,
        figsize=(10, 5),
        palette=None
):
    """
    Plot distribution of an index for a given noise_type and one/more noise_param values.

    Assumes raw_data contains: id, x_intercept, y_intercept, opt_x/opt_y and/or noisy_x/noisy_y,
    plus noise_type, noise_param, noise_free_disposal.

    group_by defaults to 'noise_param' (common when you compare σ values within a fixed noise_type).
    """

    df = raw_data.copy()

    if noise_type is not None:
        df = df[df["noise_type"] == noise_type]

    if noise_params is not None:
        if np.isscalar(noise_params):
            noise_params = [noise_params]
        df = df[df["noise_param"].isin(list(noise_params))]

    if noise_free_disposal is not None:
        df = df[df["noise_free_disposal"] == noise_free_disposal]

    if df.empty:
        raise ValueError("No rows left after applying filters.")

    if palette is None:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    else:
        colors = palette

    idx = compute_index(df, index_type=index_type.upper())

    # reattach group_by if compute_index drops it
    if group_by not in idx.columns:
        if group_by in df.columns:
            gmap = df[["id", group_by]].drop_duplicates()
            idx = idx.merge(gmap, on="id", how="left")
        else:
            raise KeyError(f"'{group_by}' not found in raw_data and not produced by compute_index().")

    binned = bin_index(idx, index_type=index_type.upper(), group_by=group_by, bin_width=bin_width)

    fig, ax = plt.subplots(figsize=figsize)

    groups = binned[group_by].dropna().unique().tolist()
    try:
        groups = sorted(groups, key=lambda x: float(x), reverse=True)
    except Exception:
        groups = sorted(groups, key=lambda x: str(x), reverse=True)

    # Bar plot
    bins = binned["bin"].unique().tolist()
    x = np.arange(len(bins))
    width = 0.8 / max(1, len(groups))

    for j, g in enumerate(groups):
        color = colors[j % len(colors)]
        sub = binned[binned[group_by] == g].set_index("bin").reindex(bins)
        sub["fraction"] = sub["fraction"].fillna(0.0)
        ax.bar(
            x + (j - (len(groups) - 1) / 2) * width,
            sub["fraction"].values,
            width=width,
            label=str(g),
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    # Settings for the axes
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bins], rotation=45, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_xlabel(f"{index_type.upper()} score (binned)")
    ax.set_ylim(0, 1)

    # Settings for the labels
    legend_title = ""
    if group_by == "noise_param":
        legend_title = "Noise parameter"
    ax.legend(title=legend_title, loc="upper left", frameon=True)

    # Set title
    base = title or f"{index_type.upper()} distribution"
    if noise_type is not None:
        base += f" (noise_type={noise_type})"
    ax.set_title(base)

    fig.tight_layout()

    # Save
    if save:
        if not savepath:
            raise ValueError("savepath must be provided when save=True")

        p = Path(savepath)
        if p.suffix == "" and (str(savepath).endswith("/") or p.exists() and p.is_dir()):
            p.mkdir(parents=True, exist_ok=True)
            fname = (title or f"{index_type.upper()}_distribution").replace(" ", "_")
            p = p / f"{fname}.png"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(p, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    return fig

# --------------------------------------------------------------------------------------
# 4) plot_distribution_index(): compare two budget constraints
# --------------------------------------------------------------------------------------
def plot_classification_irrational(
        df_am,
        df_opt,
        *,
        index_type = "CCEI",
        threshold = 0.95,
        noise_param_col = "noise_param",
        budget_label_col = "budget_label",
        noise_level_labels = ("Very low", "Low", "Medium", "High"),
        legend_anchor = (1, 1.12),
        title = None,
        figsize = (10, 5),
        palette = None,
        save = False,
        savepath = None,

):
    required = ["id", "x_intercept", "y_intercept", noise_param_col, budget_label_col]
    _ensure_cols(df_am, required, where = "plot_classification_irrational(df_am)")
    _ensure_cols(df_opt, required, where= "plot_classification_irrational(df_opt)")

    # 1. Compute per-id CCEI within each noise_param
    idx_am = compute_index(df_am, index_type=index_type.upper())
    idx_opt = compute_index(df_opt, index_type=index_type.upper())

    # Re-attach the labels
    if budget_label_col not in idx_am.columns:
        idx_am = idx_am.merge(df_am[["id", budget_label_col]].drop_duplicates(), on="id", how="left")
    if budget_label_col not in idx_opt.columns:
        idx_opt = idx_opt.merge(df_opt[["id", budget_label_col]].drop_duplicates(), on="id", how="left")

    if noise_param_col not in idx_am.columns:
        idx_am = idx_am.merge(df_am[["id", noise_param_col]].drop_duplicates(), on="id", how="left")
    if noise_param_col not in idx_opt.columns:
        idx_opt = idx_opt.merge(df_opt[["id", noise_param_col]].drop_duplicates(), on="id", how="left")

    # 2. Classify
    score_col = index_type.upper()
    idx_am["_rational"] = (idx_am[score_col] >= threshold)
    idx_opt["_rational"] = (idx_opt[score_col] >= threshold)

    # 3. Aggregate shares (%)
    g_am = idx_am.groupby([noise_param_col, budget_label_col], observed=True)["_rational"].mean().reset_index()
    g_opt = idx_opt.groupby([noise_param_col, budget_label_col], observed=True)["_rational"].mean().reset_index()
    g_am["share_pct"] = 100.0 * g_am["_rational"]
    g_opt["share_pct"] = 100.0 * g_opt["_rational"]

    # 4. Combine + order noise levels
    comp = pd.concat([g_am, g_opt], ignore_index=True)

    # enforce x order by sorted numeric noise_param if possible
    noise_vals = comp[noise_param_col].dropna().unique().tolist()
    try:
        noise_vals = sorted(noise_vals, key=lambda x: float(x))
    except Exception:
        noise_vals = sorted(noise_vals, key=lambda x: str(x))

    # map to "very low/low/medium/high" by rank (first 4 only)
    label_map = {}
    for i, nv in enumerate(noise_vals):
        if i < len(noise_level_labels):
            label_map[nv] = noise_level_labels[i]
        else:
            label_map[nv] = str(nv)

    comp["_noise_label"] = comp[noise_param_col].map(label_map)
    xcats = [label_map[nv] for nv in noise_vals]

    # determine the two budget labels (one per dataset, expected)
    am_labels = df_am[budget_label_col].dropna().unique().tolist()
    opt_labels = df_opt[budget_label_col].dropna().unique().tolist()

    am_label = am_labels[0] if len(am_labels) else "Andreoni-Miller"
    opt_label = opt_labels[0] if len(opt_labels) else "Optimised"
    labs = [am_label, opt_label]

    # pivot to wide for plotting
    wide = comp.pivot_table(index="_noise_label", columns=budget_label_col, values="share_pct", aggfunc="mean")
    wide = wide.reindex(xcats)
    wide = wide.reindex(columns=labs)

    # Set the palette
    if palette is None:
        palette = {am_label: "#000000", opt_label: "#7f7f7f"}
    if isinstance(palette, (list, tuple)):
        palette = {labs[i]: palette[i] for i in range(min(len(labs), len(palette)))}

    fig, ax = plt.subplots(figsize=figsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    x = np.arange(len(wide.index))
    width = 0.8 / max(1, len(wide.columns))

    # Retrieve the noise type
    noise_type = df_am.noise_type[0]

    for j, col in enumerate(wide.columns):
        vals = wide[col].to_numpy()
        xpos = x + (j - (len(wide.columns) - 1) / 2) * width
        bars = ax.bar(xpos, vals, width=width, edgecolor="black", linewidth=0.5, label=str(col), color=palette.get(col, None))

        # annotate %
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.5,
                    f"{v:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(wide.index.tolist())
    ax.set_xlabel(f"Noise intensity ({noise_type})")
    ax.set_ylabel("Share classified as rational (%) \n (CCEI ≥ 0.95)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=legend_anchor)
    if title is not None:
        ax.set_title(title, pad = 20)
    fig.tight_layout()

    if save:
        if not savepath:
            raise ValueError("savepath must be provided when save=True")
        os.makedirs(os.path.dirname(savepath) or ".", exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
    else:
        plt.show()

    return fig