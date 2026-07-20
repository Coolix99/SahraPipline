from __future__ import annotations

import json
import shutil
from pathlib import Path
import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bebi103.viz import corner, predictive_regression
from bokeh.io import output_file, save as bokeh_save
from cmdstanpy import CmdStanModel
from scipy.integrate import solve_ivp
from scipy.stats import mannwhitneyu
from utilsScalar import scalar_path

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.family"] = "DejaVu Sans"

SCRIPT_DIR = Path(__file__).resolve().parent
STAN_DIR = SCRIPT_DIR / "stan"
STAN_PATHS = {
    "wt": STAN_DIR / "area_ode_wt.stan",
    "smoc_dev": STAN_DIR / "area_ode_smoc_dev.stan",
    "smoc_reg": STAN_DIR / "area_ode_smoc_reg.stan",
}
STAN_FUNCTIONS = STAN_DIR / "area_ode_functions.stanfunctions"
RESULTS_ROOT = Path(scalar_path) / "area_ode_staged_results"

COLORS = {
    "Development": "#2077b5",
    "Regeneration": "#f57e1f",
    "4850cut": "#5b3a2a",
    "7230cut": "#d8b07a",
    "smoc_dev": "#7b5ea7",
    "smoc_reg": "#df3066",
}

CONDITION_SETS = {
    "all": None,
    "dev_reg": ["Development", "Regeneration"],
    "dev_4850": ["Development", "4850cut"],
    "reg_4850": ["Regeneration", "4850cut"],
    "dev_7230": ["Development", "7230cut"],
    "dev_smocdev": ["Development", "smoc_dev"],
    "reg_smocreg": ["Regeneration", "smoc_reg"],
    "smocdev_smocreg": ["smoc_dev", "smoc_reg"],
    "all_no_smoc": ["Development", "Regeneration", "4850cut", "7230cut"],
    "dev": ["Development"],
    "reg": ["Regeneration"],
    "cut_4850": ["4850cut"],
    "cut_7230": ["7230cut"],
    "smoc_dev_only": ["smoc_dev"],
    "smoc_reg_only": ["smoc_reg"],
}

STAN_KEYS = {
    "Development": "Dev",
    "Regeneration": "Reg",
    "4850cut": "4850cut",
    "7230cut": "7230cut",
    "smoc_dev": "smoc_dev",
    "smoc_reg": "smoc_reg",
}

CONDITION_STAGE = {
    "Development": "wt",
    "Regeneration": "wt",
    "4850cut": "wt",
    "7230cut": "wt",
    "smoc_dev": "smoc_dev",
    "smoc_reg": "smoc_reg",
}

T0 = {
    "Development": 47.999,
    "Regeneration": 47.999,
    "4850cut": 47.999,
    "7230cut": 71.999,
    "smoc_dev": 47.999,
    "smoc_reg": 47.999,
}

STEADY_INITIAL_GROWTH = {"Development", "smoc_dev"}
WT_CONDITIONS = ["Development", "Regeneration", "4850cut", "7230cut"]


def normalize_is_control(value):
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False
    value = str(value).strip().lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False
    raise ValueError(f"Cannot interpret Is control value: {value!r}")


def normalize_smoc_condition(value):
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"smoc12_dev", "smoc_dev", "development", "dev"}:
        return "smoc_dev"
    if value in {"smoc12_reg", "smoc_reg", "regeneration", "reg"}:
        return "smoc_reg"
    raise ValueError(f"Unknown SMOC condition value: {value!r}")


def assign_smoc_conditions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "condition" not in df.columns:
        df["condition"] = pd.NA
    if "Is control" not in df.columns:
        df["Is control"] = pd.NA

    resolved = []
    for index, row in df.iterrows():
        from_name = normalize_smoc_condition(row["condition"])
        is_control = normalize_is_control(row["Is control"])
        from_control = "smoc_dev" if is_control is True else "smoc_reg" if is_control is False else None
        if from_name is not None and from_control is not None and from_name != from_control:
            raise ValueError(
                "Inconsistent SMOC condition information:\n"
                f"  row: {index}\n"
                f"  Mask Folder: {row.get('Mask Folder', '<unknown>')}\n"
                f"  condition: {row['condition']!r} -> {from_name}\n"
                f"  Is control: {row['Is control']!r} -> {from_control}"
            )
        condition = from_name if from_name is not None else from_control
        if condition is None:
            raise ValueError(
                "Cannot determine SMOC condition:\n"
                f"  row: {index}\n"
                f"  Mask Folder: {row.get('Mask Folder', '<unknown>')}\n"
                f"  condition: {row['condition']!r}\n"
                f"  Is control: {row['Is control']!r}"
            )
        resolved.append(condition)
    df["condition"] = resolved
    return df


def get_data() -> pd.DataFrame:
    wt_path = Path(scalar_path) / "WT_scalars.csv"
    smoc_path = Path(scalar_path) / "Smoc12_scalars.csv"
    df_wt = pd.read_csv(wt_path)
    required_wt = {"time in hpf", "condition", "Surface Area"}
    missing_wt = required_wt.difference(df_wt.columns)
    if missing_wt:
        raise KeyError(f"WT_scalars.csv is missing columns: {sorted(missing_wt)}")
    df_wt = df_wt.loc[
        df_wt["condition"].isin(WT_CONDITIONS),
        ["time in hpf", "condition", "Surface Area"],
    ].copy()

    df_smoc = pd.read_csv(smoc_path)
    required_smoc = {"time in hpf", "Surface Area"}
    missing_smoc = required_smoc.difference(df_smoc.columns)
    if missing_smoc:
        raise KeyError(f"Smoc12_scalars.csv is missing columns: {sorted(missing_smoc)}")
    df_smoc = assign_smoc_conditions(df_smoc)
    df_smoc = df_smoc[["time in hpf", "condition", "Surface Area"]].copy()

    df = pd.concat([df_wt, df_smoc], ignore_index=True)
    df["time in hpf"] = pd.to_numeric(df["time in hpf"], errors="raise")
    df["Surface Area"] = pd.to_numeric(df["Surface Area"], errors="raise") / 10000.0
    if df[["time in hpf", "Surface Area"]].isna().any().any():
        raise ValueError("The input data contain missing time or surface-area values.")

    missing_conditions = set(STAN_KEYS).difference(df["condition"].unique())
    if missing_conditions:
        raise ValueError(f"No observations found for conditions: {sorted(missing_conditions)}")
    print("\nLoaded conditions:")
    print(df.groupby("condition").size())
    return df


def ode_system(t, y, alpha, beta_, A_end):
    A, g = y
    return [g * A, -alpha * (g - beta_ * (A_end - A) / A_end)]


def save_figure(fig, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), dpi=300)
    fig.savefig(base_path.with_suffix(".pdf"))
    fig.savefig(base_path.with_suffix(".svg"))
    plt.close(fig)


def explorative_plotting(df: pd.DataFrame, conditions_to_plot=None):
    conditions = list(df["condition"].unique()) if conditions_to_plot is None else list(conditions_to_plot)
    data = df[df["condition"].isin(conditions)]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    all_ns = {condition: [] for condition in conditions}

    for condition in conditions:
        sub = data[data["condition"] == condition]
        stats = sub.groupby("time in hpf")["Surface Area"].agg(["mean", "std", "count"]).reset_index()
        t = stats["time in hpf"].to_numpy()
        mean = stats["mean"].to_numpy()
        std = stats["std"].fillna(0).to_numpy()
        counts = stats["count"].to_numpy()
        for time, count in zip(t, counts):
            all_ns[condition].append(count)
            print(f"{condition} | t={time}: n={count}")
        ax.fill_between(t, mean - std, mean + std, color=COLORS[condition], alpha=0.2, linewidth=0)
        ax.plot(t, mean, color=COLORS[condition], linewidth=2, zorder=2)
        ax.scatter(
            t, mean, color=COLORS[condition], edgecolor="white", linewidth=1.2,
            s=120, zorder=3, label=condition,
        )

    if len(conditions) == 2:
        cond1, cond2 = conditions
        df1 = data[data["condition"] == cond1]
        df2 = data[data["condition"] == cond2]
        common_times = sorted(set(df1["time in hpf"]).intersection(df2["time in hpf"]))

        def p_to_stars(p):
            if p <= 0.001:
                return "***"
            if p <= 0.01:
                return "**"
            if p <= 0.05:
                return "*"
            return ""

        for time in common_times:
            values1 = df1[df1["time in hpf"] == time]["Surface Area"].to_numpy()
            values2 = df2[df2["time in hpf"] == time]["Surface Area"].to_numpy()
            if len(values1) == 0 or len(values2) == 0:
                continue
            _, p_value = mannwhitneyu(values1, values2, alternative="two-sided")
            stars = p_to_stars(p_value)
            if stars:
                y = max(np.mean(values1), np.mean(values2))
                for index, _ in enumerate(stars):
                    ax.text(
                        time, y + 0.05 * np.ptp(ax.get_ylim()) + index * 0.04, "*",
                        ha="center", va="bottom", fontsize=18, color="black",
                    )

    xticks = [48, 60, 72, 84, 96, 108, 120, 132, 144]
    ax.set_xlabel("Developmental time [hpf]", fontsize=20)
    ax.set_ylabel("Surface Area", fontsize=20)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(xticks[0], xticks[-1])
    plt.subplots_adjust(left=0.15, right=0.97, bottom=0.2025, top=0.979)
    print("\n--- Total sample sizes ---")
    for condition in conditions:
        print(f"{condition}: total n = {sum(all_ns[condition])}")
    ax.legend(frameon=False, fontsize=14)
    return fig


def plot_prior_predictive(df: pd.DataFrame, condition: str, n_samples=300, seed=42):
    rng = np.random.default_rng(seed)
    sub = df[df["condition"] == condition]
    t_start = 72.0 if condition == "7230cut" else 48.0
    t_plot = np.linspace(t_start, df["time in hpf"].max(), 200)
    trajectories = []
    for _ in range(n_samples):
        alpha_tilde = rng.normal(-0.5, 0.5)
        beta_tilde = rng.normal(0, 0.1)
        A_end_tilde = rng.normal(1.0, 0.1)
        A_0_tilde = rng.normal(0.75 if condition == "7230cut" else 0.3, 0.15)
        alpha = 10.0 ** alpha_tilde
        beta_ = (alpha / 4.0) * 10.0 ** beta_tilde
        A_end = 10.0 ** A_end_tilde
        A_0 = 10.0 ** A_0_tilde
        g_0 = beta_ * (A_end - A_0) / A_end if condition == "Development" else 0.0
        solution = solve_ivp(
            ode_system, [T0[condition], t_plot[-1]], [A_0, g_0], t_eval=t_plot,
            args=(alpha, beta_, A_end), method="RK45",
        )
        if solution.success and np.all(np.isfinite(solution.y[0])):
            trajectories.append(solution.y[0])
    if not trajectories:
        raise RuntimeError(f"No valid prior trajectories were generated for {condition}.")

    trajectories = np.asarray(trajectories)
    lower, median, upper = np.quantile(trajectories, [0.05, 0.5, 0.95], axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for trajectory in trajectories[:50]:
        ax.plot(t_plot, trajectory, color="gray", alpha=0.08, linewidth=1)
    ax.fill_between(t_plot, lower, upper, color=COLORS[condition], alpha=0.18, linewidth=0)
    ax.plot(t_plot, median, color=COLORS[condition], linewidth=2.5, label="prior median")

    stats = sub.groupby("time in hpf")["Surface Area"].agg(["mean", "std"]).reset_index()
    std = stats["std"].fillna(0).to_numpy()
    ax.fill_between(
        stats["time in hpf"], stats["mean"] - std, stats["mean"] + std,
        color=COLORS[condition], alpha=0.2, linewidth=0,
    )
    ax.scatter(
        stats["time in hpf"], stats["mean"], color=COLORS[condition], edgecolor="white",
        linewidth=1.2, s=120, zorder=3, label="observed mean",
    )
    xticks = [48, 60, 72, 84, 96, 108, 120, 132, 144]
    ax.set_xlim(45, 150)
    ax.set_ylim(0, 15)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, fontsize=16)
    ax.set_yticks(np.arange(0, 15, 2))
    ax.set_yticklabels(np.arange(0, 15, 2), fontsize=16)
    ax.set_xlabel("Developmental time [hpf]", fontsize=20)
    ax.set_ylabel("Surface Area", fontsize=20)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(45, xticks[-1])
    ax.legend(frameon=False, fontsize=12)
    plt.subplots_adjust(left=0.15, right=0.97, bottom=0.2025, top=0.979)
    return fig

def condition_data(df: pd.DataFrame, condition: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["condition"] == condition].sort_values("time in hpf")
    times = sub["time in hpf"].astype(float).to_numpy()
    areas = sub["Surface Area"].astype(float).to_numpy()
    if len(times) == 0:
        raise ValueError(f"No observations available for {condition}.")
    if np.any(np.diff(times) < 0):
        raise ValueError(f"Times are not sorted for {condition}.")
    if np.any(times <= T0[condition]):
        raise ValueError(f"All {condition} times must be greater than {T0[condition]}.")
    return times, areas


def prepare_wt_stan_data(df: pd.DataFrame, n_ppc_48=100, n_ppc_72=100) -> dict:
    data = {}
    for condition in WT_CONDITIONS:
        key = STAN_KEYS[condition]
        times, areas = condition_data(df, condition)
        data[f"N_{key}"] = len(times)
        data[f"t_{key}"] = times
        data[f"A_{key}"] = areas
    t_max = float(df["time in hpf"].max())
    data["N_ppc_48"] = n_ppc_48
    data["t_ppc_48"] = np.linspace(48.0, t_max, n_ppc_48)
    data["N_ppc_72"] = n_ppc_72
    data["t_ppc_72"] = np.linspace(72.0, t_max, n_ppc_72)
    return data


def prepare_smoc_stan_data(
    df: pd.DataFrame, condition: str, fixed_parameters: dict[str, float], n_ppc=100,
) -> dict:
    if condition not in {"smoc_dev", "smoc_reg"}:
        raise ValueError(f"Unknown SMOC condition: {condition}")
    times, areas = condition_data(df, condition)
    data = {
        f"N_{condition}": len(times),
        f"t_{condition}": times,
        f"A_{condition}": areas,
        "alpha_fixed": float(fixed_parameters["alpha"]),
        "beta_fixed": float(fixed_parameters["beta_"]),
        "N_ppc_48": n_ppc,
        "t_ppc_48": np.linspace(48.0, float(df["time in hpf"].max()), n_ppc),
    }
    if condition == "smoc_reg":
        data["A_end_fixed"] = float(fixed_parameters["A_end"])
    return data


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def compile_models() -> dict[str, CmdStanModel]:
    missing = [path for path in [*STAN_PATHS.values(), STAN_FUNCTIONS] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stan files: {[str(path) for path in missing]}")
    options = {"include-paths": [str(STAN_DIR)]}
    return {
        stage: CmdStanModel(stan_file=str(path), stanc_options=options)
        for stage, path in STAN_PATHS.items()
    }


def fit_stan_stage(
    model: CmdStanModel, data: dict, stage_dir: Path, seed=12345,
    chains=4, iter_warmup=3000, iter_sampling=1000,
):
    stage_dir.mkdir(parents=True, exist_ok=True)
    with (stage_dir / "stan_data.json").open("w", encoding="utf-8") as handle:
        json.dump({key: json_ready(value) for key, value in data.items()}, handle, indent=2)
    cmdstan_dir = stage_dir / "cmdstan"
    cmdstan_dir.mkdir(exist_ok=True)
    fit = model.sample(
        data=data, chains=chains, parallel_chains=chains, iter_warmup=iter_warmup,
        iter_sampling=iter_sampling, seed=seed, output_dir=str(cmdstan_dir),
        show_progress=True, show_console=False,
    )
    fit.summary().to_csv(stage_dir / "cmdstan_summary.csv")
    try:
        diagnosis = fit.diagnose()
    except RuntimeError as error:
        diagnosis = f"CmdStan diagnose failed: {error}"
    (stage_dir / "diagnose.txt").write_text(diagnosis, encoding="utf-8")
    return fit


def stage_specs(stage_name: str):
    if stage_name == "wt":
        return {
            "label": "WT",
            "ppc": ["A_Dev_ppc", "A_Reg_ppc", "A_4850cut_ppc", "A_7230cut_ppc"],
            "output": [
                "alpha", "beta_", "A_end", "A_0_Dev", "A_0_Reg", "A_0_4850cut",
                "A_0_7230cut", "g_0_Dev", "sigma_Dev", "sigma_rel_Dev", "sigma_Reg",
                "sigma_rel_Reg", "sigma_4850cut", "sigma_rel_4850cut", "sigma_7230cut",
                "sigma_rel_7230cut",
            ],
            "derived": {"g_0_Dev"},
            "fixed": set(),
            "corners": {
                "Dev": ["A_end", "A_0_Dev", "alpha", "beta_"],
                "Reg": ["A_end", "A_0_Reg", "alpha", "beta_"],
                "4850cut": ["A_end", "A_0_4850cut", "alpha", "beta_"],
                "7230cut": ["A_end", "A_0_7230cut", "alpha", "beta_"],
            },
            "regression": {
                "Dev": ("A_Dev_ppc", "t_ppc_48", "t_Dev", "A_Dev"),
                "Reg": ("A_Reg_ppc", "t_ppc_48", "t_Reg", "A_Reg"),
                "4850cut": ("A_4850cut_ppc", "t_ppc_48", "t_4850cut", "A_4850cut"),
                "7230cut": ("A_7230cut_ppc", "t_ppc_72", "t_7230cut", "A_7230cut"),
            },
        }
    if stage_name == "smoc_dev":
        return {
            "label": "SMOC development",
            "ppc": ["A_smoc_dev_ppc"],
            "output": [
                "alpha", "beta_", "A_end_smoc_dev", "A_0_smoc_dev", "g_0_smoc_dev",
                "sigma_smoc_dev", "sigma_rel_smoc_dev",
            ],
            "derived": {"g_0_smoc_dev"},
            "fixed": {"alpha", "beta_"},
            "corners": {
                "smoc_dev": [
                    "A_end_smoc_dev", "A_0_smoc_dev", "sigma_smoc_dev", "sigma_rel_smoc_dev"
                ]
            },
            "regression": {
                "smoc_dev": ("A_smoc_dev_ppc", "t_ppc_48", "t_smoc_dev", "A_smoc_dev")
            },
        }
    if stage_name == "smoc_reg":
        return {
            "label": "SMOC regeneration",
            "ppc": ["A_smoc_reg_ppc"],
            "output": [
                "alpha", "beta_", "A_end", "A_0_smoc_reg", "sigma_smoc_reg",
                "sigma_rel_smoc_reg",
            ],
            "derived": set(),
            "fixed": {"alpha", "beta_", "A_end"},
            "corners": {
                "smoc_reg": ["A_0_smoc_reg", "sigma_smoc_reg", "sigma_rel_smoc_reg"]
            },
            "regression": {
                "smoc_reg": ("A_smoc_reg_ppc", "t_ppc_48", "t_smoc_reg", "A_smoc_reg")
            },
        }
    raise ValueError(f"Unknown stage: {stage_name}")


def make_posterior_summary(
    posterior_df: pd.DataFrame, stage_name: str,
    fixed_from_wt: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    specs = stage_specs(stage_name)
    fixed_from_wt = fixed_from_wt or {}
    rows = []
    for parameter in specs["output"]:
        if parameter in specs["fixed"]:
            if parameter not in fixed_from_wt:
                raise KeyError(f"Missing WT summary for fixed parameter {parameter}")
            stats = fixed_from_wt[parameter]
            rows.append({
                "parameter": parameter,
                "mean": float(stats["mean"]),
                "std": float(stats["std"]),
                "status": "fixed from WT posterior mean",
                "source": "WT posterior",
                "fixed_value": float(stats["mean"]),
            })
            continue
        status = "derived" if parameter in specs["derived"] else "fitted"
        rows.append({
            "parameter": parameter,
            "mean": float(posterior_df[parameter].mean()),
            "std": float(posterior_df[parameter].std()),
            "status": status,
            "source": specs["label"],
            "fixed_value": np.nan,
        })
    return pd.DataFrame(rows)


def fixed_stats_from_summary(
    summary: pd.DataFrame, parameters: list[str],
) -> dict[str, dict[str, float]]:
    indexed = summary.set_index("parameter")
    missing = set(parameters).difference(indexed.index)
    if missing:
        raise KeyError(f"Summary is missing parameters: {sorted(missing)}")
    return {
        parameter: {
            "mean": float(indexed.loc[parameter, "mean"]),
            "std": float(indexed.loc[parameter, "std"]),
        }
        for parameter in parameters
    }


def fixed_values(fixed_stats: dict[str, dict[str, float]]) -> dict[str, float]:
    return {parameter: values["mean"] for parameter, values in fixed_stats.items()}


def save_posterior_and_diagnostics(
    fit, data: dict, stage_name: str, stage_dir: Path,
    fixed_from_wt: dict[str, dict[str, float]] | None = None,
) -> tuple[Path, pd.DataFrame]:
    specs = stage_specs(stage_name)
    samples = az.from_cmdstanpy(posterior=fit, posterior_predictive=specs["ppc"])
    posterior = samples.posterior
    missing = set(specs["output"]).difference(posterior.data_vars)
    if missing:
        raise KeyError(f"Stan output is missing variables: {sorted(missing)}")
    posterior_df = pd.DataFrame({
        name: posterior[name].values.reshape(-1) for name in specs["output"]
    })
    posterior_path = stage_dir / "posterior_samples.csv"
    posterior_df.to_csv(posterior_path, index=False)
    diagnostic_vars = [name for name in specs["output"] if name not in specs["fixed"]]
    az.summary(samples, var_names=diagnostic_vars).to_csv(stage_dir / "arviz_summary.csv")
    summary = make_posterior_summary(posterior_df, stage_name, fixed_from_wt)
    summary.to_csv(stage_dir / "posterior_summary.csv", index=False)
    print(f"\n{specs['label']} parameter summary")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    corner_dir = stage_dir / "diagnostics" / "corner"
    ppc_dir = stage_dir / "diagnostics" / "ppc"
    corner_dir.mkdir(parents=True, exist_ok=True)
    ppc_dir.mkdir(parents=True, exist_ok=True)
    for name, parameters in specs["corners"].items():
        output_file(str(corner_dir / f"corner_{name}_linear.html"))
        bokeh_save(corner(samples, parameters=parameters, xtick_label_orientation=np.pi / 4))
    for name, (ppc_var, t_ppc_key, t_key, area_key) in specs["regression"].items():
        values = samples.posterior_predictive[ppc_var]
        value_dim = next(dim for dim in values.dims if dim not in {"chain", "draw"})
        stacked = values.stack(sample=("chain", "draw")).transpose("sample", value_dim).values
        observed = np.column_stack((np.asarray(data[t_key]), np.asarray(data[area_key])))
        output_file(str(ppc_dir / f"ppc_{name}_linear.html"))
        bokeh_save(predictive_regression(
            stacked, samples_x=np.asarray(data[t_ppc_key]), data=observed,
            x_axis_label="t", y_axis_label="A",
        ))
    return posterior_path, summary


def save_combined_summary(summaries: dict[str, pd.DataFrame]) -> Path:
    combined = pd.concat(
        [summary.assign(stage=stage_specs(stage)["label"]) for stage, summary in summaries.items()],
        ignore_index=True,
    )
    columns = ["stage", "parameter", "mean", "std", "status", "source", "fixed_value"]
    path = RESULTS_ROOT / "fits" / "posterior_summary_all.csv"
    combined[columns].to_csv(path, index=False)
    return path


def posterior_prediction_cache(
    df: pd.DataFrame, posterior_paths: dict[str, Path], n_draws=1000, seed=2026,
):
    rng = np.random.default_rng(seed)
    posts = {stage: pd.read_csv(path) for stage, path in posterior_paths.items()}
    t_plot = np.linspace(df["time in hpf"].min(), df["time in hpf"].max(), 200)
    predictions = {}

    for condition, stan_key in STAN_KEYS.items():
        full_post = posts[CONDITION_STAGE[condition]]
        post = full_post
        if n_draws is not None and len(post) > n_draws:
            post = post.sample(n_draws, random_state=seed)

        active = t_plot >= T0[condition]
        curves = []
        for row in post.itertuples(index=False):
            alpha = getattr(row, "alpha")
            beta_ = getattr(row, "beta_")
            A_0 = getattr(row, f"A_0_{stan_key}")
            A_end = getattr(row, "A_end_smoc_dev") if condition == "smoc_dev" else getattr(row, "A_end")
            g_0 = beta_ * (A_end - A_0) / A_end if condition in STEADY_INITIAL_GROWTH else 0.0
            solution = solve_ivp(
                ode_system, [T0[condition], t_plot[-1]], [A_0, g_0],
                t_eval=t_plot[active], args=(alpha, beta_, A_end), method="RK45",
            )
            if not solution.success:
                continue
            Ahat = solution.y[0]
            sigma_abs = getattr(row, f"sigma_{stan_key}")
            sigma_rel = getattr(row, f"sigma_rel_{stan_key}")
            sigma = np.maximum(sigma_abs + sigma_rel * Ahat, np.finfo(float).eps)
            curve = np.full_like(t_plot, np.nan, dtype=float)
            curve[active] = rng.normal(Ahat, sigma)
            curves.append(curve)

        if not curves:
            raise RuntimeError(f"No posterior predictive curves were generated for {condition}.")

        curves = np.asarray(curves)
        lower = np.full_like(t_plot, np.nan, dtype=float)
        upper = np.full_like(t_plot, np.nan, dtype=float)
        lower[active], upper[active] = np.quantile(curves[:, active], [0.05, 0.95], axis=0)

        alpha_mean = full_post["alpha"].mean()
        beta_mean = full_post["beta_"].mean()
        A_0_mean = full_post[f"A_0_{stan_key}"].mean()
        A_end_mean = (
            full_post["A_end_smoc_dev"].mean()
            if condition == "smoc_dev"
            else full_post["A_end"].mean()
        )
        g_0_mean = (
            beta_mean * (A_end_mean - A_0_mean) / A_end_mean
            if condition in STEADY_INITIAL_GROWTH
            else 0.0
        )
        mean_solution = solve_ivp(
            ode_system, [T0[condition], t_plot[-1]], [A_0_mean, g_0_mean],
            t_eval=t_plot[active], args=(alpha_mean, beta_mean, A_end_mean), method="RK45",
        )
        if not mean_solution.success:
            raise RuntimeError(f"Mean-parameter ODE solution failed for {condition}.")

        central = np.full_like(t_plot, np.nan, dtype=float)
        central[active] = mean_solution.y[0]
        predictions[condition] = {
            "time": t_plot,
            "mean": central,
            "lower": lower,
            "upper": upper,
        }

    return predictions, posts

def plot_fit(
    df: pd.DataFrame, predictions: dict, posts: dict[str, pd.DataFrame], conditions=None,
):
    conditions = list(df["condition"].unique()) if conditions is None else list(conditions)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for condition in conditions:
        sub = df[df["condition"] == condition]
        prediction = predictions[condition]
        ax.scatter(
            sub["time in hpf"], sub["Surface Area"], color=COLORS[condition],
            edgecolor="white", linewidth=1.0, s=80, alpha=0.6, zorder=3,
        )
        ax.plot(
            prediction["time"], prediction["mean"], color=COLORS[condition],
            linewidth=2.5, zorder=2,
        )
        ax.fill_between(
            prediction["time"], prediction["lower"], prediction["upper"],
            color=COLORS[condition], alpha=0.25, linewidth=0,
        )
    wt_A_end = posts["wt"]["A_end"].mean()
    ax.axhline(wt_A_end, color="black", linestyle="--", linewidth=2, alpha=0.8)
    if "smoc_dev" in conditions:
        smoc_A_end = posts["smoc_dev"]["A_end_smoc_dev"].mean()
        ax.axhline(
            smoc_A_end, color=COLORS["smoc_dev"], linestyle="--", linewidth=2, alpha=0.8,
        )
    tau = 1.0 / posts["wt"]["alpha"].mean()
    xticks = [48, 60, 72, 84, 96, 108, 120, 132, 144]
    ax.set_xlim(45, 150)
    ax.set_ylim(0, 15)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=45, fontsize=16)
    ax.set_yticks(np.arange(0, 15, 2))
    ax.set_yticklabels(np.arange(0, 15, 2), fontsize=16)
    x0 = xticks[-1] - 40
    y0 = 0.1 * ax.get_ylim()[1]
    ax.plot([x0, x0 + tau], [y0, y0], color="black", linewidth=3)
    ax.text(x0 + tau / 2, y0 * 0.85, r"$\tau$", ha="center", va="top", fontsize=14)
    ax.set_xlabel("Developmental time [hpf]", fontsize=20)
    ax.set_ylabel("Surface Area", fontsize=20)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_bounds(ax.get_xlim()[0], xticks[-1])
    plt.subplots_adjust(left=0.15, right=0.97, bottom=0.2025, top=0.979)
    return fig


def write_run_metadata(
    smoc_dev_fixed: dict[str, dict[str, float]],
    smoc_reg_fixed: dict[str, dict[str, float]],
):
    metadata_dir = RESULTS_ROOT / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "fit_order": ["WT", "SMOC development", "SMOC regeneration"],
        "smoc_development_fixed_from_wt": smoc_dev_fixed,
        "smoc_regeneration_fixed_from_wt": smoc_reg_fixed,
        "smoc_development_fitted": [
            "A_end_smoc_dev", "A_0_smoc_dev", "sigma_smoc_dev", "sigma_rel_smoc_dev"
        ],
        "smoc_regeneration_fitted": [
            "A_0_smoc_reg", "sigma_smoc_reg", "sigma_rel_smoc_reg"
        ],
    }
    with (metadata_dir / "staged_fit_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def copy_stan_sources():
    model_dir = RESULTS_ROOT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    for path in [STAN_FUNCTIONS, *STAN_PATHS.values()]:
        shutil.copy2(path, model_dir / path.name)


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    copy_stan_sources()
    df = get_data()
    data_dir = RESULTS_ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    df.to_csv(data_dir / "area_data_scaled.csv", index=False)
    for name, conditions in CONDITION_SETS.items():
        print(f"\nExplorative: {name}")
        fig = explorative_plotting(df, conditions)
        save_figure(fig, RESULTS_ROOT / "explorative" / name / "plot")
    for condition in WT_CONDITIONS:
        print(f"\nPrior predictive: {condition}")
        fig = plot_prior_predictive(df, condition)
        save_figure(fig, RESULTS_ROOT / "prior_predictive" / condition / "plot")
    models = compile_models()
    fit_dirs = {
        "wt": RESULTS_ROOT / "fits" / "01_wt",
        "smoc_dev": RESULTS_ROOT / "fits" / "02_smoc_development",
        "smoc_reg": RESULTS_ROOT / "fits" / "03_smoc_regeneration",
    }
    print("\nFitting stage 1: WT data")
    wt_data = prepare_wt_stan_data(df)
    wt_fit = fit_stan_stage(models["wt"], wt_data, fit_dirs["wt"], seed=5)
    wt_posterior, wt_summary = save_posterior_and_diagnostics(
        wt_fit, wt_data, "wt", fit_dirs["wt"]
    )
    wt_stats = fixed_stats_from_summary(wt_summary, ["alpha", "beta_", "A_end"])
    smoc_dev_fixed = {key: wt_stats[key] for key in ["alpha", "beta_"]}
    print("\nFitting stage 2: SMOC development with WT alpha and beta fixed; final size fitted")
    smoc_dev_data = prepare_smoc_stan_data(df, "smoc_dev", fixed_values(smoc_dev_fixed))
    smoc_dev_fit = fit_stan_stage(
        models["smoc_dev"], smoc_dev_data, fit_dirs["smoc_dev"], seed=102
    )
    smoc_dev_posterior, smoc_dev_summary = save_posterior_and_diagnostics(
        smoc_dev_fit, smoc_dev_data, "smoc_dev", fit_dirs["smoc_dev"], smoc_dev_fixed
    )
    smoc_reg_fixed = {key: wt_stats[key] for key in ["alpha", "beta_", "A_end"]}
    print("\nFitting stage 3: SMOC regeneration with WT alpha, beta and final size fixed")
    smoc_reg_data = prepare_smoc_stan_data(df, "smoc_reg", fixed_values(smoc_reg_fixed))
    smoc_reg_fit = fit_stan_stage(
        models["smoc_reg"], smoc_reg_data, fit_dirs["smoc_reg"], seed=103
    )
    smoc_reg_posterior, smoc_reg_summary = save_posterior_and_diagnostics(
        smoc_reg_fit, smoc_reg_data, "smoc_reg", fit_dirs["smoc_reg"], smoc_reg_fixed
    )
    save_combined_summary({
        "wt": wt_summary, "smoc_dev": smoc_dev_summary, "smoc_reg": smoc_reg_summary
    })
    write_run_metadata(smoc_dev_fixed, smoc_reg_fixed)
    posterior_paths = {
        "wt": wt_posterior,
        "smoc_dev": smoc_dev_posterior,
        "smoc_reg": smoc_reg_posterior,
    }
    predictions, posts = posterior_prediction_cache(df, posterior_paths)
    for name, conditions in CONDITION_SETS.items():
        print(f"\nPosterior comparison: {name}")
        fig = plot_fit(df, predictions, posts, conditions)
        save_figure(fig, RESULTS_ROOT / "posterior_comparisons" / name / "plot")
    print(f"\nAll results were written to: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
