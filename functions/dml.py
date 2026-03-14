import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import norm

def dml_two_treatments(
    X, D1, D2, y,
    modely, modeld1, modeld2=None,
    *, nfolds=5,
    classifier_y=False,
    classifier_d1=False,
    classifier_d2=False,
    clu=None, cluster=True,
    progress=True
):
    """
    DML for Partially Linear Model with TWO treatments (D1, D2) using cross-fitting.
    """

    if modeld2 is None:
        modeld2 = modeld1

    y = np.asarray(y).ravel()
    D1 = np.asarray(D1).ravel()
    D2 = np.asarray(D2).ravel()

    cv = KFold(n_splits=nfolds, shuffle=True, random_state=123)

    from tqdm.auto import tqdm
    import statsmodels.formula.api as smf

    pbar = tqdm(total=5, desc="DML steps", disable=not progress)

    def _supports_predict_proba(model):
        return hasattr(model, "predict_proba")

    def _crossfit_hat(model, X, target, use_proba=False):
        if use_proba and _supports_predict_proba(model):
            pred = cross_val_predict(
                model, X, target, cv=cv, method="predict_proba", n_jobs=-1
            )
            if pred.ndim == 2 and pred.shape[1] >= 2:
                return pred[:, 1]
            return np.asarray(pred).ravel()

        return cross_val_predict(model, X, target, cv=cv, n_jobs=-1)

    yhat = _crossfit_hat(modely, X, y, use_proba=classifier_y)
    pbar.update(1)

    D1hat = _crossfit_hat(modeld1, X, D1, use_proba=classifier_d1)
    pbar.update(1)

    D2hat = _crossfit_hat(modeld2, X, D2, use_proba=classifier_d2)
    pbar.update(1)

    resy = y - yhat
    resD1 = D1 - D1hat
    resD2 = D2 - D2hat

    dml_data = pd.DataFrame({
        "resy": resy,
        "resD1": resD1,
        "resD2": resD2
    })
    pbar.update(1)

    if cluster:
        if clu is None:
            raise ValueError("cluster=True but clu is None. Provide cluster ids in clu.")
        dml_data["clu"] = np.asarray(clu)

        ols_mod = smf.ols("resy ~ 1 + resD1 + resD2", data=dml_data).fit(
            cov_type="cluster",
            cov_kwds={"groups": dml_data["clu"]}
        )
    else:
        ols_mod = smf.ols("resy ~ 1 + resD1 + resD2", data=dml_data).fit()

    pbar.update(1)
    pbar.close()

    point1 = ols_mod.params["resD1"]
    point2 = ols_mod.params["resD2"]
    stderr1 = ols_mod.bse["resD1"]
    stderr2 = ols_mod.bse["resD2"]
    epsilon = ols_mod.resid

    return (
        point1, point2, stderr1, stderr2,
        yhat, D1hat, D2hat,
        resy, resD1, resD2,
        epsilon
    )


def summary_two_treatments(
    point1, point2, stderr1, stderr2,
    yhat, D1hat, D2hat, resy, resD1, resD2, epsilon,
    X, D1, D2, y,
    *, name1="D1", name2="D2", binary_y=None,
    binary_d1=None, binary_d2=None
):
    """
    Summary function for DML with two treatments.
    Returns a DataFrame with one row per treatment.
    """

    y = np.asarray(y).ravel()
    D1 = np.asarray(D1).ravel()
    D2 = np.asarray(D2).ravel()

    if binary_d1 is None:
        unique_d1 = np.unique(D1[~pd.isna(D1)])
        binary_d1 = len(unique_d1) <= 2 and set(unique_d1).issubset({0, 1})

    if binary_d2 is None:
        unique_d2 = np.unique(D2[~pd.isna(D2)])
        binary_d2 = len(unique_d2) <= 2 and set(unique_d2).issubset({0, 1})

    if binary_y is None:
        unique_y = np.unique(y[~pd.isna(y)])
        binary_y = len(unique_y) <= 2 and set(unique_y).issubset({0, 1})

    rmse_y = np.sqrt(np.mean(resy**2))
    rmse_eps = np.sqrt(np.mean(epsilon**2))

    rmse_d1 = np.sqrt(np.mean(resD1**2))
    rmse_d2 = np.sqrt(np.mean(resD2**2))

    acc_d1 = np.mean(np.abs(resD1) < 0.5) if binary_d1 else np.nan
    acc_d2 = np.mean(np.abs(resD2) < 0.5) if binary_d2 else np.nan
    acc_y = np.mean(np.abs(resy) < 0.5) if binary_y else np.nan

    return pd.DataFrame({
        "estimate": [point1, point2],
        "stderr": [stderr1, stderr2],
        "lower": [point1 - 1.96 * stderr1, point2 - 1.96 * stderr2],
        "upper": [point1 + 1.96 * stderr1, point2 + 1.96 * stderr2],
        "rmse y": [rmse_y, rmse_y],
        "accuracy y": [acc_y, acc_y],
        "rmse D": [rmse_d1, rmse_d2],
        "accuracy D": [acc_d1, acc_d2],
        "rmse final": [rmse_eps, rmse_eps],
        "n": [len(y), len(y)],
    }, index=[name1, name2])


def run_dml_grid(X, y, D1, D2, learners_y, learners_d, nfolds=5):
    results_summary = []
    models = {}

    for name_y, learner_y in learners_y.items():
        for name_d, learner_d in learners_d.items():
            print(f"Running combination: {name_y} / {name_d}")

            dml_out = dml_two_treatments(
                X=X,
                D1=D1,
                D2=D2,
                y=y,
                modely=learner_y,
                modeld1=learner_d,
                modeld2=learner_d,
                nfolds=nfolds,
                classifier_y=True,
                classifier_d1=True,
                classifier_d2=True,
                cluster=False
            )

            (
                point1, point2, stderr1, stderr2,
                yhat, D1hat, D2hat,
                resy, resD1, resD2,
                epsilon
            ) = dml_out

            summ = summary_two_treatments(
                point1, point2, stderr1, stderr2,
                yhat, D1hat, D2hat,
                resy, resD1, resD2, epsilon,
                X, D1, D2, y,
                name1="CVE_treated",
                name2="OPP_treated"
            ).reset_index(names="treatment")

            rmse_d1 = np.sqrt(np.mean(resD1**2))
            rmse_d2 = np.sqrt(np.mean(resD2**2))

            summ["rmse D1"] = rmse_d1
            summ["rmse D2"] = rmse_d2
            summ["learner_y"] = name_y
            summ["learner_d"] = name_d

            results_summary.append(summ)

            models[f"{name_y}__{name_d}"] = {
                "summary": summ,
                "raw_output": dml_out
            }

    results_summary = pd.concat(results_summary, ignore_index=True)

    results_summary = results_summary.sort_values(
        by=["rmse y", "rmse D1", "rmse D2"],
        ascending=True
    )

    return results_summary


def select_best_learners(results_summary, verbose=True):
    """
    Select the best learners for Y, D1 and D2 based on mean RMSE.
    """

    best_y = (
        results_summary
        .groupby("learner_y", as_index=False)["rmse y"]
        .mean()
        .sort_values("rmse y")
    )

    best_d1 = (
        results_summary
        .groupby("learner_d", as_index=False)["rmse D1"]
        .mean()
        .sort_values("rmse D1")
    )

    best_d2 = (
        results_summary
        .groupby("learner_d", as_index=False)["rmse D2"]
        .mean()
        .sort_values("rmse D2")
    )

    best_learner_y = best_y.iloc[0]["learner_y"]
    best_learner_d1 = best_d1.iloc[0]["learner_d"]
    best_learner_d2 = best_d2.iloc[0]["learner_d"]

    if verbose:
        print("Best learner for Y:")
        print(best_y.to_string(index=False))

        print("\nBest learner for D1:")
        print(best_d1.to_string(index=False))

        print("\nBest learner for D2:")
        print(best_d2.to_string(index=False))

        print("\nSelected learners:")
        print("Y  =", best_learner_y)
        print("D1 =", best_learner_d1)
        print("D2 =", best_learner_d2)

    return best_learner_y, best_learner_d1, best_learner_d2


def stars_from_pvalue(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def make_dml_final_table(
    point1, point2, stderr1, stderr2,
    resy, resD1, resD2, epsilon,
    *,
    name1="CVE_treated",
    name2="OPP_treated",
    learner_y=None,
    learner_d1=None,
    learner_d2=None,
    n=None
):
    z1 = point1 / stderr1 if stderr1 > 0 else np.nan
    z2 = point2 / stderr2 if stderr2 > 0 else np.nan

    pval1 = round(2 * (1 - norm.cdf(abs(z1))), 3) if np.isfinite(z1) else np.nan
    pval2 = round(2 * (1 - norm.cdf(abs(z2))), 3) if np.isfinite(z2) else np.nan

    lower1 = point1 - 1.96 * stderr1
    upper1 = point1 + 1.96 * stderr1
    lower2 = point2 - 1.96 * stderr2
    upper2 = point2 + 1.96 * stderr2

    rmse_y = np.sqrt(np.mean(resy**2))
    rmse_d1 = np.sqrt(np.mean(resD1**2))
    rmse_d2 = np.sqrt(np.mean(resD2**2))
    rmse_final = np.sqrt(np.mean(epsilon**2))

    out = pd.DataFrame({
        "treatment": [name1, name2],
        "estimate": [point1, point2],
        "std.error": [stderr1, stderr2],
        "z": [z1, z2],
        "p.value": [pval1, pval2],
        "ci.lower": [lower1, lower2],
        "ci.upper": [upper1, upper2],
        "rmse_y": [rmse_y, rmse_y],
        "rmse_d1": [rmse_d1, rmse_d1],
        "rmse_d2": [rmse_d2, rmse_d2],
        "rmse_final": [rmse_final, rmse_final],
        "learner_y": [learner_y, learner_y],
        "learner_d1": [learner_d1, learner_d1],
        "learner_d2": [learner_d2, learner_d2],
        "n": [n, n]
    })

    return out
