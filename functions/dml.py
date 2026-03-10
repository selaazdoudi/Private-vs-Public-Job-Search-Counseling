import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold

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

    Stage 1 (cross-fitting):
        yhat  = E[y|X]
        D1hat = E[D1|X]
        D2hat = E[D2|X]

    Residuals:
        resy  = y  - yhat
        resD1 = D1 - D1hat
        resD2 = D2 - D2hat

    Final stage:
        resy ~ 1 + resD1 + resD2

    Important:
    - If classifier_* = True but the model does NOT implement predict_proba
      (e.g. MLPRegressor, LassoCV, RandomForestRegressor, etc.),
      the function automatically falls back to predict().
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
        """
        Returns True if model/pipeline exposes predict_proba, False otherwise.
        Works for sklearn Pipelines too.
        """
        return hasattr(model, "predict_proba")

    def _crossfit_hat(model, X, target, use_proba=False):
        """
        Cross-fitted prediction helper:
        - uses predict_proba[:, 1] if requested and available
        - otherwise falls back to predict()
        """
        if use_proba and _supports_predict_proba(model):
            pred = cross_val_predict(
                model, X, target, cv=cv, method="predict_proba", n_jobs=-1
            )
            # binary classification: keep prob of class 1
            if pred.ndim == 2 and pred.shape[1] >= 2:
                return pred[:, 1]
            # defensive fallback
            return np.asarray(pred).ravel()

        # fallback: ordinary prediction
        return cross_val_predict(model, X, target, cv=cv, n_jobs=-1)

    # 1) yhat
    yhat = _crossfit_hat(modely, X, y, use_proba=classifier_y)
    pbar.update(1)

    # 2) D1hat
    D1hat = _crossfit_hat(modeld1, X, D1, use_proba=classifier_d1)
    pbar.update(1)

    # 3) D2hat
    D2hat = _crossfit_hat(modeld2, X, D2, use_proba=classifier_d2)
    pbar.update(1)

    # 4) Residuals
    resy = y - yhat
    resD1 = D1 - D1hat
    resD2 = D2 - D2hat

    dml_data = pd.DataFrame({
        "resy": resy,
        "resD1": resD1,
        "resD2": resD2
    })
    pbar.update(1)

    # 5) Final OLS
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

    return point1, point2, stderr1, stderr2, yhat, D1hat, D2hat, resy, resD1, resD2, epsilon


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

    # Convert to 1d arrays
    y = np.asarray(y).ravel()
    D1 = np.asarray(D1).ravel()
    D2 = np.asarray(D2).ravel()

    # If not specified, infer whether treatments are binary
    if binary_d1 is None:
        unique_d1 = np.unique(D1[~pd.isna(D1)])
        binary_d1 = len(unique_d1) <= 2 and set(unique_d1).issubset({0, 1})

    if binary_d2 is None:
        unique_d2 = np.unique(D2[~pd.isna(D2)])
        binary_d2 = len(unique_d2) <= 2 and set(unique_d2).issubset({0, 1})

    if binary_y is None:
        unique_y = np.unique(y[~pd.isna(y)])
        binary_y = len(unique_y) <= 2 and set(unique_y).issubset({0, 1})

    # Common metrics
    rmse_y = np.sqrt(np.mean(resy**2))
    rmse_eps = np.sqrt(np.mean(epsilon**2))

    # Treatment-specific metrics
    rmse_d1 = np.sqrt(np.mean(resD1**2))
    rmse_d2 = np.sqrt(np.mean(resD2**2))

    acc_d1 = np.mean(np.abs(resD1) < 0.5) if binary_d1 else np.nan
    acc_d2 = np.mean(np.abs(resD2) < 0.5) if binary_d2 else np.nan
    acc_y = np.mean(np.abs(resy) < 0.5) if binary_y else np.nan
    return pd.DataFrame({
        "estimate":   [point1, point2],
        "stderr":     [stderr1, stderr2],
        "lower":      [point1 - 1.96 * stderr1, point2 - 1.96 * stderr2],
        "upper":      [point1 + 1.96 * stderr1, point2 + 1.96 * stderr2],
        "rmse y":     [rmse_y, rmse_y],
        "accuracy y":[acc_y, acc_y],
        "rmse D":     [rmse_d1, rmse_d2],
        "accuracy D": [acc_d1, acc_d2],
        "rmse final": [rmse_eps, rmse_eps],
        "n":          [len(y), len(y)],
    }, index=[name1, name2])