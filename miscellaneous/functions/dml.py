import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, KFold
from scipy.stats import norm
from tqdm.auto import tqdm
import statsmodels.api as sm
from sklearn.base import TransformerMixin, BaseEstimator
from formulaic import Formula

class FormulaTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, formula, array=False):
        self.formula = formula
        self.array = array

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = Formula(self.formula).get_model_matrix(X)
        if self.array:
            return df.values
        return df


def make_transformer(array=False):
    formula = (
        "0 "
        "+ poly(age, degree=3, raw=True)"
        "+ poly(exper, degree=3, raw=True)"
        "+ poly(duree_listes_horsAR, degree=3, raw=True)"
        "+ poly(nenf, degree=2, raw=True)"
        "+ North + IdF + French + African + femme + marie"
        "+ Interim + EndInterim + tempcomp"
        "+ ce1 + ce2"
        "+ nivetude3 + nivetude4"
        "+ salaireB + salaireC + salaireD + salaireE"
        "+ Q1 + Q2 + Q3"
        "+ EconLayoff + PersLayoff"
        "+ primo + Insertion"
        "+ age:exper"
        "+ femme:nenf"
        "+ femme:exper"
        "+ duree_listes_horsAR:exper"
        "+ French:IdF"
        "+ African:IdF"
        "+ nivetude3:exper"
        "+ nivetude4:exper"
    )
    return FormulaTransformer(formula=formula, array=array)


def dml_single_treatment(
    X, D, y, w,
    modely, modeld,
    *,
    nfolds=5,
    classifier_y=False,
    classifier_d=False,
    progress=True
):
    """
    Double Machine Learning (DML) pour un modèle partiellement linéaire
    avec un seul traitement D et un outcome y, via cross-fitting.

    Paramètres
    ----------
    X : array-like ou DataFrame
        Covariables.
    D : array-like
        Traitement.
    y : array-like
        Variable de résultat.
    modely : estimator sklearn-compatible
        Modèle pour estimer E[y | X].
    modeld : estimator sklearn-compatible
        Modèle pour estimer E[D | X].
    nfolds : int, default=5
        Nombre de folds pour le cross-fitting.
    classifier_y : bool, default=False
        Utiliser predict_proba pour modely si y est binaire.
    classifier_d : bool, default=False
        Utiliser predict_proba pour modeld si D est binaire.
    clu : array-like, optional
        Identifiants de clusters.
    cluster : bool, default=True
        Si True, erreurs standards clusterisées.
    progress : bool, default=True
        Afficher une barre de progression.

    Returns
    -------
    point : float
        Estimateur DML de l'effet de D sur y.
    stderr : float
        Erreur standard associée.
    yhat : ndarray
        Prédictions cross-fittées de y.
    Dhat : ndarray
        Prédictions cross-fittées de D.
    resy : ndarray
        Résidus de y.
    resD : ndarray
        Résidus de D.
    epsilon : ndarray
        Résidus de la régression finale.
    ols_mod : RegressionResults
        Objet statsmodels complet.
    """

    y = np.asarray(y).ravel()
    D = np.asarray(D).ravel()

    cv = KFold(n_splits=nfolds, shuffle=True, random_state=123) #crossfitting
    pbar = tqdm(total=4, desc="DML steps", disable=not progress)#avancement

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

        return np.asarray(
            cross_val_predict(model, X, target, cv=cv, n_jobs=-1)
        ).ravel()

    # Étape 1 : nuisance function pour y
    yhat = _crossfit_hat(modely, X, y, use_proba=classifier_y)
    pbar.update(1)

    # Étape 2 : nuisance function pour D
    Dhat = _crossfit_hat(modeld, X, D, use_proba=classifier_d)
    pbar.update(1)

    # Étape 3 : résidualisation
    resy = y - yhat
    resD = D - Dhat

    dml_data = pd.DataFrame({
        "resy": resy,
        "resD": resD,
        "w": np.asarray(w).astype(float)
    })
    pbar.update(1)

    # Étape 4 : régression finale pondérée
    resD_cst = sm.add_constant(dml_data["resD"])
    ols_mod = sm.WLS(dml_data["resy"], resD_cst, weights=dml_data["w"]).fit()

    pbar.update(1)
    pbar.close()

    point = ols_mod.params["resD"]
    stderr = ols_mod.bse["resD"]
    epsilon = ols_mod.resid

    return point, stderr, yhat, Dhat, resy, resD, epsilon, ols_mod

def summary_single_treatment(
    point, stderr,
    yhat, Dhat, resy, resD, epsilon,
    X, D, y,
    *, name="D", binary_y=None, binary_d=None
):
    """
    Summary function for DML with one treatment.
    Returns a DataFrame with one row for the treatment.
    """

    y = np.asarray(y).ravel()
    D = np.asarray(D).ravel()

    if binary_d is None:
        unique_d = np.unique(D[~pd.isna(D)])
        binary_d = len(unique_d) <= 2 and set(unique_d).issubset({0, 1})

    if binary_y is None:
        unique_y = np.unique(y[~pd.isna(y)])
        binary_y = len(unique_y) <= 2 and set(unique_y).issubset({0, 1})

    rmse_y = np.sqrt(np.mean(resy**2))
    rmse_d = np.sqrt(np.mean(resD**2))
    rmse_eps = np.sqrt(np.mean(epsilon**2))

    acc_y = np.mean(np.abs(resy) < 0.5) if binary_y else np.nan
    acc_d = np.mean(np.abs(resD) < 0.5) if binary_d else np.nan

    return pd.DataFrame({
        "estimate": [point],
        "stderr": [stderr],
        "lower": [point - 1.96 * stderr],
        "upper": [point + 1.96 * stderr],
        "rmse y": [rmse_y],
        "accuracy y": [acc_y],
        "rmse D": [rmse_d],
        "accuracy D": [acc_d],
        "rmse final": [rmse_eps],
        "n": [len(y)],
    }, index=[name])

def run_dml_grid(X, y, D, w, learners_y, learners_d, nfolds=5):
    results_summary = []
    models = {}

    for name_y, learner_y in learners_y.items():
        for name_d, learner_d in learners_d.items():
            print(f"Running combination: {name_y} / {name_d}")

            dml_out = dml_single_treatment(
                X=X,
                D=D,
                y=y,
                w=w,
                modely=learner_y,
                modeld=learner_d,
                nfolds=nfolds,
                classifier_y=True,
                classifier_d=True,
                progress=True
            )

            (
                point, stderr,
                yhat, Dhat,
                resy, resD,
                epsilon,
                ols_mod
            ) = dml_out

            summ = summary_single_treatment(
                point, stderr,
                yhat, Dhat,
                resy, resD, epsilon,
                X, D, y,
                name="treated"
            ).reset_index(names="treatment")

            rmse_d = np.sqrt(np.mean(resD**2))

            summ["rmse D"] = rmse_d
            summ["learner_y"] = name_y
            summ["learner_d"] = name_d

            results_summary.append(summ)

            models[f"{name_y}__{name_d}"] = {
                "summary": summ,
                "raw_output": dml_out,
                "ols_mod": ols_mod
            }

    results_summary = pd.concat(results_summary, ignore_index=True)

    results_summary = results_summary.sort_values(
        by=["rmse y", "rmse D"],
        ascending=True
    )

    return results_summary, models

def select_best_learners(results_summary, verbose=True):
    """
    Select the best learners for Y and D based on mean RMSE.
    """

    best_y = (
        results_summary
        .groupby("learner_y", as_index=False)["rmse y"]
        .mean()
        .sort_values("rmse y")
    )

    best_d = (
        results_summary
        .groupby("learner_d", as_index=False)["rmse D"]
        .mean()
        .sort_values("rmse D")
    )

    best_learner_y = best_y.iloc[0]["learner_y"]
    best_learner_d = best_d.iloc[0]["learner_d"]

    if verbose:
        print("Best learner for Y:")
        print(best_y.to_string(index=False))

        print("\nBest learner for D:")
        print(best_d.to_string(index=False))

        print("\nSelected learners:")
        print("Y =", best_learner_y)
        print("D =", best_learner_d)

    return best_learner_y, best_learner_d

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

def stars(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""

def fmt_coef_and_se(model, var):
    coef = model.params[var]
    se = model.bse[var]
    pval = model.pvalues[var]
    return f"{coef:.3f}{stars(pval)}", f"({se:.3f})"

def fmt_iv_coef_and_se(model, var):
    coef = model.params[var]
    se = model.std_errors[var]
    pval = model.pvalues[var]
    return f"{coef:.3f}{stars(pval)}", f"({se:.3f})"
