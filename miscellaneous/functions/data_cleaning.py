import numpy as np
import pandas as pd


def to_str(s: pd.Series) -> pd.Series:
    """Safely cast to pandas string dtype while preserving missing values."""
    return s.astype("string")


def inrange(s: pd.Series, a: int, b: int) -> pd.Series:
    """Return True when values are between a and b, inclusive."""
    return s.between(a, b, inclusive="both")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and enrich the input dataframe with derived variables.

    Parameters
    ----------
    data : pd.DataFrame
        Raw input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with added features.
    """
    df = data.copy()

    # Cast key columns to string where needed
    string_cols = [
        "nregion", "motins", "exper", "rsqstat", "temps", "zus",
        "salaire", "cemploi", "sexe", "nation", "lot", "ale"
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = to_str(df[col])

    # Cast numeric columns
    numeric_cols = ["ndem", "CS", "nenf", "age", "mois_saisie_occ"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1) Region dummies
    df["IdF"] = (df["nregion"] == "116").astype(int)
    df["North"] = (df["nregion"] == "311").astype(int)

    # 2) Unemployment reason
    df["EconLayoff"] = (df["motins"] == "1").astype(int)
    df["PersLayoff"] = (df["motins"] == "2").astype(int)
    df["EndCDD"] = (df["motins"] == "4").astype(int)
    df["EndInterim"] = (df["motins"] == "5").astype(int)
    df["Otherend"] = (
        1 - df["EconLayoff"] - df["PersLayoff"] - df["EndCDD"] - df["EndInterim"]
    ).astype(int)

    # 3) Experience in job
    df["exper0"] = (df["exper"] == "00").astype(int)
    df["exper1_5"] = df["exper"].isin(["01", "02", "03", "04", "05"]).astype(int)
    df["experM5"] = (1 - df["exper0"] - df["exper1_5"]).astype(int)

    # 4) Statistical risk
    df["rsqstat2"] = (df["rsqstat"] == "RS2").astype(int)
    df["rsqstat3"] = (df["rsqstat"] == "RS3").astype(int)
    df["Orsqstat"] = (1 - df["rsqstat2"] - df["rsqstat3"]).astype(int)

    # 5) Full-time search
    df["tempcomp"] = (df["temps"] == "1").astype(int)
    df["Otemp"] = (1 - df["tempcomp"]).astype(int)

    # 6) Sensitive suburban area
    df["dezus"] = (df["zus"] == "ZU").astype(int)

    # 7) Wage target
    df["salaireA"] = (df["salaire"] == "A").astype(int)
    df["salaireB"] = (df["salaire"] == "B").astype(int)
    df["salaireC"] = (df["salaire"] == "C").astype(int)
    df["salaireD"] = (df["salaire"] == "D").astype(int)
    df["salaireE"] = (df["salaire"] == "E").astype(int)
    df["salaireG"] = ((df["salaire"] == "G") | (df["salaire"].fillna("") == "")).astype(int)

    # 8) Employment component
    df["ce1"] = (df["cemploi"] == "CE1").astype(int)
    df["ce2"] = (df["cemploi"] == "CE2").astype(int)
    df["cemiss"] = (df["cemploi"].fillna("") == "").astype(int)

    # 9) First unemployment spell
    df["primo"] = (df["ndem"] == 1).astype(int)

    # 10) Occupation categories
    df["Cadre"] = (df["CS"] == 3).astype(int)
    df["Techn"] = (df["CS"] == 4).astype(int)
    df["EmployQ"] = (df["CS"] == 51).astype(int)
    df["EmployNQ"] = (df["CS"] == 56).astype(int)
    df["OuvrQ"] = (df["CS"] == 61).astype(int)
    df["OuvrNQ"] = df["CS"].isin([66, 99]).astype(int)

    # 11) Nationality groups
    nation = df["nation"].fillna("")

    df["African"] = ((nation >= "31") & (nation <= "49")).astype(int)
    df["EasternEurope"] = (
        ((nation >= "90") & (nation <= "98")) |
        nation.isin(["24", "25", "27"])
    ).astype(int)
    df["SouthEuropTurkey"] = nation.isin(
        ["02", "03", "14", "19", "21", "22", "24", "26", "27"]
    ).astype(int)

    if "etranger" in df.columns:
        df["French"] = pd.to_numeric(df["etranger"], errors="coerce").fillna(0).astype(int)
    else:
        df["French"] = np.nan

    df["Otherregion"] = (1 - df["IdF"] - df["North"]).astype(int)

    if df["French"].notna().any():
        df["Othernation"] = (
            1 - df["French"].fillna(0).astype(int) - df["African"]
        ).astype(int)
    else:
        df["Othernation"] = np.nan

    # 12) Children and sex
    df["nochild"] = (df["nenf"] == 0).astype(int)
    df["onechild"] = (df["nenf"] == 1).astype(int)
    df["twoormorechild"] = (df["nenf"] > 1).astype(int)
    df["woman"] = (df["sexe"] == "2").astype(int)

    # 13) Type of operator
    counseling_lots = {"6", "10", "14", "15", "16", "17"}
    interim_lots = {"12", "13", "19", "24", "25"}
    insertion_lots = {"7", "18", "22", "23"}

    df["TypeOPP"] = ""
    df.loc[df["lot"].isin(counseling_lots), "TypeOPP"] = "Counseling"
    df.loc[df["lot"].isin(interim_lots), "TypeOPP"] = "Interim"
    df.loc[df["lot"].isin(insertion_lots), "TypeOPP"] = "Insertion"

    df["conseil"] = (df["TypeOPP"] == "Counseling").astype(int)
    df["interim"] = (df["TypeOPP"] == "Interim").astype(int)
    df["insertion"] = (df["TypeOPP"] == "Insertion").astype(int)

    # 14) Clean local area code
    df["alec"] = df["ale"]

    recode_alec = {
        "77111": "77103",
        "75884": "75861",
        "59121": "59113",
        "42002": "42024",
        "42040": "42024",
        "26031": "26023",
    }
    df["alec"] = df["alec"].replace(recode_alec)

    # 15) Dominant operator type by area
    area_means = (
        df.groupby("alec")[["interim", "insertion", "conseil"]]
        .mean()
        .rename(columns={
            "interim": "Einterim",
            "insertion": "Einsertion",
            "conseil": "Econseil"
        })
    )

    df = df.join(area_means, on="alec")

    df["Interim"] = (
        (df["Einterim"] > df["Econseil"]) &
        (df["Einterim"] > df["Einsertion"])
    ).astype(int)

    df["Insertion"] = (
        (df["Einsertion"] > df["Econseil"]) &
        (df["Einsertion"] > df["Einterim"])
    ).astype(int)

    df["Conseil"] = (
        (df["Econseil"] > df["Einterim"]) &
        (df["Econseil"] > df["Einsertion"])
    ).astype(int)

    df["Interimnc"] = df["Interim"]
    df["Insertionnc"] = df["Insertion"]
    df["Conseilnc"] = df["Conseil"]

    df["AreaTypeOPP"] = np.where(
        df["Interim"] == 1, "Interim",
        np.where(
            df["Insertion"] == 1, "Insertion",
            np.where(df["Conseil"] == 1, "Counseling", "")
        )
    )

    # 16) Assignment quarter
    df["Q1"] = inrange(df["mois_saisie_occ"], 1, 3).astype(int)
    df["Q2"] = inrange(df["mois_saisie_occ"], 4, 6).astype(int)
    df["Q3"] = inrange(df["mois_saisie_occ"], 7, 9).astype(int)
    df["Q4"] = inrange(df["mois_saisie_occ"], 10, 12).astype(int)

    return df