library(readr)
library(dplyr)

# load data
df <- read_csv("data/df_eda_clean.csv")
data <- df

# region variables
data <- data %>%
  mutate(
    IdF = as.integer(nregion == "116"),
    North = as.integer(nregion == "311"),
    Otherregion = 1 - IdF - North
  )

# reason for job loss
data <- data %>%
  mutate(
    EconLayoff = as.integer(motins == "1"),
    PersLayoff = as.integer(motins == "2"),
    EndCDD = as.integer(motins == "4"),
    EndInterim = as.integer(motins == "5"),
    Otherend = 1 - (EconLayoff + PersLayoff + EndCDD + EndInterim)
  )

# experience
data <- data %>%
  mutate(
    exper0 = as.integer(exper == "00"),
    exper1_5 = as.integer(exper %in% c("01", "02", "03", "04", "05")),
    experM5 = 1 - exper0 - exper1_5
  )

# statistical risk group
data <- data %>%
  mutate(
    rsqstat2 = as.integer(rsqstat == "RS2"),
    rsqstat3 = as.integer(rsqstat == "RS3"),
    Orsqstat = 1 - rsqstat2 - rsqstat3
  )

# job search type
data <- data %>%
  mutate(
    tempcomp = as.integer(temps == "1"),
    Otemp = 1 - tempcomp
  )

# sensitive area
data <- data %>%
  mutate(
    dezus = as.integer(zus == "ZU")
  )

# wage expectations
for (s in c("A", "B", "C", "D", "E")) {
  data[[paste0("salaire", s)]] <- as.integer(data$salaire == s)
}

data <- data %>%
  mutate(
    salaireG = as.integer(salaire %in% c("G", ""))
  )

# job type
data <- data %>%
  mutate(
    ce1 = as.integer(cemploi == "CE1"),
    ce2 = as.integer(cemploi == "CE2"),
    cemiss = as.integer(cemploi == "")
  )

# first time unemployed
data <- data %>%
  mutate(
    primo = as.integer(ndem == 1)
  )

# socioprofessional category
data <- data %>%
  mutate(
    Cadre = as.integer(CS == 3),
    Techn = as.integer(CS == 4),
    EmployQ = as.integer(CS == 51),
    EmployNQ = as.integer(CS == 56),
    OuvrQ = as.integer(CS == 61),
    OuvrNQ = as.integer(CS %in% c(66, 99))
  )

# nationality groups
data <- data %>%
  mutate(
    nation_int = suppressWarnings(as.numeric(nation)),
    African = as.integer(nation_int >= 31 & nation_int <= 49),
    EasternEurope = as.integer(
      (nation_int >= 90 & nation_int <= 98) |
        (nation_int %in% c(24, 25, 27))
    ),
    SouthEuropTurkey = as.integer(
      nation_int %in% c(2, 3, 14, 19, 21, 22, 24, 27, 26)
    )
  )

# children
data <- data %>%
  mutate(
    nochild = as.integer(nenf == 0),
    onechild = as.integer(nenf == 1),
    twoormorechild = as.integer(nenf > 1)
  )

# randomization cohort
data <- data %>%
  mutate(
    Q1 = as.integer(between(mois_saisie_occ, 1, 3)),
    Q2 = as.integer(between(mois_saisie_occ, 4, 6)),
    Q3 = as.integer(between(mois_saisie_occ, 7, 9)),
    Q4 = as.integer(between(mois_saisie_occ, 10, 12))
  )

# We have a bunch of categorical variables with no values, we input "inconnu"
cols_na <- names(data)[colSums(is.na(data)) > 0]

data[cols_na] <- lapply(data[cols_na], function(x) {
  x <- as.character(x)
  x[is.na(x)] <- "inconnu"
  as.factor(x)
})

# This is the set of covariates we will consider for analysis

covars = c(
  "IdF",
  "North",
  "Otherregion",
  "EconLayoff",
  "PersLayoff",
  "EndCDD",
  "EndInterim",
  "Otherend",
  "exper0",
  "exper1_5",
  "experM5",
  "rsqstat2",
  "rsqstat3",
  "Orsqstat",
  "tempcomp",
  "Otemp",
  "dezus",
  "salaireA",
  "salaireB",
  "salaireC",
  "salaireD",
  "salaireE",
  "salaireG",
  "ce1",
  "ce2",
  "cemiss",
  "primo",
  "Cadre",
  "Techn",
  "EmployQ",
  "EmployNQ",
  "OuvrQ",
  "OuvrNQ",
  "African",
  "EasternEurope",
  "SouthEuropTurkey",
  "nochild",
  "onechild",
  "twoormorechild",
  "femme",
  "Q1",
  "Q2",
  "Q3",
  "Q4",
  "agegr2635",
  "agegr3645",
  "agegr4655",
  "agegr56"
)

