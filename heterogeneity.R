library(haven)
library(tidyverse)
library(grf)

data <- read_dta("dataPrivatePublic.dta")

#We are going to estimate the CATE using generalized random forest. Let's prepare the dataset

training_data <- data %>%
  select(acceptationCVE, acceptationOPP, CVE, OPP, #compliance variables and instruments
         nenf, exper, femme, contains("nivetude"), etranger, contains("agegr"), matches("Emploi_[0-9]+"),
         contains("POIDS")) %>%
  mutate(exper = as.numeric(exper))

sapply(training_data, class)

training_OPP <- training_data %>%
  filter(CVE == 0)

training_CVE <- training_data %>%
  filter(OPP == 0)

#On calcule la part de l'échantillon assignée à chaque traitement
n <- nrow(data)

training_data %>%
  filter(CVE != 0 | OPP != 0) %>%
  summarise(part_CVE = sum(CVE)/n, part_OPP = sum(OPP)/n) %>%
  mutate(part_controle = 1 - part_CVE - part_OPP)

#On calcule le nombre de compliers 
compliers <- (sum((training_data$OPP == 1 & training_data$acceptationOPP == 1)) +
sum((training_data$CVE == 1 & training_data$acceptationCVE == 1)) +
sum((training_data$OPP == 0 & training_data$acceptationOPP == 0 & 
    training_data$CVE == 0 & training_data$acceptationCVE == 0))
)/n

######
#Estimation of the grf pour le programme privé
######

#3 Mois

grf_opp_3mois <- instrumental_forest(
  X = training_OPP %>%
    select(nenf, exper, femme, contains("nivetude"), etranger, contains("agegr")) %>% as.matrix(),
  Y = training_OPP %>%
    select(EMPLOI_3MOIS) %>% as.matrix(),
  W = training_OPP %>%
    select(acceptationOPP) %>% as.matrix(),
  Z = training_OPP %>%
    select(OPP) %>% as.matrix(),
  sample.weights = training_OPP$POIDS_PZ_3MOIS
)

tau_hat_opp3M <- predict(grf_opp_3mois)$predictions
summary(tau_hat_opp3M)
hist(tau_hat_opp3M)

#6 Mois

grf_opp_6mois <- instrumental_forest(
  X = training_OPP %>%
    select(nenf, exper, femme, contains("nivetude"), etranger, contains("agegr")) %>% as.matrix(),
  Y = training_OPP %>%
    select(EMPLOI_6MOIS) %>% as.matrix(),
  W = training_OPP %>%
    select(acceptationOPP) %>% as.matrix(),
  Z = training_OPP %>%
    select(OPP) %>% as.matrix(),
  sample.weights = training_OPP$POIDS_PZ_6MOIS
)

tau_hat_opp6M <- predict(grf_opp_6mois)$predictions
summary(tau_hat_opp6M)
hist(tau_hat_opp6M)

#9 Mois

grf_opp_9mois <- instrumental_forest(
  X = training_OPP %>%
    select(nenf, exper, femme, contains("nivetude"), etranger, contains("agegr")) %>% as.matrix(),
  Y = training_OPP %>%
    select(EMPLOI_9MOIS) %>% as.matrix(),
  W = training_OPP %>%
    select(acceptationOPP) %>% as.matrix(),
  Z = training_OPP %>%
    select(OPP) %>% as.matrix(),
  sample.weights = training_OPP$POIDS_PZ_9MOIS
)

tau_hat_opp9M <- predict(grf_opp_9mois)$predictions
summary(tau_hat_opp9M)
hist(tau_hat_opp9M)

#12 Mois

grf_opp_12mois <- instrumental_forest(
  X = training_OPP %>%
    select(nenf, exper, femme, contains("nivetude"), etranger, contains("agegr")) %>% as.matrix(),
  Y = training_OPP %>%
    select(EMPLOI_12MOIS) %>% as.matrix(),
  W = training_OPP %>%
    select(acceptationOPP) %>% as.matrix(),
  Z = training_OPP %>%
    select(OPP) %>% as.matrix(),
  sample.weights = training_OPP$POIDS_PZ_12MOIS
)

tau_hat_opp12M <- predict(grf_opp_12mois)$predictions
summary(tau_hat_opp12M)
hist(tau_hat_opp12M)

#Attempt at computing some GATEs

gate_femme_3M <- aggregate(tau ~ femme, data = data.frame(tau = tau_hat_opp3M, femme = training_OPP$femme), mean) %>%
  mutate(time = rep(3, 2))

gate_femme_6M <- aggregate(tau ~ femme, data = data.frame(tau = tau_hat_opp6M, femme = training_OPP$femme), mean) %>%
  mutate(time = rep(6, 2))

gate_femme_9M <- aggregate(tau ~ femme, data = data.frame(tau = tau_hat_opp9M, femme = training_OPP$femme), mean) %>%
  mutate(time = rep(9, 2))

gate_femme_12M <- aggregate(tau ~ femme, data = data.frame(tau = tau_hat_opp12M, femme = training_OPP$femme), mean) %>%
  mutate(time = rep(12, 2))

cate_chart_template <- rbind(gate_femme_3M, gate_femme_6M, gate_femme_9M, gate_femme_12M)

cate_chart_template$femme <- ifelse(cate_chart_template$femme == 1, "femme", "homme")

ggplot(cate_chart_template, aes(x = time, y = tau, color = femme, group = femme)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = cate_chart_template$time) +
  scale_y_continuous(breaks = seq(0, 0.14, by = 0.02), limits = c(0, 0.12)) +
  labs(
    x = "Time",
    y = "Estimated treatment effect",
    color = "Sex",
    title = "Estimated treatment effects on compliers over time"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

gate_etranger_3M <- aggregate(tau ~ etranger, data = data.frame(tau = tau_hat_opp3M, etranger = training_OPP$etranger), mean) %>%
  mutate(time = rep(3, 2))

gate_etranger_6M <- aggregate(tau ~ etranger, data = data.frame(tau = tau_hat_opp6M, etranger = training_OPP$etranger), mean) %>%
  mutate(time = rep(6, 2))

gate_etranger_9M <- aggregate(tau ~ etranger, data = data.frame(tau = tau_hat_opp9M, etranger = training_OPP$etranger), mean) %>%
  mutate(time = rep(9, 2))

gate_etranger_12M <- aggregate(tau ~ etranger, data = data.frame(tau = tau_hat_opp12M, etranger = training_OPP$etranger), mean) %>%
  mutate(time = rep(12, 2))

cate_chart_template_2 <- rbind(gate_etranger_3M, gate_etranger_6M, gate_etranger_9M, gate_etranger_12M)

cate_chart_template_2$etranger <- ifelse(cate_chart_template_2$etranger == 1, "Etrangers", "Français")

ggplot(cate_chart_template_2, aes(x = time, y = tau, color = etranger, group = etranger)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = cate_chart_template_2$time) +
  scale_y_continuous(breaks = seq(0, 0.14, by = 0.02), limits = c(0, 0.12)) +
  labs(
    x = "Time",
    y = "Estimated treatment effect",
    color = "Sex",
    title = "Estimated treatment effects on compliers over time"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "top",
    panel.grid.minor = element_blank()
  )

######
#Estimation of the grf pour le programme public
######

#3 Mois

grf_cve_3mois <- instrumental_forest(
  X = training_cve %>%
    select(nenf, exper, femme, contains("nivetude"), etranger, contains("agegr")) %>% as.matrix(),
  Y = training_cve %>%
    select(EMPLOI_3MOIS) %>% as.matrix(),
  W = training_cve %>%
    select(acceptationCVE) %>% as.matrix(),
  Z = training_cve %>%
    select(CVE) %>% as.matrix(),
  sample.weights = training_cve$POIDS_PZ_3MOIS
)

tau_hat_cve3M <- predict(grf_cve_3mois)$predictions
summary(tau_hat_cve3M)
hist(tau_hat_cve3M)

#6 Mois

grf_cve_6mois <- instrumental_forest(
  X = training_cve %>%
    select(nenf, exper, femme, contains("nivetude"), etranger, contains("agegr")) %>% as.matrix(),
  Y = training_cve %>%
    select(EMPLOI_6MOIS) %>% as.matrix(),
  W = training_cve %>%
    select(acceptationCVE) %>% as.matrix(),
  Z = training_cve %>%
    select(CVE) %>% as.matrix(),
  sample.weights = training_cve$POIDS_PZ_6MOIS
)

tau_hat_cve6M <- predict(grf_cve_6mois)$predictions
summary(tau_hat_cve6M)
hist(tau_hat_cve6M)

#9 Mois

grf_cve_9mois <- instrumental_forest(
  X = training_cve %>%
    select(nenf, exper, femme, contains("nivetude"), etranger, contains("agegr")) %>% as.matrix(),
  Y = training_cve %>%
    select(EMPLOI_9MOIS) %>% as.matrix(),
  W = training_cve %>%
    select(acceptationCVE) %>% as.matrix(),
  Z = training_cve %>%
    select(CVE) %>% as.matrix(),
  sample.weights = training_cve$POIDS_PZ_9MOIS
)

tau_hat_cve9M <- predict(grf_cve_9mois)$predictions
summary(tau_hat_cve9M)
hist(tau_hat_cve9M)

#12 Mois

grf_cve_12mois <- instrumental_forest(
  X = training_cve %>%
    select(nenf, exper, femme, contains("nivetude"), etranger, contains("agegr")) %>% as.matrix(),
  Y = training_cve %>%
    select(EMPLOI_12MOIS) %>% as.matrix(),
  W = training_cve %>%
    select(acceptationCVE) %>% as.matrix(),
  Z = training_cve %>%
    select(CVE) %>% as.matrix(),
  sample.weights = training_cve$POIDS_PZ_12MOIS
)

tau_hat_cve12M <- predict(grf_cve_12mois)$predictions
summary(tau_hat_cve12M)
hist(tau_hat_cve12M)


