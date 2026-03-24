## Public vs Private Job-Search Counseling in France
## Machine Learning for Econometrics

This project revisits a French randomized controlled trial (RCT) on intensive job-search counseling, comparing a **public program** (CVE, run by ANPE) and a **private program** (OPP, outsourced and funded by Unédic). The project is based on the institutional setting studied by Behaghel et al. and asks a broader question: **does intensive counseling improve employment outcomes, how should we identify its causal effect when take-up is selective, and do treatment effects differ across individuals?**

## Context

This project was conducted as part of the **Machine Learning for Econometrics** course at **ENSAE / École Polytechnique**.

### Contributors
- Salma El Aazdoudi
- Lyna El Kamel
- Stephane Kweke Ngahane
- Sofiene Taamouti

---

## Institutional Background

Job-search counseling has often been found to be more effective than traditional active labour market policies such as training or subsidized employment. In France, an important institutional reform took place in **2005**, when ANPE lost its monopoly over job-seeker placement. In **2007**, two intensive counseling programs were scaled up:

- **CVE**: a public intensive counseling program run by ANPE
- **OPP**: a private outsourced counseling program funded by Unédic

Both programs were much more intensive than the standard track, with a caseload of about **40 job-seekers per advisor**, compared with roughly **120 in the regular system**.

The original experiment includes about **43,977 individuals**. Assignment to CVE, OPP, or the standard track was randomized, but **actual participation in the assigned program was not**. This creates an **encouragement design** with substantial non-compliance.

---

## Research Question

Our project asks:
- Methodologically, can we recover RCT causal effects using a DML approach?
- Does intensive job-search counseling increase exit to employment?
- How do the **public (CVE)** and **private (OPP)** programs compare?
- Do treatment effects vary across individuals, and is there evidence consistent with **parking** in the private program?

The outcomes of interest are employment at **3, 6, 9, and 12 months**.

---

## Project Overview

The analysis is organized in three complementary parts.

### Part I — Experimental Design and Exploratory Data Analysis

We start by revisiting the original RCT and using exploratory data analysis to understand the structure of treatment assignment and take-up. Although assignment to CVE, OPP, or the standard track was randomized, actual participation in the assigned program was not.

The EDA reveals two key patterns. First, take-up rates differ markedly across programs, reaching about **30.8%** in CVE and **42.6%** in OPP. Second, within each programme arm, treated and non-treated individuals differ in their observed characteristics, suggesting that take-up is selective rather than random.

These descriptive results motivate the empirical strategy developed in the rest of the project: if participation is selective, credible estimation requires methods that flexibly account for differences in observable characteristics across individuals.

### Part II — Average Treatment Effect on the Treated via Double Machine Learning

Our main empirical strategy focuses on individuals **within each assigned programme arm**. In other words, among those assigned to CVE or OPP, some take up treatment and others do not. We compare these treated and non-treated individuals **within the assigned group**, rather than relying on the standard-track control group. The goal is to see if we can recover the RCT ATET estimation.

This approach requires a **selection-on-observables** assumption:

Under this assumption, and after controlling flexibly for baseline characteristics, we can recover the causal effect of actual participation on the treated.

To do so, we use a **Double Machine Learning (DML)** procedure based on a **partially linear model**:

We compare this approach to more standard econometric methods such as OLS and IV.

### Part III — Heterogeneous Treatment Effects and the Parking Hypothesis

Average effects may hide important heterogeneity. This is particularly relevant for the private program, where the payment structure may create incentives to focus effort on individuals who are easier to place into employment, while devoting less effort to more fragile job-seekers.

We study this issue in three steps:

1. **Construct baseline employability**  
   We estimate each individual’s probability of finding a job without treatment, using the control group and flexible prediction methods.

2. **Estimate grouped effects by employability quartile**  
   We compare treatment effects across quartiles of predicted baseline employability.

3. **Estimate flexible heterogeneous effects using machine learning**  
   We use instrumental-forest / causal-forest methods to estimate **Conditional LATEs (CLATEs)** without imposing a rigid parametric structure.

This part of the project is designed to test whether the private program exhibits patterns consistent with **parking**.

---

## Main Findings

Our presentation highlights several results:

- Treatment take-up is clearly **non-random within assigned arms**, and baseline covariates significantly predict participation.
- This makes flexible adjustment for observables central to identification.
- DML is somewhat sensitive, and the gap between DML and IV is a reason to worry that omitted variables may still matter
- In the heterogeneity analysis for OPP, grouped IV results suggest somewhat stronger effects for lower baseline-score quartiles.
- However, the machine-learning heterogeneity results remain **fairly similar across quartiles**, with only slightly larger effects in the lowest quartile.
- Overall, we find **no strong evidence of parking** in the private program.

---

## Repository Structure

```bash
Private-vs-Public-Job-Search-Counseling/
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── eda/
│   ├── att_dml/
│   └── heterogeneity/
├── miscellaneous/
│   ├── README.md
│   └── itt_highdimensional.ipynb
└── presentation/
    └── slides.pdf
