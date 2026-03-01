# Who, When, and Which Program?
## Public vs Private Job-Search Counseling in a French RCT:
## High-Dimensional ITT, DML-LATE, and Causal Forest Heterogeneity

This project builds on a French randomized controlled trial (RCT) comparing public (CVE) and private (OPP) job-search counseling programs, conducted by Behagel et.al (2013). The objective is to move beyond average treatment effects and provide a high-dimensional, and policy-relevant evaluation of program performance and optimal allocation.

---

## Project Overview

The analysis is structured in three complementary steps:

### Part 1 — High-Dimensional Intent-to-Treat (ITT)

We re-estimate the Intent-to-Treat (ITT) effect — the impact of being offered a program — using a Double Selection Post-Lasso approach. This method improves precision in a high-dimensional setting while avoiding manual variable selection and specification search. 

### Part 2 — Local Average Treatment Effects (LATE) via Double Machine Learning (DML)

We estimate the causal effect of actual participation (program take-up) using a Double Machine Learning (DML) framework. As the treatment was not randomized, we re-esimate the Local Average Treatment Effect for compliers under a DML framework. 

### Part 3 — Heterogeneous Treatment Effects and the “Parking” Hypothesis

We investigate treatment effect heterogeneity using Causal Forests (Generalized Random Forests). This step aims to detect whether private providers may have strategically “parked” certain individuals — exerting minimal effort on participants unlikely to generate performance bonuses.

Unlike linear interaction models that average effects within broad groups used in the original paper, causal forests allow flexible detection of complex, multidimensional heterogeneity patterns.

---

## Policy Objective

Beyond impact estimation, the project aims to derive optimal policy rules to allocate individuals to public or private counseling programs based on predicted treatment effects.

## Repository Structure

```bash
Who-When-and-Which-Program/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
|   |── part0_EDA/
│   ├── part1_itt_double_selection/
│   ├── part2_late_dml/
│   ├── part3_heterogeneity_grf/
│   └── pipeline/
├── notebooks/
├── outputs/
└── reports/
