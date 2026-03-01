# Who, When, and Which Program?
## Public vs Private Job-Search Counseling in a French RCT:
## High-Dimensional ITT, DML-LATE, and Causal Forest Heterogeneity

This project analyzes a French randomized controlled trial (RCT) comparing public (CVE) and private (OPP) job-search counseling programs. The objective is to move beyond average treatment effects and provide a rigorous, high-dimensional, and policy-relevant evaluation of program performance and optimal allocation.

---

## Project Overview

The analysis is structured in three complementary steps:

### Part 1 — High-Dimensional Intent-to-Treat (ITT)

We estimate the Intent-to-Treat (ITT) effect — the impact of being offered a program — using a Double Selection Post-Lasso approach. This method improves precision in a high-dimensional setting while avoiding manual variable selection and specification search. It ensures valid inference even when the set of potential controls is large.

### Part 2 — Local Average Treatment Effects (LATE) via Double Machine Learning (DML)

We estimate the causal effect of actual participation (program take-up) using a Double Machine Learning (DML) framework. Random assignment serves as an instrument for participation, allowing identification of the Local Average Treatment Effect (LATE) for compliers. Cross-fitting and flexible machine learning methods are used to control for selection bias while maintaining valid inference.

### Part 3 — Heterogeneous Treatment Effects and the “Parking” Hypothesis

We investigate treatment effect heterogeneity using Causal Forests (Generalized Random Forests). This step aims to detect whether private providers may have strategically “parked” certain individuals — exerting minimal effort on participants unlikely to generate performance bonuses.

Unlike linear interaction models that average effects within broad groups, causal forests allow flexible detection of complex, multidimensional heterogeneity patterns.

---

## Policy Objective

Beyond impact estimation, the project derives optimal policy rules to allocate individuals to public or private counseling programs based on predicted treatment effects, linking causal inference with welfare-improving program design.
