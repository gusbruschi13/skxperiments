# Glossary

Core terms of experimentation and causal inference, in the library's language.
Ordered from the most fundamental to the most specific.

## Foundations

**Potential outcomes (`Y(0)`, `Y(1)`).** The two results a unit *would* have
under control and under treatment. The fundamental problem of causal
inference: we observe only **one** per unit; the other is counterfactual.

**Individual effect (ITE).** `Y(1) - Y(0)` for a unit. Never directly
observable.

**ATE (*Average Treatment Effect*).** The average of the individual effects in
the population of interest. The default target of the library's estimators.

**SATE vs. PATE.** *Sample* ATE (average effect over **these** units) vs.
*Population* ATE (average effect in the larger population). The distinction
connects to the next point.

**Finite population vs. superpopulation.** Inferring about *these* units
(finite; `NeymanCI`) vs. about a larger population they are sampled from
(superpopulation; `BootstrapCI`). Not a technicality: it changes what the
confidence interval means.

## Design

**Randomization.** Assigning treatment at random. It is what lets us attribute
outcome differences to the treatment rather than to confounders.

**Confounder.** A variable that affects both the chance of receiving treatment
and the outcome. Randomization breaks confounding by construction.

**CRD (*Completely Randomized Design*).** Half (or a proportion `p`) of the
units to treatment, at random, with no structure.

**Blocking.** Randomizing within strata (blocks) to guarantee balance on known
variables. Implemented by `BlockedCRD`.

**Rerandomization.** Re-drawing until the allocation is balanced by a criterion
(Mahalanobis distance). Implemented by `ReRandomizedCRD`.

**Factorial 2^K.** A design with `K` binary factors and all combinations.
Lets you estimate main effects and interactions.

## Inference

**Sharp null (Fisher).** The **strong** null hypothesis: **zero effect for
every** unit. This is what `RandomizationTest` tests by permuting treatment.

**Randomization-based inference.** The p-value comes from the randomization
mechanism you chose, not from distributional assumptions.

**Neyman variance.** Conservative estimator of the ATE variance under a finite
population (`NeymanCI`).

**Bootstrap.** Resampling with replacement to approximate the sampling
distribution under a superpopulation (`BootstrapCI`): percentile or BCa.

**p-value.** Probability, under the null, of an effect as or more extreme than
the observed one. Small suggests the null is implausible.

**Confidence interval.** A range of plausible values for the effect, at level
`1 - alpha`.

## Multiple testing

**FWER (*family-wise error rate*).** Probability of **at least one** false
positive across a family of tests. Controlled by Bonferroni and Holm.

**FDR (*false discovery rate*).** Expected **proportion** of false positives
among discoveries. Controlled by Benjamini-Hochberg (BH).

## Diagnostics

**SMD (*Standardized Mean Difference*).** Standardized difference in means,
used to check covariate balance. Rule of thumb: `|SMD| > 0.1` flags relevant
imbalance.

**SRM (*Sample Ratio Mismatch*).** The observed allocation departs from the
intended one. It is an **implementation-bug** alarm (asymmetric logging, bot
filtering), not a scientific hypothesis, which is why the threshold is a strict
0.001.

**A/A test.** Re-randomizing over data with no effect to check pipeline
calibration: the false-positive rate should match `alpha`.
