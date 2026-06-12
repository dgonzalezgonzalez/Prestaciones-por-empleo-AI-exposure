# Blindspot Report

**Output:** Descriptive figures, occupation exposure tables, DiD specification table, and monthly event-study estimates  
**Date:** 2026-06-03

## Vice 1: The Unexplained Feature

- Features listed: 3 descriptive exposure groups, 13 regression specifications, 63 event-time coefficients, 3 exposure measures.
- Hardest to explain: the main two-way fixed-effects DiD is positive and statistically significant, but the occupation-specific trend specification is essentially zero.
- Resolved? FLAG.

Findings:

- The main DiD estimate is +0.089 log points, about +9.3 percent, with CNO4 and month fixed effects and CNO4-clustered standard errors.
- The same estimand with CNO4-specific linear trends is -0.003 log points and not statistically distinguishable from zero.
- The event study shows post-period coefficients that grow over time, reaching roughly 0.13 log points by 36-42 months after September 2022.
- The pre-event coefficients are not flat. The joint pretrend test rejects strongly, with Wald p-value about 2.1e-08.
- Descriptively, high-exposure occupations are persistently different from zero-exposure occupations. They have higher mean log unemployment levels throughout the panel and appear to improve less after 2022.

Interpretation risk:

- A simple positive post coefficient can be generated either by AI-driven relative unemployment increases or by continuation/curvature of pre-existing occupation-specific trends. The trend-augmented result makes the second explanation too plausible to ignore.

## Vice 2: The Convenient Absence

- Missing checks identified: unemployment rates with labor-force/employment denominators; richer age/gender/province heterogeneity; alternative treatment dates beyond September and November 2022; region-month controls using the province panels; occupation composition controls; explicit matched-description validation for high-exposure CNO4 occupations.
- Missing subgroups: young workers, gender, province/region, high-turnover occupations, administrative versus professional high-exposure groups.
- Unexplained N changes: small differences come from zero or missing `parados` when using `log(parados)`; the `log(parados + 1)` specification restores those rows.

Findings:

- The current outcome is a count of registered unemployed workers by prior occupation, not an unemployment rate. Occupation fixed effects absorb time-invariant size, but differential occupation growth or shrinking labor-force denominators can still confound the interpretation.
- The data contain `contratos`, and the contracts robustness is positive, not negative. That is awkward for a pure job-destruction story and deserves discussion.
- The exposure measure is a semantic-transfer estimate from Anthropic/O*NET to CNO4. It is exposure/potential penetration, not observed Spanish AI adoption by occupation-month.

## Virtue 1: The Unasked Question

- Heterogeneity opportunities: age and gender tables can test whether unemployment effects are concentrated among entrants or specific demographic groups; province disaggregations can separate national occupation trends from regional shocks.
- Mechanism evidence: `contratos` can proxy job starts or hiring demand; the current contracts estimate is positive, suggesting possible churn or demand growth in exposed occupations rather than straightforward displacement.
- Secondary findings: the occupation exposure list validates many "usual suspects" but also surfaces questionable or domain-specific high-exposure occupations such as tourism and PR-related roles.

Findings:

- There may be a better paper in the mechanism than in the average unemployment count. The strongest next step is to compare unemployment and contracts jointly, and then split by age where the Anthropic paper predicts more action among young workers/new entrants.

## Virtue 2: The Unexploited Strength

- Undersold design features: the monthly CNO4 panel is rich, balanced, and long enough to show dynamics rather than just a post dummy.
- Unused falsification tests: province-month designs, pre-period placebo trends by subgroup, low-exposure placebo thresholds, and outcomes that should not respond.
- Positioning opportunities: the project is a direct Spain-specific counterpart to Anthropic's unemployment/hiring framework, using a transparent CNO4 adaptation of observed exposure.

Findings:

- The event-study graph is the central object. It both reveals the post divergence and shows why the causal claim is fragile. That is stronger and more honest than leading with a single DiD coefficient.

## Ruling

[ ] CLEAR - proceed to interpretation. No vices found; virtues noted for consideration.  
[ ] CONDITIONAL - proceed but acknowledge open questions explicitly. Vices flagged but manageable.  
[x] HOLD - do not make a causal claim that AI penetration increased unemployment until the pretrend/trend sensitivity is resolved.

Recommended wording:

- "The simple DiD estimates show a relative post-2022 increase in unemployment counts for high-exposure occupations, but the evidence is not robust enough to interpret causally. Event-study pretrends and occupation-specific trend sensitivity indicate that differential pre-existing occupation trends are a serious threat to identification."
