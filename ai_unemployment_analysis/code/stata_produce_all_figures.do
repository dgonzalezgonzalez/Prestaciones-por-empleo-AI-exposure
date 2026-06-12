version 17
clear all
set more off

* Backward-compatible entry point.
* The requested Stata-first analysis now lives in:
*   code/stata_empirical_analysis_plan.do
*
* It constructs the samples, estimates TWFE with reghdfe, estimates
* synthetic DID with sdid, estimates synthetic control with method(sc),
* and writes Stata-generated tables and figures.

global ROOT "C:\Users\ngonzalezp\OneDrive - AIREF\Escritorio\Unempolyment_Benefits\ai_unemployment_analysis"
do "$ROOT\code\stata_empirical_analysis_plan.do"
