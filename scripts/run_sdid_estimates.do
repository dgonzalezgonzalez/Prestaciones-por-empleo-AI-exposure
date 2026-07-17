version 19
clear all
set more off
set linesize 255
set seed 20260608

args reps
if "`reps'" == "" local reps 100

local intervention = tm(2022m9)
local seed_base = 20260608
local burgundy "#83082A"
local text "#404040"
local grid "#CCCCCC"
local ci "#E397A0"

graph set window fontface "Century Gothic"
graph set window fontfacemono "Century Gothic"
graph set window fontfacesans "Century Gothic"
graph set window fontfaceserif "Century Gothic"
graph set svg fontface "Century Gothic"
graph set svg fontfacemono "Century Gothic"
graph set svg fontfacesans "Century Gothic"
graph set svg fontfaceserif "Century Gothic"
graph set pdf fontface "Century Gothic"
graph set pdf fontfacemono "Century Gothic"
graph set pdf fontfacesans "Century Gothic"
graph set pdf fontfaceserif "Century Gothic"

cap mkdir "analysis/econometrics_outputs/sdid"
cap mkdir "analysis/econometrics_outputs/sdid/Graficos"
cap mkdir "analysis/econometrics_outputs/sdid/Graficos/unemployment"
cap mkdir "analysis/econometrics_outputs/sdid/Graficos/contracts"
capture log close
log using "analysis/econometrics_outputs/sdid/stata_sdid.log", replace text

which sdid
which sdid_event

import delimited using "data/processed/sepe_cno4_monthly_ai_exposure.csv", varnames(1) stringcols(_all) clear encoding(UTF-8)
keep if dimension == "total" & category == "Total" & gender == "Total"
replace cno4 = substr("0000" + cno4, length("0000" + cno4) - 3, 4)
encode cno4, gen(cno_id)
gen period_m = monthly(period, "YM")
format period_m %tm
gen post = period_m >= `intervention'

foreach v in contratos parados observed_exposure_rf observed_exposure_cosine_weighte observed_exposure_cosine_neare {
    destring `v', replace force
}
rename observed_exposure_rf exp_rf
rename observed_exposure_cosine_weighte exp_cosine_weighted
rename observed_exposure_cosine_neare exp_cosine_nearest

gen y_unemployment = ln(parados + 1)
gen y_contracts = contratos
keep cno4 cno_id period period_m post y_unemployment y_contracts exp_*
isid cno_id period_m
tempfile panel event_master
save `panel', replace
local have_events 0

tempname results
postfile `results' str20 outcome str24 exposure_measure str24 mode str24 control_rule ///
    double att std_error ci_low ci_high p_value int n_treated n_control n_periods n_pre_periods n_post_periods ///
    double q25 q75 int reps_completed using "analysis/econometrics_outputs/sdid/sdid_estimates.dta", replace

local outcomes "unemployment contracts"
local all_specs "cosine_weighted top_vs_zero cosine_nearest top_vs_zero rf top_vs_bottom cosine_weighted top_vs_bottom cosine_nearest top_vs_bottom"

foreach outcome of local outcomes {
    local y y_`outcome'
    local spec_count : word count `all_specs'
    forvalues s = 1(2)`spec_count' {
        local exposure : word `s' of `all_specs'
        local mode : word `=`s' + 1' of `all_specs'
        local expvar exp_`exposure'
        local control_rule "bottom quartile"
        if "`mode'" == "top_vs_zero" local control_rule "zero exposure"
        local stem sdid_`exposure'_`mode'
        local spec_seed = `seed_base' + `s' + cond("`outcome'" == "contracts", 1000, 0)

        use `panel', clear
        preserve
            bys cno_id: keep if _n == 1
            quietly summarize `expvar', detail
            local q25 = r(p25)
            local q75 = r(p75)
        restore

        if "`mode'" == "top_vs_zero" {
            keep if `expvar' >= `q75' | `expvar' == 0
        }
        else {
            keep if `expvar' >= `q75' | `expvar' <= `q25'
        }
        gen treated_unit = `expvar' >= `q75'
        gen D = treated_unit * post
        drop if missing(`y') | missing(D)
        bys cno_id: egen n_periods_unit = count(`y')
        quietly summarize n_periods_unit
        keep if n_periods_unit == r(max)
        drop n_periods_unit

        quietly levelsof cno_id if treated_unit == 1, local(treated_ids)
        quietly levelsof cno_id if treated_unit == 0, local(control_ids)
        local n_treated : word count `treated_ids'
        local n_control : word count `control_ids'
        quietly levelsof period_m, local(periods)
        local n_periods : word count `periods'
        quietly count if post == 0
        local n_pre_periods = r(N) / (`n_treated' + `n_control')
        quietly count if post == 1
        local n_post_periods = r(N) / (`n_treated' + `n_control')

        xtset cno_id period_m
        noisily di as text "Running Stata sdid: `outcome' `exposure' `mode'"
        noisily sdid `y' cno_id period_m D, vce(bootstrap) reps(`reps') seed(`spec_seed') method(sdid)
        local att = e(ATT)
        local se = e(se)
        local ci_low = e(ATT_l)
        local ci_high = e(ATT_r)
        local p_value = 2 * normal(-abs(`att' / `se'))
        post `results' ("`outcome'") ("`exposure'") ("`mode'") ("`control_rule'") ///
            (`att') (`se') (`ci_low') (`ci_high') (`p_value') (`n_treated') (`n_control') ///
            (`n_periods') (`n_pre_periods') (`n_post_periods') (`q25') (`q75') (`reps')

        matrix S = e(series)
        preserve
            clear
            svmat double S
            rename S1 period_m
            rename S2 synthetic_control_mean
            rename S3 treated_mean
            gen event_time = period_m - `intervention'
            gen str20 outcome = "`outcome'"
            gen str24 exposure_measure = "`exposure'"
            gen str24 mode = "`mode'"
            order outcome exposure_measure mode period_m event_time treated_mean synthetic_control_mean
            export excel using "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_levels_`exposure'_`mode'.xlsx", firstrow(variables) cell(D5) replace
            putexcel set "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_levels_`exposure'_`mode'.xlsx", modify
            putexcel B2 = "Synthetic difference-in-differences: treated and synthetic control"
            putexcel B3 = "Fuente: AIReF a partir de SEPE y Anthropic Economic Index."
            putexcel B4 = "Nota: Tratamiento definido por exposición alta a inteligencia artificial."
            twoway ///
                (line treated_mean event_time, lcolor("`burgundy'") lwidth(medthick)) ///
                (line synthetic_control_mean event_time, lcolor("`text'") lpattern(dash) lwidth(medthick)), ///
                xline(-0.5, lcolor("`text'") lpattern(dash)) ///
                ytitle("") xtitle("") ///
                legend(order(1 "Treated" 2 "Synthetic control") rows(1) position(6) ring(1) region(lcolor(none))) ///
                graphregion(color(white) lcolor(white) margin(zero)) plotregion(color(white) lcolor(white)) ///
                ylabel(, angle(horizontal) labsize(small) glcolor("`grid'") gmin gmax) ///
                xlabel(, labsize(small) nogrid) ///
                yscale(lcolor("`text'")) xscale(lcolor("`text'")) ///
                xsize(6) ysize(3) scheme(s2color) name(g_levels, replace)
            graph save "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_levels_`exposure'_`mode'.gph", replace
            graph export "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_levels_`exposure'_`mode'.png", width(1713) replace
            graph export "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_levels_`exposure'_`mode'.pdf", replace
            graph export "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_levels_`exposure'_`mode'.svg", replace
        restore

        noisily di as text "Running Stata sdid_event: `outcome' `exposure' `mode'"
        noisily sdid_event `y' cno_id period_m D, effects(60) placebo(all) vce(bootstrap) brep(`reps') method(sdid)
        matrix H = e(H)
        preserve
            clear
            local rows = rowsof(H)
            set obs `rows'
            gen str32 term = ""
            gen event_time = .
            gen estimate = .
            gen std_error = .
            gen ci_low = .
            gen ci_high = .
            gen switchers = .
            local rownames : rownames H
            forvalues i = 1/`rows' {
                local term : word `i' of `rownames'
                replace term = "`term'" in `i'
                replace estimate = H[`i', 1] in `i'
                replace std_error = H[`i', 2] in `i'
                replace ci_low = H[`i', 3] in `i'
                replace ci_high = H[`i', 4] in `i'
                replace switchers = H[`i', 5] in `i'
                if substr("`term'", 1, 7) == "Effect_" {
                    local et = real(substr("`term'", 8, .)) - 1
                    replace event_time = `et' in `i'
                }
                if substr("`term'", 1, 8) == "Placebo_" {
                    local et = -real(substr("`term'", 9, .)) - 1
                    replace event_time = `et' in `i'
                }
            }
            drop if term == "ATT"
            local obs = _N + 1
            set obs `obs'
            replace term = "Reference" in `obs'
            replace event_time = -1 in `obs'
            replace estimate = 0 in `obs'
            gen str20 outcome = "`outcome'"
            gen str24 exposure_measure = "`exposure'"
            gen str24 mode = "`mode'"
            gen int reps_completed = `reps'
            order outcome exposure_measure mode term event_time estimate std_error ci_low ci_high switchers reps_completed
            sort event_time
            tempfile current_event
            save `current_event', replace
            export excel using "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_eventstudy_`exposure'_`mode'.xlsx", firstrow(variables) cell(D5) replace
            putexcel set "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_eventstudy_`exposure'_`mode'.xlsx", modify
            putexcel B2 = "Synthetic difference-in-differences event study"
            putexcel B3 = "Fuente: AIReF a partir de SEPE y Anthropic Economic Index."
            putexcel B4 = "Nota: El periodo t=-1 se fija como referencia con estimación igual a cero."
            twoway ///
                (rcap ci_high ci_low event_time if event_time != -1, lcolor("`ci'")) ///
                (connected estimate event_time, sort lcolor("`burgundy'") mcolor("`burgundy'") msymbol(O) msize(vsmall) lwidth(medthick)), ///
                yline(0, lcolor("`text'")) xline(-0.5, lcolor("`text'") lpattern(dash)) ///
                ytitle("") xtitle("") ///
                legend(off) graphregion(color(white) lcolor(white) margin(zero)) plotregion(color(white) lcolor(white)) ///
                ylabel(, angle(horizontal) labsize(small) glcolor("`grid'") gmin gmax) ///
                xlabel(, labsize(small) nogrid) ///
                yscale(lcolor("`text'")) xscale(lcolor("`text'")) ///
                xsize(6) ysize(3) scheme(s2color) name(g_event, replace)
            graph save "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_eventstudy_`exposure'_`mode'.gph", replace
            graph export "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_eventstudy_`exposure'_`mode'.png", width(1713) replace
            graph export "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_eventstudy_`exposure'_`mode'.pdf", replace
            graph export "analysis/econometrics_outputs/sdid/Graficos/`outcome'/sdid_eventstudy_`exposure'_`mode'.svg", replace
            use `current_event', clear
            if `have_events' == 0 {
                save `event_master', replace
                local have_events 1
            }
            else {
                append using `event_master'
                save `event_master', replace
            }
        restore
    }
}

postclose `results'
use "analysis/econometrics_outputs/sdid/sdid_estimates.dta", clear
export delimited using "analysis/econometrics_outputs/sdid/sdid_estimates.csv", replace

file open tex using "analysis/econometrics_outputs/sdid/sdid_estimates.tex", write replace
file write tex "\begin{tabular}{llccc}" _n
file write tex "\toprule" _n
file write tex "Outcome & Specification & ATT & Treated & Control \\" _n
file write tex "\midrule" _n
quietly {
    forvalues i = 1/`=_N' {
        local outcome = outcome[`i']
        local spec = exposure_measure[`i'] + ", " + subinstr(mode[`i'], "_", " ", .)
        local att = string(att[`i'], "%9.3f")
        local se = string(std_error[`i'], "%9.3f")
        local nt = string(n_treated[`i'], "%9.0f")
        local nc = string(n_control[`i'], "%9.0f")
        local p = p_value[`i']
        local stars ""
        if `p' < 0.10 local stars "*"
        if `p' < 0.05 local stars "**"
        if `p' < 0.01 local stars "***"
        file write tex "`outcome' & `spec' & `att'`stars' & `nt' & `nc' \\" _n
        file write tex " &  & (`se') &  &  \\" _n
    }
}
file write tex "\bottomrule" _n
file write tex "\multicolumn{5}{l}{\footnotesize Notes: Standard errors in parentheses. *** p<0.01, ** p<0.05, * p<0.10.} \\" _n
file write tex "\end{tabular}" _n
file close tex
if `have_events' == 1 {
    use `event_master', clear
    sort outcome exposure_measure mode event_time
    export delimited using "analysis/econometrics_outputs/sdid/sdid_eventstudy_paths.csv", replace
}

log close
exit
