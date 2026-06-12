version 17
clear all
set more off

global ROOT  "C:/Users/ngonzalezp/OneDrive - AIREF/Escritorio/Unempolyment_Benefits/ai_unemployment_analysis"
global STOUT "$ROOT/output/stata"
global STDTA "$STOUT/data"
global STTAB "$STOUT/tables"
global STFIG "$STOUT/figures"
global STLOG "$STOUT/logs"
global SDID_EVENT_REPS 50

capture mkdir "$STOUT"
capture mkdir "$STDTA"
capture mkdir "$STTAB"
capture mkdir "$STFIG"
capture mkdir "$STLOG"

capture log close
log using "$STLOG/stata_sdid_event_focused.log", replace text

display as text "Focused sdid_event run started"
display as text "Stata version: " c(stata_version)

cap which sdid
if _rc ssc install sdid, replace

cap which sdid_event
if _rc ssc install sdid_event, replace

cap which distinct
if _rc ssc install distinct, replace

capture program drop keep_balanced_panel
program define keep_balanced_panel, rclass
    syntax, OUTCOME(name)

    keep if binary_sample == 1
    keep if !missing(`outcome', did_post, unit_id, ym)
    duplicates drop unit_id ym, force

    quietly levelsof ym, local(allmonths)
    local T : word count `allmonths'
    bysort unit_id: egen n_months_unit = count(ym)
    keep if n_months_unit == `T'

    bysort unit_id: egen ever_treated = max(treat_high)
    bysort unit_id: egen ever_zero = max(treat_zero)
    keep if ever_treated == 1 | ever_zero == 1

    quietly distinct unit_id if ever_treated == 1
    return scalar treated_units = r(ndistinct)
    quietly distinct unit_id if ever_zero == 1
    return scalar control_units = r(ndistinct)
    quietly distinct unit_id
    return scalar units = r(ndistinct)
    quietly distinct ym
    return scalar periods = r(ndistinct)
    quietly count
    return scalar nobs = r(N)
end

capture program drop export_event_results
program define export_event_results
    syntax, SAMPLE(string) SUBGROUP(string) TAG(string) OUTCOME(name) ESTIMATOR(string) EFFECTS(integer) PLACEBOS(integer) VCE(string)

    matrix H = e(H)
    clear
    svmat double H
    gen row = _n
    drop if row == 1

    gen int event_time = .
    replace event_time = _n - 1 if _n <= `effects'
    replace event_time = -(_n - `effects') if _n > `effects'

    capture confirm variable H1
    if _rc {
        capture rename h1 H1
        capture rename h2 H2
        capture rename h3 H3
        capture rename h4 H4
        capture rename h5 H5
    }
    rename H1 estimate
    capture rename H2 se
    capture rename H3 ci_low
    capture rename H4 ci_high
    capture rename H5 p_value
    capture confirm variable se
    if _rc gen se = .
    capture confirm variable ci_low
    if _rc gen ci_low = .
    capture confirm variable ci_high
    if _rc gen ci_high = .
    capture confirm variable p_value
    if _rc gen p_value = .

    gen str40 sample = "`sample'"
    gen str80 subgroup = "`subgroup'"
    gen str32 outcome = "`outcome'"
    gen str12 estimator = "`estimator'"
    gen str20 vce = "`vce'"
    order sample subgroup outcome estimator vce event_time estimate se ci_low ci_high p_value
    sort event_time

    local tablefile "$STTAB/sdid_event_focused_`estimator'_`tag'_`outcome'.csv"
    export delimited using "`tablefile'", replace

    local estimator_label = upper("`estimator'")
    if "`estimator'" == "sc" {
        local estimator_label "SC"
    }
    local graphfile "$STFIG/sdid_event_focused_`estimator'_`tag'_`outcome'.png"

    twoway ///
        (rarea ci_low ci_high event_time if !missing(ci_low, ci_high), color(gs12%45) lcolor(gs10)) ///
        (scatter estimate event_time, mcolor(navy) msymbol(circle) msize(small)) ///
        (line estimate event_time, lcolor(navy) lwidth(medthin)), ///
        yline(0, lcolor(gs9)) ///
        xline(0, lcolor(gs6) lpattern(dash)) ///
        title("`estimator_label' sdid_event: `sample'") ///
        subtitle("`subgroup'; outcome: `outcome'; vce(`vce')") ///
        note("Event 0 is first post month. Produced with Stata sdid_event.", size(vsmall)) ///
        xtitle("Periods relative to first post-treatment month") ///
        ytitle("Event-study estimate") ///
        legend(off) ///
        graphregion(color(white)) plotregion(color(white))
    graph export "`graphfile'", replace width(1600)

    c_local tablefile "`tablefile'"
    c_local graphfile "`graphfile'"
end

capture program drop run_sdid_event_one
program define run_sdid_event_one
    syntax, SAMPLE(string) SUBGROUP(string) TAG(string) OUTCOME(name) ESTIMATOR(string) VCE(string)

    preserve
        keep_balanced_panel, outcome(`outcome')
        local nobs = r(nobs)
        local units = r(units)
        local periods = r(periods)
        local treated_units = r(treated_units)
        local control_units = r(control_units)

        if `nobs' == 0 | `treated_units' == 0 | `control_units' == 0 | `periods' < 2 {
            post SDIDEVENT ("`sample'") ("`subgroup'") ("`outcome'") ("`estimator'") ("`vce'") ///
                (.) (.) (.) (`nobs') (`units') (`periods') (`treated_units') (`control_units') ///
                ("not_enough_balanced_data") ("") ("")
            restore
            exit
        }

        quietly distinct ym if post_sep2022 == 1
        local effects = r(ndistinct)
        quietly distinct ym if post_sep2022 == 0
        local placebos = r(ndistinct) - 1
        if `placebos' < 0 {
            local placebos = 0
        }

        display as text "Running sdid_event: sample=`sample' subgroup=`subgroup' outcome=`outcome' estimator=`estimator'"
        display as text "  nobs=`nobs' units=`units' periods=`periods' treated=`treated_units' controls=`control_units' effects=`effects' placebos=`placebos'"

        local vceopt "vce(off)"
        if "`vce'" == "placebo" {
            local vceopt "vce(placebo) brep($SDID_EVENT_REPS)"
        }
        capture noisily sdid_event `outcome' unit_id ym did_post, effects(`effects') placebo(`placebos') `vceopt' method(`estimator')
        if _rc {
            local rc = _rc
            display as error "sdid_event failed with rc=`rc'"
            post SDIDEVENT ("`sample'") ("`subgroup'") ("`outcome'") ("`estimator'") ("`vce'") ///
                (`effects') (`placebos') (.) (`nobs') (`units') (`periods') (`treated_units') (`control_units') ///
                ("sdid_event_rc_`rc'") ("") ("")
            restore
            exit
        }

        matrix H_att = e(H)
        local att = H_att[1,1]
        export_event_results, sample("`sample'") subgroup("`subgroup'") tag("`tag'") outcome(`outcome') estimator("`estimator'") effects(`effects') placebos(`placebos') vce("`vce'")
        post SDIDEVENT ("`sample'") ("`subgroup'") ("`outcome'") ("`estimator'") ("`vce'") ///
            (`effects') (`placebos') (`att') (`nobs') (`units') (`periods') (`treated_units') (`control_units') ///
            ("ok") ("`tablefile'") ("`graphfile'")
    restore
end

capture program drop make_gender_panel
program define make_gender_panel
    use "$STDTA/stata_master_clean.dta", clear
    keep if lower(dimension) == "gender"
    gen gender_group = category
    egen unit_id = group(cno4)
    save "$STDTA/panel_gender_cno4_for_sdid_event.dta", replace
end

capture program drop make_age3_panel
program define make_age3_panel
    use "$STDTA/panel_age_cno4.dta", clear
    gen str20 age3_group = ""
    replace age3_group = "<18 to 29" if inlist(age_group, "<18", "18-24", "25-29")
    replace age3_group = "30-39" if age_group == "30-39"
    replace age3_group = "40 to >44" if inlist(age_group, "40-44", ">44")
    keep if age3_group != ""

    collapse (sum) parados contratos ///
        (firstnm) exposure_nearest exposure_weighted exposure_rf treat_high treat_zero binary_sample post_sep2022 did_post cno2 cno1d event_time_sep2022, ///
        by(age3_group cno4 ym period occupation_title)

    gen ln_parados = ln(parados) if parados > 0
    gen ln_contratos = ln(contratos) if contratos > 0
    gen parados_contratos = parados / contratos if contratos > 0 & parados < .
    capture drop cno4_id cno2_id cno1_id unit_id
    egen cno4_id = group(cno4)
    egen cno2_id = group(cno2)
    egen cno1_id = group(cno1d)
    egen unit_id = group(cno4)
    save "$STDTA/panel_age3_cno4_for_sdid_event.dta", replace
end

tempfile sdidevent
postfile SDIDEVENT str40 sample str80 subgroup str32 outcome str12 estimator str20 vce ///
    double effects placebos att nobs units periods treated_units control_units ///
    str60 status str244 table_file str244 graph_file using `sdidevent', replace

* Total CNO4: actual sdid_event command with placebo inference.
use "$STDTA/panel_total_cno4.dta", clear
run_sdid_event_one, sample("total_cno4") subgroup("all") tag("total_cno4") outcome(ln_parados) estimator("sdid") vce("placebo")

* Province-CNO4: actual sdid_event command. Placebo inference is very slow on this 800k-row balanced panel.
use "$STDTA/panel_province_cno4.dta", clear
run_sdid_event_one, sample("province_cno4") subgroup("all") tag("province_cno4") outcome(ln_parados) estimator("sdid") vce("off")

* Age buckets requested in memo v2.
make_age3_panel
use "$STDTA/panel_age3_cno4_for_sdid_event.dta", clear
keep if age3_group == "<18 to 29"
run_sdid_event_one, sample("age3_cno4") subgroup("<18 to 29") tag("age3_lt18_to_29") outcome(ln_parados) estimator("sdid") vce("placebo")

use "$STDTA/panel_age3_cno4_for_sdid_event.dta", clear
keep if age3_group == "30-39"
run_sdid_event_one, sample("age3_cno4") subgroup("30-39") tag("age3_30_39") outcome(ln_parados) estimator("sdid") vce("placebo")

use "$STDTA/panel_age3_cno4_for_sdid_event.dta", clear
keep if age3_group == "40 to >44"
run_sdid_event_one, sample("age3_cno4") subgroup("40 to >44") tag("age3_40_to_gt44") outcome(ln_parados) estimator("sdid") vce("placebo")

* Gender subgroups.
make_gender_panel
use "$STDTA/panel_gender_cno4_for_sdid_event.dta", clear
keep if gender_group == "Hombre"
run_sdid_event_one, sample("gender_cno4") subgroup("Hombre") tag("gender_hombre") outcome(ln_parados) estimator("sdid") vce("placebo")

use "$STDTA/panel_gender_cno4_for_sdid_event.dta", clear
keep if gender_group == "Mujer"
run_sdid_event_one, sample("gender_cno4") subgroup("Mujer") tag("gender_mujer") outcome(ln_parados) estimator("sdid") vce("placebo")

postclose SDIDEVENT
use `sdidevent', clear
export delimited using "$STTAB/stata_sdid_event_focused_index.csv", replace
save "$STDTA/stata_sdid_event_focused_index.dta", replace

display as text "Focused sdid_event run completed."
display as text "Index:   $STTAB/stata_sdid_event_focused_index.csv"
display as text "Figures: $STFIG/sdid_event_focused_*.png"
display as text "Tables:  $STTAB/sdid_event_focused_*.csv"

log close
exit, clear
