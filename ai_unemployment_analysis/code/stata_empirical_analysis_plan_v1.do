version 17
clear all
set more off

* ================================================================
* Stata-first empirical analysis plan
* ================================================================
* This do-file constructs the analysis samples from the raw SEPE CSV
* and estimates:
*   1. TWFE models using reghdfe.
*   2. TWFE event-study models using reghdfe.
*   3. Synthetic DID using sdid.
*   4. Synthetic control using sdid, method(sc).
*   5. SDID/SC event-study graphs using sdid_event.
*
* Notes:
* - The SC estimator is run through sdid's documented method(sc) option.
*   The Stata Journal sdid article documents that method(sc) implements
*   the synthetic-control estimator in the same framework.
* - log(parados) and log(contratos) are left missing when the original
*   variable is zero or negative. The ratio outcome is missing when
*   contratos <= 0.
* - Province-CNO4 TWFE uses province-by-CNO4 unit FE, plus the requested
*   province-by-month FE. This is the natural panel FE for that sample.

* ----------------------------
* User controls
* ----------------------------

global ROOT "C:\Users\ngonzalezp\OneDrive - AIREF\Escritorio\Unempolyment_Benefits\ai_unemployment_analysis"
global RAW  "$ROOT\data\raw\sepe_cno4_monthly_ai_exposure.csv"

global STOUT "$ROOT\output\stata"
global STDTA "$STOUT\data"
global STTAB "$STOUT\tables"
global STFIG "$STOUT\figures"
global STLOG "$STOUT\logs"

* Inference for sdid/sc. For first development runs, set SDID_VCE to
* noinference. For final runs, placebo is the closest match to the
* previous synthetic exercises, but it can be slow on province-CNO4.
global SDID_VCE  "placebo"
global SDID_REPS 50
global SEED      20260604

* sdid_event uses vce(off|bootstrap|placebo) and brep().
global SDID_EVENT_VCE "placebo"

* Use all feasible event/placebo periods up to these caps.
global SDID_EVENT_MAX_EFFECTS 999
global SDID_EVENT_MAX_PLACEBOS 999

capture mkdir "$STOUT"
capture mkdir "$STDTA"
capture mkdir "$STTAB"
capture mkdir "$STFIG"
capture mkdir "$STLOG"

capture log close
log using "$STLOG\stata_empirical_analysis_plan.log", replace text

* ----------------------------
* Install required packages
* ----------------------------

cap which ftools
if _rc ssc install ftools, replace

cap which reghdfe
if _rc ssc install reghdfe, replace

cap which sdid
if _rc ssc install sdid, replace

cap which unique
if _rc ssc install unique, replace

cap which sdid_event
if _rc ssc install sdid_event, replace

* ----------------------------
* Helper programs
* ----------------------------

***** Data cleaning
capture program drop make_clean_master
program define make_clean_master
    import delimited using "$RAW", clear varnames(1) stringcols(_all) bindquote(strict) encoding("UTF-8")

    rename *, lower
    foreach v in contratos parados personas observed_exposure_rf observed_exposure_cosine_weight observed_exposure_cosine_weighte observed_exposure_cosine_nearest {
        capture confirm variable `v'
        if !_rc destring `v', replace force
    }

    replace period = trim(period)
    replace cno4 = trim(cno4)
    replace cno4 = substr("0000" + cno4, length("0000" + cno4) - 3, 4) if cno4 != ""
    keep if strlen(cno4) == 4

    gen ym = monthly(period, "YM")
    format ym %tm
    gen year_month = ym

    gen cno2 = substr(cno4, 1, 2)
    gen cno1d = substr(cno4, 1, 1)
    egen cno4_id = group(cno4), label
    egen cno2_id = group(cno2), label
    egen cno1_id = group(cno1d), label
    egen cno2_ym = group(cno2 ym), label
    egen cno1_ym = group(cno1d ym), label

    gen exposure_nearest = observed_exposure_cosine_nearest
    gen exposure_weighted = .
    capture confirm variable observed_exposure_cosine_weight
    if !_rc replace exposure_weighted = observed_exposure_cosine_weight
    capture confirm variable observed_exposure_cosine_weighte
    if !_rc replace exposure_weighted = observed_exposure_cosine_weighte if missing(exposure_weighted)
    gen exposure_rf = observed_exposure_rf
    gen exposure_10pp = exposure_nearest / 0.10

    gen ln_parados = ln(parados) if parados > 0
    gen ln_contratos = ln(contratos) if contratos > 0
    gen parados_contratos = parados / contratos if contratos > 0 & parados < .

    gen event_time_sep2022 = ym - tm(2022m9)
    gen post_sep2022 = ym > tm(2022m9)

    preserve
        keep if lower(dimension) == "total" & lower(category) == "total" & lower(gender) == "total"
        quietly summarize exposure_nearest, detail
        scalar p75_exposure = r(p75)
    restore

    gen treat_high = exposure_nearest > p75_exposure if exposure_nearest < .
    gen treat_zero = exposure_nearest == 0 if exposure_nearest < .
    gen binary_sample = treat_high == 1 | treat_zero == 1
    gen did_post = treat_high * post_sep2022

    gen byte exposure_group = .
    replace exposure_group = 0 if treat_zero == 1
    replace exposure_group = 1 if exposure_nearest > 0 & exposure_nearest <= p75_exposure
    replace exposure_group = 2 if treat_high == 1
    label define exposure_group 0 "zero" 1 "middle" 2 "high", replace
    label values exposure_group exposure_group

    label variable ln_parados "log(parados)"
    label variable ln_contratos "log(contratos)"
    label variable parados_contratos "parados / contratos"
    label variable did_post "High exposure x post Sep 2022"

    compress
    save "$STDTA\stata_master_clean.dta", replace

    file open meta using "$STTAB\stata_thresholds.txt", write replace
    file write meta "Main high exposure threshold (total panel p75): " %9.6f (p75_exposure) _n
    file write meta "Post definition: ym > September 2022, so post starts October 2022." _n
    file close meta
end


***** Make Panels
capture program drop make_panel_files
program define make_panel_files
    use "$STDTA\stata_master_clean.dta", clear
    keep if lower(dimension) == "total" & lower(category) == "total" & lower(gender) == "total"
    gen str40 sample_name = "total_cno4"
    gen str80 subgroup_name = "all"
    egen unit_id = group(cno4), label
    compress
    save "$STDTA\panel_total_cno4.dta", replace

    use "$STDTA\stata_master_clean.dta", clear
    keep if lower(dimension) == "province"
    gen province = category
    egen province_id = group(province), label
    egen province_ym = group(province ym), label
    egen unit_id = group(province cno4), label
    gen str40 sample_name = "province_cno4"
    gen str80 subgroup_name = "all"
    compress
    save "$STDTA\panel_province_cno4.dta", replace

    use "$STDTA\stata_master_clean.dta", clear
    keep if lower(dimension) == "age"
    gen age_group = category
    egen age_id = group(age_group), label
    egen unit_id = group(age_group cno4), label
    gen str40 sample_name = "age_cno4"
    gen str80 subgroup_name = age_group
    compress
    save "$STDTA\panel_age_cno4.dta", replace
end


***** Restrictions for sample
capture program drop add_restriction_flags
program define add_restriction_flags
    capture drop cno2_has_high cno2_has_zero cno2_identifying cno1_has_high cno1_has_zero cno1_identifying
    bysort cno2: egen cno2_has_high = max(treat_high) if binary_sample == 1
    bysort cno2: egen cno2_has_zero = max(treat_zero) if binary_sample == 1
    gen byte cno2_identifying = cno2_has_high == 1 & cno2_has_zero == 1

    bysort cno1d: egen cno1_has_high = max(treat_high) if binary_sample == 1
    bysort cno1d: egen cno1_has_zero = max(treat_zero) if binary_sample == 1
    gen byte cno1_identifying = cno1_has_high == 1 & cno1_has_zero == 1
end



***** Sample counts
capture program drop write_sample_counts
program define write_sample_counts
    tempfile counts
    postfile COUNTS str40 sample str80 subgroup double rows units periods high_units zero_units cno2_ident_units cno1_ident_units using `counts', replace

    use "$STDTA\panel_total_cno4.dta", clear
    add_restriction_flags
    quietly count
    local rows = r(N)
    quietly distinct unit_id
    local units = r(ndistinct)
    quietly distinct ym
    local periods = r(ndistinct)
    quietly distinct unit_id if treat_high == 1
    local high_units = r(ndistinct)
    quietly distinct unit_id if treat_zero == 1
    local zero_units = r(ndistinct)
    quietly distinct unit_id if cno2_identifying == 1
    local cno2_units = r(ndistinct)
    quietly distinct unit_id if cno1_identifying == 1
    local cno1_units = r(ndistinct)
    post COUNTS ("total_cno4") ("all") (`rows') (`units') (`periods') (`high_units') (`zero_units') (`cno2_units') (`cno1_units')

    use "$STDTA\panel_province_cno4.dta", clear
    add_restriction_flags
    quietly count
    local rows = r(N)
    quietly distinct unit_id
    local units = r(ndistinct)
    quietly distinct ym
    local periods = r(ndistinct)
    quietly distinct unit_id if treat_high == 1
    local high_units = r(ndistinct)
    quietly distinct unit_id if treat_zero == 1
    local zero_units = r(ndistinct)
    quietly distinct unit_id if cno2_identifying == 1
    local cno2_units = r(ndistinct)
    quietly distinct unit_id if cno1_identifying == 1
    local cno1_units = r(ndistinct)
    post COUNTS ("province_cno4") ("all") (`rows') (`units') (`periods') (`high_units') (`zero_units') (`cno2_units') (`cno1_units')

    use "$STDTA\panel_age_cno4.dta", clear
    levelsof age_group, local(ages)
    foreach a of local ages {
        preserve
            keep if age_group == "`a'"
            add_restriction_flags
            quietly count
            local rows = r(N)
            quietly distinct unit_id
            local units = r(ndistinct)
            quietly distinct ym
            local periods = r(ndistinct)
            quietly distinct unit_id if treat_high == 1
            local high_units = r(ndistinct)
            quietly distinct unit_id if treat_zero == 1
            local zero_units = r(ndistinct)
            quietly distinct unit_id if cno2_identifying == 1
            local cno2_units = r(ndistinct)
            quietly distinct unit_id if cno1_identifying == 1
            local cno1_units = r(ndistinct)
            post COUNTS ("age_cno4") ("`a'") (`rows') (`units') (`periods') (`high_units') (`zero_units') (`cno2_units') (`cno1_units')
        restore
    }
    postclose COUNTS
    use `counts', clear
    export delimited using "$STTAB\stata_sample_counts.csv", replace
end


***** Sample counts

capture program drop distinct
program define distinct, rclass
    syntax varname [if]
    preserve
        marksample touse
        keep if `touse'
        keep `varlist'
        duplicates drop
        quietly count
        return scalar ndistinct = r(N)
    restore
end


***** Catalog of specifications

capture program drop write_spec_catalog
program define write_spec_catalog
    file open specs using "$STTAB\stata_spec_catalog.csv", write replace
    file write specs "spec_id,estimator,panel,restriction,command_template,notes" _n
    file write specs `"spec1,TWFE,total/age,high_vs_zero,"reghdfe OUTCOME did_post if binary_sample == 1, absorb(unit_id ym cno2_ym) vce(cluster cno4_id)","CNO4 FE, month FE, CNO2-by-month FE.""' _n
    file write specs `"spec1,TWFE,province,high_vs_zero,"reghdfe OUTCOME did_post if binary_sample == 1, absorb(unit_id ym cno2_ym province_ym) vce(cluster cno4_id)","CNO4-by-province FE, month FE, CNO2-by-month FE, province-by-month FE.""' _n
    file write specs `"spec2,TWFE,total/age,high_vs_zero_and_cno2_identifying,"reghdfe OUTCOME did_post if binary_sample == 1 & cno2_identifying == 1, absorb(unit_id ym cno2_ym) vce(cluster cno4_id)","Spec1 restricted to CNO2 groups with both high- and zero-exposure occupations.""' _n
    file write specs `"spec2,TWFE,province,high_vs_zero_and_cno2_identifying,"reghdfe OUTCOME did_post if binary_sample == 1 & cno2_identifying == 1, absorb(unit_id ym cno2_ym province_ym) vce(cluster cno4_id)","Province version of Spec2.""' _n
    file write specs `"spec3,TWFE,total/age,high_vs_zero,"reghdfe OUTCOME did_post if binary_sample == 1, absorb(unit_id ym cno1_ym) vce(cluster cno4_id)","CNO4 FE, month FE, CNO1-by-month FE.""' _n
    file write specs `"spec3,TWFE,province,high_vs_zero,"reghdfe OUTCOME did_post if binary_sample == 1, absorb(unit_id ym cno1_ym province_ym) vce(cluster cno4_id)","CNO4-by-province FE, month FE, CNO1-by-month FE, province-by-month FE.""' _n
    file write specs `"spec4,TWFE,total/age,high_vs_zero_and_cno1_identifying,"reghdfe OUTCOME did_post if binary_sample == 1 & cno1_identifying == 1, absorb(unit_id ym cno1_ym) vce(cluster cno4_id)","Spec3 restricted to CNO1 groups with both high- and zero-exposure occupations.""' _n
    file write specs `"spec4,TWFE,province,high_vs_zero_and_cno1_identifying,"reghdfe OUTCOME did_post if binary_sample == 1 & cno1_identifying == 1, absorb(unit_id ym cno1_ym province_ym) vce(cluster cno4_id)","Province version of Spec4.""' _n
    file write specs `"sdid_spec1,SDID,total/province/each_age_group,high_vs_zero_unrestricted,"sdid OUTCOME unit_id ym did_post if binary_sample == 1, vce($SDID_VCE) method(sdid)","No CNO1/CNO2 identifying restriction is applied.""' _n
    file write specs `"sc_spec1,Synthetic control,total/province/each_age_group,high_vs_zero_unrestricted,"sdid OUTCOME unit_id ym did_post if binary_sample == 1, vce($SDID_VCE) method(sc)","Synthetic control is run through sdid method(sc).""' _n
    file write specs `"sdid_event_spec1,SDID event,total/province/each_age_group,high_vs_zero_unrestricted,"sdid_event OUTCOME unit_id ym did_post, vce($SDID_EVENT_VCE) brep($SDID_REPS) method(sdid)","Balanced high-vs-zero panel; no CNO1/CNO2 identifying restriction.""' _n
    file write specs `"sc_event_spec1,SC event,total/province/each_age_group,high_vs_zero_unrestricted,"sdid_event OUTCOME unit_id ym did_post, vce($SDID_EVENT_VCE) brep($SDID_REPS) method(sc)","Balanced high-vs-zero panel; no CNO1/CNO2 identifying restriction.""' _n
    file close specs
end


***** Descriptive Statistics

capture program drop descriptive_graphs
program define descriptive_graphs
    syntax, SAMPLE(string) SUBGROUP(string) TAG(string)
    preserve
        keep if exposure_group < .
        collapse (mean) ln_parados ln_contratos parados_contratos ///
            (sum) parados contratos, by(ym exposure_group)
        gen ln_total_parados = ln(parados) if parados > 0
        gen ln_total_contratos = ln(contratos) if contratos > 0
        export delimited using "$STTAB\stata_descriptives_`tag'.csv", replace

        foreach y in ln_parados ln_contratos parados_contratos {
            twoway ///
                (line `y' ym if exposure_group == 0, lcolor(navy) lwidth(medthick)) ///
                (line `y' ym if exposure_group == 1, lcolor(olive) lwidth(medthick)) ///
                (line `y' ym if exposure_group == 2, lcolor(maroon) lwidth(medthick)), ///
                xline(`=tm(2022m9)', lcolor(gs6) lpattern(dash)) ///
                title("`sample' `subgroup': `y' by exposure group") ///
                xtitle("") ytitle("Mean `y'") ///
                legend(order(1 "Zero" 2 "Middle" 3 "High") rows(1) region(lcolor(none))) ///
                graphregion(color(white)) plotregion(color(white))
            graph export "$STFIG\desc_`tag'_`y'.png", replace width(1600)
        }
    restore
end


******* TWFE STATIC estimates
capture program drop run_twfe_one
program define run_twfe_one
    syntax, SAMPLE(string) SUBGROUP(string) OUTCOME(name) SPEC(string) RESTRICT(string) ABSORB(string)

    quietly count if `restrict' & binary_sample == 1 & !missing(`outcome', did_post, cno4_id)
    local n0 = r(N)
    if `n0' == 0 {
        post TWFE ("`sample'") ("`subgroup'") ("`outcome'") ("`spec'") ("`restrict'") ("`absorb'") ///
            (.) (.) (.) (.) (.) (.) (.) (.) ("no_observations")
        exit
    }

    capture noisily reghdfe `outcome' did_post if `restrict' & binary_sample == 1, absorb(`absorb') vce(cluster cno4_id)
    if _rc {
        local rc = _rc
        post TWFE ("`sample'") ("`subgroup'") ("`outcome'") ("`spec'") ("`restrict'") ("`absorb'") ///
            (.) (.) (.) (.) (.) (.) (.) (.) ("reghdfe_rc_`rc'")
        exit
    }

    local beta = _b[did_post]
    local se = _se[did_post]
    local p = 2 * ttail(e(df_r), abs(`beta' / `se'))
    local cil = `beta' - invttail(e(df_r), 0.025) * `se'
    local cih = `beta' + invttail(e(df_r), 0.025) * `se'
    local pct = .
    if inlist("`outcome'", "ln_parados", "ln_contratos") {
        local pct = 100 * (exp(`beta') - 1)
    }
    post TWFE ("`sample'") ("`subgroup'") ("`outcome'") ("`spec'") ("`restrict'") ("`absorb'") ///
        (`beta') (`se') (`p') (`cil') (`cih') (`pct') (e(N)) (e(r2)) ("ok")
end


******* TWFE DYNAMIC estimates

capture program drop run_twfe_event_one
program define run_twfe_event_one
    syntax, SAMPLE(string) SUBGROUP(string) TAG(string) OUTCOME(name) SPECID(string) SPECLABEL(string) RESTRICT(string) ABSORB(string)

    preserve
        quietly count if `restrict' & binary_sample == 1 & !missing(`outcome', treat_high, event_time_sep2022, cno4_id)
        local n0 = r(N)
        if `n0' == 0 {
            post TWFEES ("`sample'") ("`subgroup'") ("`outcome'") ("`specid'") ("`speclabel'") ("`restrict'") ("`absorb'") ///
                (.) (.) (.) ("no_observations") ("")
            restore
            exit
        }

        levelsof event_time_sep2022 if `restrict' & binary_sample == 1 & !missing(`outcome'), local(events)
        local terms ""
        foreach e of local events {
            if `e' != -1 {
                if `e' < 0 {
                    local v "twfe_evt_m`=abs(`e')'"
                }
                else {
                    local v "twfe_evt_p`e'"
                }
                capture drop `v'
                gen byte `v' = treat_high == 1 & event_time_sep2022 == `e'
                local terms "`terms' `v'"
            }
        }

        capture noisily reghdfe `outcome' `terms' if `restrict' & binary_sample == 1, absorb(`absorb') vce(cluster cno4_id)
        if _rc {
            local rc = _rc
            post TWFEES ("`sample'") ("`subgroup'") ("`outcome'") ("`specid'") ("`speclabel'") ("`restrict'") ("`absorb'") ///
                (.) (.) (.) ("reghdfe_event_rc_`rc'") ("")
            restore
            exit
        }
        local model_n = e(N)
        local model_r2 = e(r2)
        local model_df = e(df_r)

        tempfile evtable
        postfile EV int event_time double estimate se ci95_low ci95_high p_value using `evtable', replace
        post EV (-1) (0) (.) (.) (.) (.)
        foreach e of local events {
            if `e' != -1 {
                if `e' < 0 {
                    local v "twfe_evt_m`=abs(`e')'"
                }
                else {
                    local v "twfe_evt_p`e'"
                }
                capture local beta = _b[`v']
                if _rc {
                    post EV (`e') (.) (.) (.) (.) (.)
                }
                else {
                    local beta = _b[`v']
                    local se = _se[`v']
                    local p = 2 * ttail(e(df_r), abs(`beta' / `se'))
                    local cil = `beta' - invttail(e(df_r), 0.025) * `se'
                    local cih = `beta' + invttail(e(df_r), 0.025) * `se'
                    post EV (`e') (`beta') (`se') (`cil') (`cih') (`p')
                }
            }
        }
        postclose EV

        local outfile "$STTAB\twfe_event_`tag'_`outcome'_`specid'.csv"
        local graphfile "$STFIG\twfe_event_`tag'_`outcome'_`specid'.png"
        use `evtable', clear
        sort event_time
        export delimited using "`outfile'", replace

        twoway ///
            (rcap ci95_low ci95_high event_time, lcolor(gs10)) ///
            (scatter estimate event_time, mcolor(maroon) msymbol(circle) msize(small)) ///
            (line estimate event_time, lcolor(maroon) lwidth(medthin)), ///
            yline(0, lcolor(gs9)) ///
            xline(0, lcolor(gs6) lpattern(dash)) ///
            title("TWFE event study `specid': `sample' `subgroup'") ///
            subtitle("Outcome: `outcome'") ///
            note("`specid' = `speclabel'. Reference month = -1.", size(vsmall)) ///
            xtitle("Months relative to September 2022") ///
            ytitle("Log-point or ratio effect") ///
            legend(off) ///
            graphregion(color(white)) plotregion(color(white))
        graph export "`graphfile'", replace width(1600)

        post TWFEES ("`sample'") ("`subgroup'") ("`outcome'") ("`specid'") ("`speclabel'") ("`restrict'") ("`absorb'") ///
            (`model_n') (`model_r2') (`model_df') ("ok") ("`graphfile'")
    restore
end



******* Balanced PANELs

capture program drop keep_balanced_panel
program define keep_balanced_panel, rclass
    syntax, OUTCOME(name) RESTRICT(string)

    keep if `restrict' & binary_sample == 1
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

******* STATIC SDID
capture program drop run_sdid_sc_one
program define run_sdid_sc_one
    syntax, SAMPLE(string) SUBGROUP(string) TAG(string) OUTCOME(name) ESTIMATOR(string) RESTRICTNAME(string) RESTRICT(string)

    preserve
        keep_balanced_panel, outcome(`outcome') restrict("`restrict'")
        local nobs = r(nobs)
        local units = r(units)
        local periods = r(periods)
        local treated_units = r(treated_units)
        local control_units = r(control_units)

        if `nobs' == 0 | `treated_units' == 0 | `control_units' == 0 | `periods' < 2 {
            post SDIDSC ("`sample'") ("`subgroup'") ("`outcome'") ("`estimator'") ("`restrictname'") ///
                (.) (.) (.) (.) (`nobs') (`units') (`periods') (`treated_units') (`control_units') ("not_enough_balanced_data")
            restore
            exit
        }

        local vceopt "vce($SDID_VCE)"
        if "$SDID_VCE" == "placebo" {
            if `control_units' <= `treated_units' {
                local vceopt "vce(noinference)"
            }
            else {
                local vceopt "vce(placebo) reps($SDID_REPS) seed($SEED)"
            }
        }
        if "$SDID_VCE" == "bootstrap" {
            local vceopt "vce(bootstrap) reps($SDID_REPS) seed($SEED)"
        }
        if "$SDID_VCE" == "jackknife" {
            local vceopt "vce(jackknife)"
        }

        capture noisily sdid `outcome' unit_id ym did_post, `vceopt' method(`estimator')
        if _rc {
            local rc = _rc
            post SDIDSC ("`sample'") ("`subgroup'") ("`outcome'") ("`estimator'") ("`restrictname'") ///
                (.) (.) (.) (.) (`nobs') (`units') (`periods') (`treated_units') (`control_units') ("sdid_rc_`rc'")
            restore
            exit
        }

        matrix b = e(b)
        scalar att = b[1,1]
        scalar se = .
        scalar p = .
        capture matrix V = e(V)
        if !_rc {
            scalar se = sqrt(V[1,1])
            if se < . {
                scalar p = 2 * (1 - normal(abs(att / se)))
            }
        }
        scalar pct = .
        if inlist("`outcome'", "ln_parados", "ln_contratos") {
            scalar pct = 100 * (exp(att) - 1)
        }
        post SDIDSC ("`sample'") ("`subgroup'") ("`outcome'") ("`estimator'") ("`restrictname'") ///
            (att) (se) (p) (pct) (`nobs') (`units') (`periods') (`treated_units') (`control_units') ("ok")
    restore
end




****** DYNAMIC SDID
capture program drop run_sdid_event_one
program define run_sdid_event_one
    syntax, SAMPLE(string) SUBGROUP(string) TAG(string) OUTCOME(name) ESTIMATOR(string)

    preserve
        keep_balanced_panel, outcome(`outcome') restrict("1")
        local nobs = r(nobs)
        local units = r(units)
        local periods = r(periods)
        local treated_units = r(treated_units)
        local control_units = r(control_units)

        if `nobs' == 0 | `treated_units' == 0 | `control_units' == 0 | `periods' < 2 {
            post SDIDEVENT ("`sample'") ("`subgroup'") ("`outcome'") ("`estimator'") ///
                (.) (.) (.) (.) (.) (`nobs') (`units') (`periods') (`treated_units') (`control_units') ("not_enough_balanced_data") ("")
            restore
            exit
        }

        quietly distinct ym if post_sep2022 == 1
        local effects = r(ndistinct)
        if `effects' > $SDID_EVENT_MAX_EFFECTS {
            local effects = $SDID_EVENT_MAX_EFFECTS
        }
        quietly distinct ym if post_sep2022 == 0
        local placebos = r(ndistinct) - 1
        if `placebos' < 0 {
            local placebos = 0
        }
        if `placebos' > $SDID_EVENT_MAX_PLACEBOS {
            local placebos = $SDID_EVENT_MAX_PLACEBOS
        }

        local event_vce "$SDID_EVENT_VCE"
        if "`event_vce'" == "placebo" & `control_units' <= `treated_units' {
            local event_vce "off"
        }
        set seed $SEED
		capture noisily sdid_event `outcome' unit_id ym did_post, effects(`effects') placebo(`placebos') vce(`event_vce') brep($SDID_REPS) method(`estimator')
        *capture noisily sdid_event `outcome' unit_id ym did_post, /*effects(`effects')*/ placebo(all) vce(`event_vce') brep($SDID_REPS) method(`estimator')
        if _rc {
            local rc = _rc
            post SDIDEVENT ("`sample'") ("`subgroup'") ("`outcome'") ("`estimator'") ///
                (`effects') (`placebos') (.) (.) (.) (`nobs') (`units') (`periods') (`treated_units') (`control_units') ("sdid_event_rc_`rc'") ("")
            restore
            exit
        }

        matrix H = e(H)
        tempfile evtable
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
        local event_spec_id "sdid_event_spec1"
        if "`estimator'" == "sc" {
            local event_spec_id "sc_event_spec1"
        }
        gen str32 spec_id = "`event_spec_id'"
        order sample subgroup outcome estimator spec_id event_time estimate se ci_low ci_high p_value
        sort event_time
        save `evtable', replace

        local estimator_label = upper("`estimator'")
        if "`estimator'" == "sc" {
            local estimator_label "SC"
        }
        local graphfile "$STFIG\sdid_event_`estimator'_`tag'_`outcome'.png"
        local tablefile "$STTAB\sdid_event_`estimator'_`tag'_`outcome'.csv"
        export delimited using "`tablefile'", replace

        twoway ///
            (rarea ci_low ci_high event_time if !missing(ci_low, ci_high), color(gs12%45) lcolor(gs10)) ///
            (scatter estimate event_time, mcolor(navy) msymbol(circle) msize(small)) ///
            (line estimate event_time, lcolor(navy) lwidth(medthin)), ///
            yline(0, lcolor(gs9)) ///
            xline(0, lcolor(gs6) lpattern(dash)) ///
            title("`estimator_label' event study: `sample' `subgroup'") ///
            subtitle("`event_spec_id'; outcome: `outcome'") ///
            note("Event 0 is first post month. No CNO1/CNO2 identifying restriction applied.", size(vsmall)) ///
            xtitle("Periods relative to first post-treatment month") ///
            ytitle("Event-study estimate") ///
            legend(off) ///
            graphregion(color(white)) plotregion(color(white))
        graph export "`graphfile'", replace width(1600)

        post SDIDEVENT ("`sample'") ("`subgroup'") ("`outcome'") ("`estimator'") ///
            (`effects') (`placebos') (.) (.) (.) (`nobs') (`units') (`periods') (`treated_units') (`control_units') ("ok") ("`graphfile'")
    restore
end


******* Check occupation list (4d) of highly exposed occup
capture program drop make_validation_outputs
program define make_validation_outputs
    use "$STDTA\panel_total_cno4.dta", clear
    keep cno4 occupation_title exposure_nearest exposure_weighted exposure_rf parados contratos treat_high
    bysort cno4: egen mean_parados = mean(parados)
    bysort cno4: egen mean_contratos = mean(contratos)
    bysort cno4: keep if _n == 1
    gsort -exposure_nearest
    gen exposure_rank = _n
    keep in 1/50
    export delimited using "$STTAB\stata_top50_exposure_occupations.csv", replace

    capture confirm file "$ROOT\output\tables\table_v1_exposure_match_validation_queue.csv"
    if !_rc {
        import delimited using "$ROOT\output\tables\table_v1_exposure_match_validation_queue.csv", clear varnames(1) encoding("UTF-8")
        contract match_review_flag
        rename _freq occupations
        export delimited using "$STTAB\stata_validation_queue_review_counts.csv", replace

        import delimited using "$ROOT\output\tables\table_v1_exposure_match_validation_queue.csv", clear varnames(1) encoding("UTF-8")
        keep if match_review_flag != "ok" | nearest_cosine_similarity < .60
        sort validation_priority_order rank_exposure
        export delimited using "$STTAB\stata_validation_queue_priority_review.csv", replace
    }
end

******* Descriptive helper to run
capture program drop run_descriptives
program define run_descriptives
    use "$STDTA\panel_total_cno4.dta", clear
    descriptive_graphs, sample("total_cno4") subgroup("all") tag("total_cno4")

    use "$STDTA\panel_province_cno4.dta", clear
    descriptive_graphs, sample("province_cno4") subgroup("all") tag("province_cno4")

    use "$STDTA\panel_age_cno4.dta", clear
    levelsof age_group, local(ages)
    foreach a of local ages {
        preserve
            keep if age_group == "`a'"
            local tag = subinstr("`a'", "<", "lt", .)
            local tag = subinstr("`tag'", ">", "gt", .)
            local tag = subinstr("`tag'", "-", "_", .)
            descriptive_graphs_for_loaded_sample, sample("age_cno4") subgroup("`a'") tag("age_cno4_`tag'")
        restore
    }
end

******* TWFE STATIC estimate helper to run
capture program drop run_all_twfe
program define run_all_twfe
    tempfile twferes
    postfile TWFE str40 sample str80 subgroup str32 outcome str32 spec str80 restriction str120 absorb ///
        double beta se p_value ci95_low ci95_high effect_pct nobs r2 str40 status using `twferes', replace

    use "$STDTA\panel_total_cno4.dta", clear
    add_restriction_flags
    foreach y in ln_parados ln_contratos parados_contratos {
        run_twfe_one, sample("total_cno4") subgroup("all") outcome(`y') spec("spec1_cno2_month") ///
            restrict("1") absorb("unit_id ym cno2_ym")
        run_twfe_one, sample("total_cno4") subgroup("all") outcome(`y') spec("spec2_cno2_month_restricted") ///
            restrict("cno2_identifying == 1") absorb("unit_id ym cno2_ym")
        run_twfe_one, sample("total_cno4") subgroup("all") outcome(`y') spec("spec3_cno1_month") ///
            restrict("1") absorb("unit_id ym cno1_ym")
        run_twfe_one, sample("total_cno4") subgroup("all") outcome(`y') spec("spec4_cno1_month_restricted") ///
            restrict("cno1_identifying == 1") absorb("unit_id ym cno1_ym")
    }

    use "$STDTA\panel_province_cno4.dta", clear
    add_restriction_flags
    foreach y in ln_parados ln_contratos parados_contratos {
        run_twfe_one, sample("province_cno4") subgroup("all") outcome(`y') spec("spec1_cno2_month_plus_province_month") ///
            restrict("1") absorb("unit_id ym cno2_ym province_ym")
        run_twfe_one, sample("province_cno4") subgroup("all") outcome(`y') spec("spec2_cno2_month_plus_province_month_restricted") ///
            restrict("cno2_identifying == 1") absorb("unit_id ym cno2_ym province_ym")
        run_twfe_one, sample("province_cno4") subgroup("all") outcome(`y') spec("spec3_cno1_month_plus_province_month") ///
            restrict("1") absorb("unit_id ym cno1_ym province_ym")
        run_twfe_one, sample("province_cno4") subgroup("all") outcome(`y') spec("spec4_cno1_month_plus_province_month_restricted") ///
            restrict("cno1_identifying == 1") absorb("unit_id ym cno1_ym province_ym")
    }

    use "$STDTA\panel_age_cno4.dta", clear
    levelsof age_group, local(ages)
    foreach a of local ages {
        preserve
            keep if age_group == "`a'"
            add_restriction_flags
            foreach y in ln_parados ln_contratos parados_contratos {
                run_twfe_one, sample("age_cno4") subgroup("`a'") outcome(`y') spec("spec1_cno2_month") ///
                    restrict("1") absorb("cno4_id ym cno2_ym")
                run_twfe_one, sample("age_cno4") subgroup("`a'") outcome(`y') spec("spec2_cno2_month_restricted") ///
                    restrict("cno2_identifying == 1") absorb("cno4_id ym cno2_ym")
                run_twfe_one, sample("age_cno4") subgroup("`a'") outcome(`y') spec("spec3_cno1_month") ///
                    restrict("1") absorb("cno4_id ym cno1_ym")
                run_twfe_one, sample("age_cno4") subgroup("`a'") outcome(`y') spec("spec4_cno1_month_restricted") ///
                    restrict("cno1_identifying == 1") absorb("cno4_id ym cno1_ym")
            }
        restore
    }

    postclose TWFE
    use `twferes', clear
    export delimited using "$STTAB\stata_twfe_reghdfe_results.csv", replace
    save "$STDTA\stata_twfe_reghdfe_results.dta", replace
end


******* TWFE DYNAMIC estimate helper to run

capture program drop run_all_twfe_events
program define run_all_twfe_events
    tempfile twfees
    postfile TWFEES str40 sample str80 subgroup str32 outcome str12 spec_id str120 spec_label str80 restriction str120 absorb ///
        double nobs r2 df_r str60 status str244 graph_file using `twfees', replace

    use "$STDTA\panel_total_cno4.dta", clear
    add_restriction_flags
    foreach y in ln_parados ln_contratos parados_contratos {
        run_twfe_event_one, sample("total_cno4") subgroup("all") tag("total_cno4") outcome(`y') specid("spec1") ///
            speclabel("CNO4 FE + month FE + CNO2-by-month FE") ///
            restrict("1") absorb("unit_id ym cno2_ym")
        run_twfe_event_one, sample("total_cno4") subgroup("all") tag("total_cno4") outcome(`y') specid("spec2") ///
            speclabel("Spec1 restricted to CNO2 groups with high and zero exposure") ///
            restrict("cno2_identifying == 1") absorb("unit_id ym cno2_ym")
        run_twfe_event_one, sample("total_cno4") subgroup("all") tag("total_cno4") outcome(`y') specid("spec3") ///
            speclabel("CNO4 FE + month FE + CNO1-by-month FE") ///
            restrict("1") absorb("unit_id ym cno1_ym")
        run_twfe_event_one, sample("total_cno4") subgroup("all") tag("total_cno4") outcome(`y') specid("spec4") ///
            speclabel("Spec3 restricted to CNO1 groups with high and zero exposure") ///
            restrict("cno1_identifying == 1") absorb("unit_id ym cno1_ym")
    }

    use "$STDTA\panel_province_cno4.dta", clear
    add_restriction_flags
    foreach y in ln_parados ln_contratos parados_contratos {
        run_twfe_event_one, sample("province_cno4") subgroup("all") tag("province_cno4") outcome(`y') specid("spec1") ///
            speclabel("CNO4-province FE + month FE + CNO2-by-month FE + province-by-month FE") ///
            restrict("1") absorb("unit_id ym cno2_ym province_ym")
        run_twfe_event_one, sample("province_cno4") subgroup("all") tag("province_cno4") outcome(`y') specid("spec2") ///
            speclabel("Spec1 restricted to CNO2 groups with high and zero exposure") ///
            restrict("cno2_identifying == 1") absorb("unit_id ym cno2_ym province_ym")
        run_twfe_event_one, sample("province_cno4") subgroup("all") tag("province_cno4") outcome(`y') specid("spec3") ///
            speclabel("CNO4-province FE + month FE + CNO1-by-month FE + province-by-month FE") ///
            restrict("1") absorb("unit_id ym cno1_ym province_ym")
        run_twfe_event_one, sample("province_cno4") subgroup("all") tag("province_cno4") outcome(`y') specid("spec4") ///
            speclabel("Spec3 restricted to CNO1 groups with high and zero exposure") ///
            restrict("cno1_identifying == 1") absorb("unit_id ym cno1_ym province_ym")
    }

    use "$STDTA\panel_age_cno4.dta", clear
    levelsof age_group, local(ages)
    foreach a of local ages {
        preserve
            keep if age_group == "`a'"
            local tag = subinstr("`a'", "<", "lt", .)
            local tag = subinstr("`tag'", ">", "gt", .)
            local tag = subinstr("`tag'", "-", "_", .)
            add_restriction_flags
            foreach y in ln_parados ln_contratos parados_contratos {
                run_twfe_event_one, sample("age_cno4") subgroup("`a'") tag("age_cno4_`tag'") outcome(`y') specid("spec1") ///
                    speclabel("CNO4 FE + month FE + CNO2-by-month FE") ///
                    restrict("1") absorb("cno4_id ym cno2_ym")
                run_twfe_event_one, sample("age_cno4") subgroup("`a'") tag("age_cno4_`tag'") outcome(`y') specid("spec2") ///
                    speclabel("Spec1 restricted to CNO2 groups with high and zero exposure") ///
                    restrict("cno2_identifying == 1") absorb("cno4_id ym cno2_ym")
                run_twfe_event_one, sample("age_cno4") subgroup("`a'") tag("age_cno4_`tag'") outcome(`y') specid("spec3") ///
                    speclabel("CNO4 FE + month FE + CNO1-by-month FE") ///
                    restrict("1") absorb("cno4_id ym cno1_ym")
                run_twfe_event_one, sample("age_cno4") subgroup("`a'") tag("age_cno4_`tag'") outcome(`y') specid("spec4") ///
                    speclabel("Spec3 restricted to CNO1 groups with high and zero exposure") ///
                    restrict("cno1_identifying == 1") absorb("cno4_id ym cno1_ym")
            }
        restore
    }

    postclose TWFEES
    use `twfees', clear
    export delimited using "$STTAB\stata_twfe_event_study_index.csv", replace
    save "$STDTA\stata_twfe_event_study_index.dta", replace
end



******* SDID STATIC estimate helper to run

capture program drop run_all_sdid_sc
program define run_all_sdid_sc
    tempfile sdidres
    postfile SDIDSC str40 sample str80 subgroup str32 outcome str12 estimator str32 restriction ///
        double att se p_value effect_pct nobs units periods treated_units control_units str60 status using `sdidres', replace

    use "$STDTA\panel_total_cno4.dta", clear
    foreach y in ln_parados ln_contratos parados_contratos {
        foreach m in sdid sc {
            run_sdid_sc_one, sample("total_cno4") subgroup("all") tag("total_cno4") outcome(`y') estimator("`m'") restrictname("unrestricted_high_vs_zero") restrict("1")
        }
    }

    use "$STDTA\panel_province_cno4.dta", clear
    foreach y in ln_parados ln_contratos parados_contratos {
        foreach m in sdid sc {
            run_sdid_sc_one, sample("province_cno4") subgroup("all") tag("province_cno4") outcome(`y') estimator("`m'") restrictname("unrestricted_high_vs_zero") restrict("1")
        }
    }

    use "$STDTA\panel_age_cno4.dta", clear
    levelsof age_group, local(ages)
    foreach a of local ages {
        preserve
            keep if age_group == "`a'"
            local tag = subinstr("`a'", "<", "lt", .)
            local tag = subinstr("`tag'", ">", "gt", .)
            local tag = subinstr("`tag'", "-", "_", .)
            foreach y in ln_parados ln_contratos parados_contratos {
                foreach m in sdid sc {
                    run_sdid_sc_one, sample("age_cno4") subgroup("`a'") tag("age_cno4_`tag'") outcome(`y') estimator("`m'") restrictname("unrestricted_high_vs_zero") restrict("1")
                }
            }
        restore
    }

    postclose SDIDSC
    use `sdidres', clear
    export delimited using "$STTAB\stata_sdid_sc_results.csv", replace
    save "$STDTA\stata_sdid_sc_results.dta", replace
end


******* SDID DYNAMIC estimate helper to run

capture program drop run_all_sdid_events
program define run_all_sdid_events
    tempfile sdidevent
    postfile SDIDEVENT str40 sample str80 subgroup str32 outcome str12 estimator ///
        double effects placebos att se p_value nobs units periods treated_units control_units str60 status str244 graph_file using `sdidevent', replace

    use "$STDTA\panel_total_cno4.dta", clear
    foreach y in ln_parados ln_contratos parados_contratos {
        foreach m in sdid sc {
            run_sdid_event_one, sample("total_cno4") subgroup("all") tag("total_cno4") outcome(`y') estimator("`m'")
        }
    }

    use "$STDTA\panel_province_cno4.dta", clear
    foreach y in ln_parados ln_contratos parados_contratos {
        foreach m in sdid sc {
            run_sdid_event_one, sample("province_cno4") subgroup("all") tag("province_cno4") outcome(`y') estimator("`m'")
        }
    }

    use "$STDTA\panel_age_cno4.dta", clear
    levelsof age_group, local(ages)
    foreach a of local ages {
        preserve
            keep if age_group == "`a'"
            local tag = subinstr("`a'", "<", "lt", .)
            local tag = subinstr("`tag'", ">", "gt", .)
            local tag = subinstr("`tag'", "-", "_", .)
            foreach y in ln_parados ln_contratos parados_contratos {
                foreach m in sdid sc {
                    run_sdid_event_one, sample("age_cno4") subgroup("`a'") tag("age_cno4_`tag'") outcome(`y') estimator("`m'")
                }
            }
        restore
    }

    postclose SDIDEVENT
    use `sdidevent', clear
    export delimited using "$STTAB\stata_sdid_event_index.csv", replace
    save "$STDTA\stata_sdid_event_index.dta", replace
end

* ----------------------------
* Execute analysis
* ----------------------------

make_clean_master
make_panel_files
write_sample_counts
write_spec_catalog
make_validation_outputs
run_descriptives
run_all_twfe
run_all_twfe_events
run_all_sdid_sc
run_all_sdid_events

display as text "Stata empirical analysis completed."
display as text "Tables:  $STTAB"
display as text "Figures: $STFIG"
display as text "Log:     $STLOG\stata_empirical_analysis_plan.log"

log close
