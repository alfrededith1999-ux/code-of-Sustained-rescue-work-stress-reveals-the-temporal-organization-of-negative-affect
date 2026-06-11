[# code-of-Sustained-rescue-work-stress-reveals-the-temporal-organization-of-negative-affect](https://github.com/alfrededith1999-ux/code-of-Sustained-rescue-work-stress-reveals-the-temporal-organization-of-negative-affect)
This repository contains analysis code used for the manuscript:

Sustained rescue-work stress reveals the temporal organization of negative affect

The study uses a two-stage longitudinal discovery–validation design to examine whether sustained rescue-work stress reveals reproducible temporal organization in negative affect. The analytic workflow covers data assembly, longitudinal measurement comparability, cross-time coupling analyses, resource–coping differentiation analyses, bounded-range analyses, boundary tests of anxiety-dominant switching, predictive validation, and robustness/sensitivity checks.

Repository status

This repository is intended to document the analytic workflow used in the manuscript. It includes Python and R scripts used during data preprocessing, feature construction, model estimation, robustness checking and result interpretation.

The repository does not include raw individual-level participant data. The individual-level longitudinal data are subject to confidentiality agreements, participant privacy protections and institutional data-governance restrictions. Full numerical reproduction of the manuscript results therefore requires authorized access to the corresponding derived analysis matrices or controlled-access data files.

Manuscript summary

The manuscript examines whether negative affective states, psychological resources and coping-related regulation show temporally organized patterns under sustained stress. Rescue work is treated as a naturalistic high-demand context rather than as an occupation-specific endpoint.

The main analytic logic is organized around four empirical layers:

Cross-time coupling of negative affective states
Depressive symptoms and anxiety symptoms were examined across comparable longitudinal windows. The primary analysis tested whether depressive symptoms at time t predicted anxiety symptoms at time t + 1, while adjusting for anxiety at time t. The reverse anxiety-to-depression pathway was also examined.
Resource–coping differentiation
Psychological resources and coping-related regulation were tested to determine whether they formed a single resource → coping → emotion mediation chain or occupied different temporal-functional positions.
Bounded-range coping effects
Linear and segmented models were compared to test whether coping-related regulation followed a simple “more is better” pattern or showed approximate bounded effective ranges.
Boundary and predictive validation
Anxiety-dominant switching was tested as a boundary hypothesis. Prediction models were used only as exploratory risk-stratification analyses, not as deployable clinical screening tools.
Data availability and privacy

Raw participant-level data are not included in this repository.

The study data include sensitive longitudinal psychological assessments from real-world rescue units. Therefore, individual-level data cannot be publicly released. De-identified derived feature matrices, codebooks and summary result tables may be made available under controlled-access conditions after institutional approval and completion of a data-use agreement.

The public code is provided to support transparency, workflow inspection and reproducibility of the analytic logic. Users should not expect this repository alone to reproduce all manuscript tables without access to the corresponding controlled-access analysis files.

Repository structure

The repository contains root-level scripts and several folders. Some file and folder names are retained in Chinese because they reflect the original analysis workflow.

Root-level workflow scripts
Script or file	Intended role in the workflow
Phase 1 发现（Discovery）.py	Discovery-stage analysis script used to identify candidate temporal-organization signals before validation.
build_master_table.py	Constructs or consolidates master analysis tables.
check_duplicates_id_wave.py	Checks duplicate participant-by-wave records.
python check_master.py	Checks the master table or master dataset structure.
rebuild_phq_and_rerun_baseline.py	Rebuilds PHQ-related variables and reruns baseline analyses.
rerun_baseline_with_pos_split.py	Reruns baseline analyses with positive-coping split or related operationalization.
step0_mi_gen_and_run.py	Step 0 workflow for multiple-imputation generation and execution.
step0_mi_lavaan_runner.py	Python runner for lavaan-related measurement or invariance analyses.
run_mi_lavaan.R	R script for multiple-imputation and lavaan-based modelling.
step1_build_derived_metrics.py	Builds derived longitudinal metrics and analytic features.
step2_crosslag_mediation_H1.py	Cross-lagged and mediation analyses for resource–coping–emotion relations.
step2_H1A_parallel_POS_NEG.py	Parallel positive/negative coping-related analyses.
step2_H1B_delta_change_mediation.py	Lagged change-score mediation analysis.
step2_dedup_for_gray.py	Deduplication step for gray-zone or risk-stratification analyses.
step3_ri_clpm_res_cop.py	RI-CLPM or resource–coping temporal modelling.
step3_transform_check.py	Checks raw, standardized and transformed scoring approaches.
step4_H3_anxiety_takeover.py	Boundary test for anxiety-dominant switching / anxiety-takeover hypothesis.
train_baseline_phq.py	Baseline prediction modelling using PHQ-related outcomes or features.
planC_multiscale_turnpos.py	Multiscale turning-point or bounded-range analysis under Plan C.
planC_traj_bayes_gmm_ml.py	Trajectory, Bayesian, GMM and machine-learning analyses under Plan C.
PlanC 风险排序 + 轨迹解释：灰区变窄（只灰不确定的人）.py	Risk ranking and trajectory interpretation script for gray-zone cases.
trajectory_risk_pipeline.py	Trajectory-informed risk pipeline.
traj_bayes_lgm_gmm_all_scales.py	Bayesian trajectory, latent growth and growth-mixture modelling across scales.
traj_bayes_lgm_gmm_all_scales_v2.py	Updated version of the trajectory/Bayesian/LGM/GMM workflow.
traj_curve_bayes_gmm_bic.py	Trajectory curve and BIC-based model comparison script.
debug_prob_collapse.py	Debugging script for probability collapse or prediction-output issues.
interpret_baseline.py	Interpretation of baseline model outputs.
interpret_baseline_robust_v2.py	Robust interpretation of baseline model outputs.
interpret_planC_results.py	Interpretation of Plan C modelling outputs.
s4_sonar_runner.py	Runner script for SONAR-style or staged analyses.
sonar_runner.py	SONAR-style analysis runner.
sonar_runner2.py	Alternative SONAR-style analysis runner.
sonar_policy_runner.py	Policy-oriented SONAR-style runner.
sonar_policy_runner_fast.py	Faster version of the policy-oriented SONAR-style runner.
shortform3_runner.py	Short-form or reduced-form analysis runner.
数据准备.py	Data preparation script.
机制分析.py	Mechanism-analysis script.
机制探索.py	Mechanism-exploration script.
灰区+发展路径主分析.py	Main analysis for gray-zone cases and developmental pathways.
计分审计.py	Scoring audit script.
路径地图.py	Pathway-mapping script.
量表可比性是地基纵向不变性分段等值.py	Longitudinal measurement comparability / segmented invariance analysis.
数据分析.zip	Compressed analysis materials or archived analysis files.
数据合并为长数据/

This folder contains wave-specific scripts used to merge raw questionnaire files into long-format data.

Script	Intended role
24年1季度.py	2024 Q1 data preparation / long-format construction.
24年二季度.py	2024 Q2 data preparation / long-format construction.
24年三季度.py	2024 Q3 data preparation / long-format construction.
24年四季度.py	2024 Q4 data preparation / long-format construction.
25年一季度.py	2025 Q1 data preparation / long-format construction.
25年二季度.py	2025 Q2 data preparation / long-format construction.
25年三季度.py	2025 Q3 data preparation / long-format construction.
25年四季度.py	2025 Q4 data preparation / long-format construction.
打印列名.py	Prints or exports raw column names for auditing and harmonization.
探索性因素分析.py	Exploratory factor analysis script.
数据体检.py	Data health check / quality-control script.
分组用的/

This folder contains grouping, participation and missingness-screening scripts.

Script	Intended role
参与概况表.py	Summarizes participation coverage and wave availability.
缺失率筛查.py	Screens missingness rates.
构建全勤队列和AB分组.py	Builds complete-attendance cohort and A/B grouping variables.
AB_bridge_分组统计.py	Summarizes A/B grouping statistics.
pyhon文件/

The folder name is preserved as uploaded. It contains database-construction, unit-repair, quality-control, holdout construction, high-risk label, visualization, CTSEM and sensitivity-analysis scripts.

Important files include:

Script	Intended role
build_psych_master_db.py	Builds the psychological master database.
build_psych_master_db - 副本.py	Backup/copy of the master database construction script.
check_psych_master_db.py	Checks the constructed psychological master database.
repair_and_diagnose_db_v2.py	Repairs and diagnoses database structure, version 2.
repair_and_diagnose_db_v3.py	Repairs and diagnoses database structure, version 3.
add_canon_fields_and_recheck.py	Adds canonical fields and rechecks database integrity.
list_all_columns_to_txt.py	Exports all column names for documentation.
make_dictionary_view.py	Builds a dictionary or data-dictionary view.
make_unit_filled_view.py	Builds unit-filled view for unit-level information.
make_bigunit_from_excelname_v3.py	Extracts broader unit information from Excel filenames.
make_bigunit_from_filename_view.py	Builds unit information from filename-derived views.
make_bigunit_from_folders_v2.py	Extracts unit information from folder structure.
make_bigunit_manual_v4.py to make_bigunit_manual_v5_3_1.py	Manual unit-coding or unit-repair scripts.
patch_24Q4_units_v1.py	Patches 2024 Q4 unit information.
patch_24Q4_units_v2_from_rawfiles.py	Patches 2024 Q4 unit information from raw files.
patch_24Q4_units_v2_1_from_rawfiles.py	Updated 2024 Q4 unit patch from raw files.
patch_24Q4_bigunit_by_context_v1.py	Patches broader unit labels by context.
patch_24Q4_bigunit_by_signature_v2.py	Patches broader unit labels by signature.
routeA_srcfile_bigunit_v1.py	Route A unit/backfill workflow using source-file unit information.
routeA_v2_backfill_by_metafile_seq.py	Route A backfill using metafile sequence.
routeA_v3_backfill_by_metaid_seq.py	Route A backfill using metadata ID sequence.
routeA_v4_personfill_keyword.py	Route A person-level filling using keyword rules.
routeA_v4_personfill_keyword_fast.py	Faster version of person-level keyword filling.
routeA_v5_backfill_bigunit_from_raw.py	Route A broader-unit backfill from raw data.
make_drop24_rekey_v6.py	Re-keying / repair workflow involving 2024 data.
qc_v6_drop24_rekey.py	Quality control for re-keying workflow.
qc_v6_24_only.py	Quality control restricted to 2024 data.
make_highrisk_labels_for_schemeC.py	Constructs high-risk labels for Scheme C prediction.
qc_highrisk_labels.py	Quality-control check for high-risk labels.
make_schemeC_unit_time_holdout.py	Constructs unit-time holdout split for Scheme C.
make_schemeC_unit_time_holdout_v2.py	Updated unit-time holdout split for Scheme C.
qc_schemeC_test_event_counts.py	Checks event counts in the Scheme C test set.
step0_efa_then_mi_auto.py	EFA followed by automated multiple-imputation workflow.
step0_pcq24_theory_mi_fix.py	Theory-guided PCQ24 multiple-imputation repair/fix script.
step3_submit_dt_ctsem_res_cop.py	CTSEM-related resource–coping dynamic-time analysis.
step4_H3_takeover_strong_v2.py	Updated anxiety-takeover / H3 boundary-test script.
psych_3d_viz_dash.py, psych_3d_viz_dash_v2.py	3D visualization dashboard scripts.
psych_3d_bigunit_dash_v4.py, psych_3d_bigunit_dash_v4_1.py	Unit-level or broader-unit visualization dashboards.
analyze_personkey_mismatch_types.py	Diagnoses participant-key mismatch types.
数据库修补.py	Database repair script.
检查并更正.py	Check-and-correct script.
312321.py, RouteA v3.py	Legacy or auxiliary scripts retained for workflow documentation.
Suggested analytic workflow

The scripts were developed iteratively. The exact execution order may depend on the local data structure and controlled-access files. A conservative workflow is:

Raw data inspection and wave-specific long-format construction
Use scripts in 数据合并为长数据/ to prepare wave-specific datasets, harmonize column names and inspect data quality.
Master database construction and unit repair
Use build_psych_master_db.py, check_psych_master_db.py, the make_bigunit_*, patch_*, routeA_*, and repair_and_diagnose_db_* scripts in pyhon文件/ to build and check the master longitudinal database.
Grouping, participation and missingness checks
Use 分组用的/ scripts and quality-control scripts such as 缺失率筛查.py, 参与概况表.py, qc_v6_drop24_rekey.py, and qc_highrisk_labels.py.
Derived metrics and scoring audit
Use step1_build_derived_metrics.py, 计分审计.py, 路径地图.py, and related scripts to build derived variables, scoring checks and analytic feature tables.
Measurement comparability and multiple-imputation / lavaan workflows
Use 量表可比性是地基纵向不变性分段等值.py, step0_mi_gen_and_run.py, step0_mi_lavaan_runner.py, and run_mi_lavaan.R.
Primary temporal analyses
Use step2_crosslag_mediation_H1.py, step2_H1A_parallel_POS_NEG.py, step2_H1B_delta_change_mediation.py, step3_ri_clpm_res_cop.py, and step3_transform_check.py for cross-time coupling, resource–coping differentiation and transformation checks.
Bounded-range and nonlinear analyses
Use planC_multiscale_turnpos.py and related Plan C scripts to examine approximate turning regions and bounded-range coping effects.
Boundary tests of anxiety-dominant switching
Use step4_H3_anxiety_takeover.py and pyhon文件/step4_H3_takeover_strong_v2.py.
Predictive validation and exploratory risk stratification
Use train_baseline_phq.py, trajectory_risk_pipeline.py, planC_traj_bayes_gmm_ml.py, PlanC 风险排序 + 轨迹解释：灰区变窄（只灰不确定的人）.py, and 灰区+发展路径主分析.py.
Trajectory and discovery-stage analyses
Use Phase 1 发现（Discovery）.py, traj_bayes_lgm_gmm_all_scales.py, traj_bayes_lgm_gmm_all_scales_v2.py, and traj_curve_bayes_gmm_bic.py.
Interpretation and reporting checks
Use interpret_baseline.py, interpret_baseline_robust_v2.py, interpret_planC_results.py, 机制分析.py, and 机制探索.py.
Software environment

The codebase contains primarily Python scripts and one R script. Because scripts were developed across multiple analytic stages, users should inspect individual import statements before execution.

Recommended baseline environment:

Python 3.10 or later
R 4.3 or later
Common Python packages likely required by the workflow:
pandas
numpy
scipy
statsmodels
scikit-learn
matplotlib
openpyxl
Common R packages likely required by the workflow:
lavaan
semTools
mice
tidyverse

A pinned requirements.txt, environment.yml and R sessionInfo() output should be archived with the final submission release.

Reproducibility notes

This repository supports inspection of the analysis workflow, but it is not a standalone public replication package because the raw participant-level data cannot be publicly shared.

To reproduce the manuscript results, users need:

access to the controlled or de-identified derived analysis files;
the wave-specific input tables corresponding to the manuscript analysis;
the correct local file paths or a project-level configuration file;
the Python and R package versions used in the final analytic environment;
the same inclusion, deduplication, scoring and wave-comparability rules described in the manuscript and Supplementary Information.

Some scripts may contain local path assumptions from the original analysis environment. These paths should be updated before running the scripts in a new environment.

Citation

If you use this repository, please cite the manuscript and the code repository.

Recommended repository archival step

Before journal submission or publication, create a versioned release, for example:

v1.0.0-submission

Then archive the release using Zenodo, Figshare, OSF or another DOI-minting repository. After the DOI is generated, update the Citation section and the manuscript Code availability statement.

License

A license file should be added before public reuse of this code. Until a formal license is specified, reuse permissions are not fully defined.

Contact

For questions about controlled-access data or analysis-code interpretation, please contact the corresponding author listed in the manuscript.
