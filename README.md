# Code for *Sustained rescue-work stress reveals the temporal organization of negative affect*
This repository contains the analysis-code archive for the manuscript **Sustained rescue-work stress reveals the temporal organization of negative affect**. The manuscript uses a two-stage longitudinal discovery–validation design to examine whether sustained rescue-work stress reveals temporally organized patterns in negative affect, psychological resources and coping-related regulation.
> **Important**: this repository documents the analytic workflow. It does not contain raw participant-level data. Full numerical reproduction requires authorized access to the corresponding controlled-access or derived analysis files.
## Static code-audit summary
The uploaded repository archive was inspected script-by-script before this README was drafted. This was a static source-code inspection, not a re-execution of the analyses.
- Python scripts inspected: **106**.
- R scripts inspected: **1** (`run_mi_lavaan.R`).
- Python syntax/AST parsing: **106/106 scripts parsed successfully**.
- Non-analysis or archival items detected: `step0_mi_lavaan_runner.py.bak`, `数据分析.zip`, `pyhon文件/python` (empty file), and `pyhon文件/312321.py` (non-manuscript Gantt-chart utility). These should be moved to a `legacy/` folder or omitted from a formal release if a cleaner submission archive is desired.
- Many scripts contain local Windows path constants from the original working environment. Users must edit file paths or add a project-level configuration file before execution.
## What this repository is for
The repository is intended to support transparency by showing how the manuscript analyses were implemented or developed. It covers the following major layers:
1. Wave-specific questionnaire harmonization and long-format/wide-format construction.
2. Master database construction, participant linkage, deduplication, unit-label repair and quality control.
3. Longitudinal measurement comparability / measurement-invariance checks.
4. Derived feature construction and scoring audits.
5. Cross-time coupling and lagged mediation / resource–coping differentiation analyses.
6. Within-person, RI-CLPM and continuous-time resource–coping analyses.
7. Bounded-range / approximate turning-region analyses.
8. Boundary tests of anxiety-dominant switching.
9. Prediction, calibration, decision-curve and risk-stratification analyses.
10. Exploratory trajectory, Bayesian growth, GMM and gray-zone analyses.
## What this repository is not
- It is **not** a public raw-data release.
- It is **not** a one-click fully reproducible package unless the controlled-access analysis matrices and local path configuration are supplied.
- It should not be interpreted as a guarantee that every exploratory or legacy script contributed directly to a final manuscript table. The tables below mark script roles conservatively.
## Data availability and privacy
Raw individual-level longitudinal data are not included because the data contain sensitive psychological assessments from real-world rescue units and are subject to confidentiality agreements, participant privacy protections and institutional data-governance restrictions.
To support reproducibility, de-identified derived analysis matrices, codebooks and summary result tables needed to reproduce the main analyses may be made available under controlled-access conditions after institutional approval and completion of a data-use agreement. The public code documents the analytic workflow but does not contain raw participant-level records.
## Suggested workflow map
| Manuscript / Supplementary component | Main scripts or folders | Notes |
|---|---|---|
| Sample construction, raw-wave harmonization and data preparation | `数据合并为长数据/`, `build_master_table.py`, `数据准备.py`, `pyhon文件/build_psych_master_db.py` | Wave-specific scripts harmonize questionnaire exports and build analysis-ready wide/master tables. |
| Deduplication, participant linkage, unit repair and database QC | `check_duplicates_id_wave.py`, `pyhon文件/check_psych_master_db.py`, `pyhon文件/repair_and_diagnose_db_v*.py`, `pyhon文件/routeA_*`, `pyhon文件/patch_*`, `pyhon文件/make_bigunit_*` | These scripts repair IDs, person keys, unit labels and database integrity. |
| Measurement comparability / longitudinal invariance | `量表可比性是地基纵向不变性分段等值.py`, `step0_mi_gen_and_run.py`, `step0_mi_lavaan_runner.py`, `run_mi_lavaan.R` | Generates Mplus or lavaan workflows for configural, metric and scalar/threshold comparability checks. |
| Derived metrics and scoring audit | `step1_build_derived_metrics.py`, `计分审计.py`, `路径地图.py` | Creates derived features, score checks and path/trajectory-ready variables. |
| Discovery-stage signal generation | `Phase 1 发现（Discovery）.py`, `traj_bayes_lgm_gmm_all_scales*.py`, `traj_curve_bayes_gmm_bic.py` | Exploratory/discovery analyses used to generate validation propositions. |
| Cross-time coupling and resource–coping differentiation | `step2_crosslag_mediation_H1.py`, `step2_H1A_parallel_POS_NEG.py`, `step2_H1B_delta_change_mediation.py`, `step3_ri_clpm_res_cop.py`, `pyhon文件/step3_submit_dt_ctsem_res_cop.py` | Lagged mediation, positive/negative coping split, RI-CLPM and continuous-time auxiliary analyses. |
| Transformation and scoring robustness | `step3_transform_check.py`, `rebuild_phq_and_rerun_baseline.py`, `rerun_baseline_with_pos_split.py` | Checks raw, standardized, transformed and alternative operationalizations. |
| Bounded-range / approximate turning-region analyses | `planC_multiscale_turnpos.py`, `planC_traj_bayes_gmm_ml.py`, `PlanC 风险排序 + 轨迹解释：灰区变窄（只灰不确定的人）.py` | Tests nonlinear/segmented patterns and risk/trajectory layers. Some scripts also include prediction components. |
| Boundary test: anxiety-dominant switching | `step4_H3_anxiety_takeover.py`, `pyhon文件/step4_H3_takeover_strong_v2.py` | Tests whether anxiety-to-depression paths strengthen under high or increasing stress contexts. |
| Predictive validation and interpretation | `train_baseline_phq.py`, `trajectory_risk_pipeline.py`, `interpret_baseline.py`, `interpret_baseline_robust_v2.py`, `interpret_planC_results.py`, `pyhon文件/make_highrisk_labels_for_schemeC.py`, `pyhon文件/make_schemeC_unit_time_holdout_v2.py` | Prediction, group/unit holdout, risk labels, calibration-style summaries and interpretation outputs. |
| Exploratory visualization / dashboards | `pyhon文件/psych_3d_*`, `s4_sonar_runner.py`, `sonar_*`, `shortform3_runner.py` | Auxiliary visualization and exploratory/staged analysis scripts; not required for core reproduction. |
## Software environment
The repository is mostly Python with one R/lavaan script. A precise frozen environment should be created for final archival release. Based on static import inspection, the main dependencies include:
```text
pandas
numpy
scipy
statsmodels
scikit-learn
matplotlib
openpyxl
rapidfuzz
factor_analyzer
pymc
plotly
dash
umap-learn
tqdm
joblib
python-dateutil
```
R dependencies for `run_mi_lavaan.R` include:
```text
lavaan
semTools
readr
```
Recommended baseline versions for a clean archival release:
- Python 3.10 or later.
- R 4.3 or later.
- Mplus is optional and only needed for Mplus-based invariance workflows. The lavaan route can be used as an open alternative.
## Running the code
Because the raw and derived data files are not included, these commands are templates only. Update paths before running.
```bash
# Example: measurement comparability using the Python -> R/lavaan workflow
python step0_mi_lavaan_runner.py --data_dir path/to/wave/files --out_dir outputs/measurement --rscript path/to/Rscript --estimator MLR

# Example: cross-lagged mediation / resource-coping analysis
python step2_crosslag_mediation_H1.py --help

# Example: anxiety-dominant switching boundary test
python step4_H3_anxiety_takeover.py --help
```
Several scripts were originally written as local one-off analysis scripts and may not expose a complete command-line interface. When a script contains hard-coded paths, edit the configuration section at the top of the script or refactor it to use command-line arguments before reuse.
## Script inventory from static inspection
The following inventory is based on direct inspection of the uploaded source archive. `LOC` indicates non-empty, non-comment lines. `Syntax` indicates whether Python AST parsing succeeded.
### `root/`
| File | Lines | LOC | Syntax | Role / first-line docstring |
|---|---:|---:|---|---|
| `Phase 1 发现（Discovery）.py` | 765 | 578 | ok | PHASE 1 — 发现（Discovery） |
| `PlanC 风险排序 + 轨迹解释：灰区变窄（只灰不确定的人）.py` | 471 | 336 | ok | PlanC 风险排序 + 轨迹解释：灰区变窄（只灰不确定的人） |
| `build_master_table.py` | 449 | 308 | ok | 一键：读取多季度 Excel（不同sheet/不同命名） -> 统一列名 -> 合并 -> 计算总分 -> 生成推进标签 |
| `check_duplicates_id_wave.py` | 24 | 17 | ok | Duplicate ID-wave check |
| `debug_prob_collapse.py` | 123 | 91 | ok | Prediction probability-collapse diagnostic |
| `interpret_baseline.py` | 276 | 198 | ok | 解读 baseline_phq_preds.csv / baseline_phq_dca.csv（标准化输出 + PASS/FAIL + Excel报告） |
| `interpret_baseline_robust_v2.py` | 357 | 290 | ok | 鲁棒解读 baseline preds/dca（v2修复版） |
| `interpret_planC_results.py` | 343 | 270 | ok | 解读方案C（或任意）模型输出的 preds/dca 结果（鲁棒版） |
| `planC_multiscale_turnpos.py` | 545 | 382 | ok | Multiscale turning-point / turn-positive prediction analysis |
| `planC_traj_bayes_gmm_ml.py` | 877 | 627 | ok | Trajectory, Bayesian LGM/GMM and ML analysis |
| `python check_master.py` | 27 | 19 | ok | Auxiliary script retained for workflow provenance. |
| `rebuild_phq_and_rerun_baseline.py` | 173 | 121 | ok | Auxiliary script retained for workflow provenance. |
| `rerun_baseline_with_pos_split.py` | 186 | 130 | ok | Auxiliary script retained for workflow provenance. |
| `s4_sonar_runner.py` | 1097 | 826 | ok | S4-SONAR Runner (NO new data) |
| `shortform3_runner.py` | 849 | 636 | ok | Adaptive Minimal-Info Risk Inference Runner (PHQ/GAD/DASS) |
| `sonar_policy_runner.py` | 1412 | 1094 | ok | sonar_policy_runner.py |
| `sonar_policy_runner_fast.py` | 1015 | 819 | ok | sonar_policy_runner_fast.py |
| `sonar_runner.py` | 1155 | 881 | ok | 心理声呐（Psychological Sonar）——不加新数据的“最小信息 + 高敏漏报控制”筛查协议 |
| `sonar_runner2.py` | 711 | 530 | ok | SONAR Runner (no new data) - STRICT FN policy |
| `step0_mi_gen_and_run.py` | 569 | 421 | ok | Step0 量表可比性（纵向不变性 / 分段等值）——一键版 |
| `step0_mi_lavaan_runner.py` | 277 | 214 | ok | Step0 纵向/分段等值（测量不变性）——不使用 Mplus：用 Python 调用 R(lavaan/semTools) |
| `step1_build_derived_metrics.py` | 336 | 259 | ok | Step1 \| 三类派生指标（水平/变化/波动）v2 修复版 |
| `step2_H1A_parallel_POS_NEG.py` | 410 | 329 | ok | Step2-H1A 并联中介（POS 与 NEG 分开）Cross-lagged Parallel Mediation |
| `step2_H1B_delta_change_mediation.py` | 359 | 300 | ok | Step2-H1B 变化量中介：RES_T1 -> ΔCOP(T2-T1) -> ΔY(T3-T2) |
| `step2_crosslag_mediation_H1.py` | 489 | 368 | ok | Step2 早期中介：Cross-lagged Mediation（H1）- v2（修复ID合并 & 分结局dropna） |
| `step2_dedup_for_gray.py` | 199 | 155 | ok | Auxiliary script retained for workflow provenance. |
| `step3_ri_clpm_res_cop.py` | 501 | 391 | ok | Step3: RI-CLPM (within-person) 双向跨期：RES <-> COP |
| `step3_transform_check.py` | 93 | 75 | ok | Auxiliary script retained for workflow provenance. |
| `step4_H3_anxiety_takeover.py` | 583 | 460 | ok | Step4 H3: 焦虑接管（条件效应）——交互调节 / 暴露上行分层 / 高暴露状态切换 |
| `train_baseline_phq.py` | 99 | 66 | ok | Auxiliary script retained for workflow provenance. |
| `traj_bayes_lgm_gmm_all_scales.py` | 474 | 349 | ok | Bayes-LGM (PyMC) -> GMM (sklearn) 轨迹分型：对所有“≥3波次有数据”的量表总分列做轨迹分类 |
| `traj_bayes_lgm_gmm_all_scales_v2.py` | 574 | 418 | ok | 从建立宽表和长表开始：master(宽) -> long(长) -> Bayes-LGM -> GMM 轨迹分型（所有>=3波次量表） |
| `traj_curve_bayes_gmm_bic.py` | 544 | 401 | ok | traj_curve_bayes_gmm_bic.py |
| `trajectory_risk_pipeline.py` | 848 | 666 | ok | Trajectory -> GMM class -> Risk model (calibrated) -> DCA -> Deployment outputs |
| `数据准备.py` | 574 | 446 | ok | PHASE 0 — 数据准备（Data preparation）+ 构念字典（Construct Dictionary）+ 可检验性审计（Feasibility Audit） |
| `机制分析.py` | 720 | 524 | ok | Full Pipeline (Mechanism -> ML validation-ready) 保姆级一键脚本（已修复空数据报错） |
| `机制探索.py` | 461 | 358 | ok | Phase 1（机制探索）写死列名版：RES → COP → Y（按波次分别跑） |
| `灰区+发展路径主分析.py` | 1079 | 859 | ok | 灰区+发展路径主分析（可直接跑） |
| `计分审计.py` | 264 | 195 | ok | Phase 0 - Scoring Audit (计分审计) |
| `路径地图.py` | 1193 | 916 | ok | Direction A: Path Map (Trajectory Patterns + Transition Risk + Protection Factors) |
| `量表可比性是地基纵向不变性分段等值.py` | 579 | 419 | ok | Step0 量表纵向可比性（分段多时点等值）——Mplus 输入自动换行修复版 |

### `数据合并为长数据/`
| File | Lines | LOC | Syntax | Role / first-line docstring |
|---|---:|---:|---|---|
| `24年1季度.py` | 582 | 441 | ok | 问卷星（数字结果）多Excel合并 -> 标准化宽表： |
| `24年三季度.py` | 719 | 545 | ok | 问卷星多Excel合并（宽表）——写死 META/DEMO + 写死量表题目列 + 注意力题(1/2) + 维度/总分 + 识别CD-RISC 25 |
| `24年二季度.py` | 565 | 443 | ok | 问卷星Excel（新表头-写死版）： |
| `24年四季度.py` | 438 | 316 | ok | 问卷星（24年4季度：攀枝花/重庆/阿坝）——写死原文列名硬匹配 -> 合并宽表 + 量表计分 |
| `25年一季度.py` | 537 | 378 | ok | 问卷星Excel（多文件）→ 量表识别/重命名（量表_题号）→ 注意力题PASS → 量表维度/总分 → 合并宽表 |
| `25年三季度.py` | 355 | 287 | ok | 问卷星导出Excel：批量标准化列名 + 量表题目重命名(宽表) + 维度/总分 + 注意力题PASS标记 |
| `25年二季度.py` | 363 | 265 | ok | 问卷星多Excel（每个文件只读第1个sheet）→ 识别量表 → 统一命名(量表_题号) → 计算维度/总分 → 合并宽表输出 |
| `25年四季度.py` | 489 | 348 | ok | 问卷星导出Excel：自动识别量表区块 -> 统一命名为 量表_题号 -> 计算维度/总分 -> 合并宽表 |
| `打印列名.py` | 53 | 34 | ok | Print/export Excel column names |
| `探索性因素分析.py` | 479 | 364 | ok | EFA（探索性因素分析）批处理：按“每个时间点(=每个季度文件) × 每个量表”分别跑 |
| `数据体检.py` | 117 | 102 | ok | Data quality-control report |

### `分组用的/`
| File | Lines | LOC | Syntax | Role / first-line docstring |
|---|---:|---:|---|---|
| `AB_bridge_分组统计.py` | 249 | 184 | ok | 在你的文件夹内： |
| `参与概况表.py` | 73 | 55 | ok | Participation coverage summary |
| `构建全勤队列和AB分组.py` | 391 | 277 | ok | 生成两个Excel： |
| `缺失率筛查.py` | 193 | 140 | ok | 检查： |

### `pyhon文件/`
| File | Lines | LOC | Syntax | Role / first-line docstring |
|---|---:|---:|---|---|
| `312321.py` | 296 | 212 | ok | Legacy/non-manuscript visualization utility; not part of manuscript reproduction. |
| `RouteA v3.py` | 312 | 262 | ok | RouteA v3: backfill BIG/SUB unit by (WAVE + META_ID + META_SEQ) |
| `add_canon_fields_and_recheck.py` | 98 | 72 | ok | Auxiliary script retained for workflow provenance. |
| `analyze_personkey_mismatch_types.py` | 98 | 72 | ok | Participant-key mismatch diagnostic |
| `build_psych_master_db - 副本.py` | 700 | 520 | ok | build_psych_master_db.py |
| `build_psych_master_db.py` | 700 | 520 | ok | build_psych_master_db.py |
| `check_psych_master_db.py` | 474 | 344 | ok | check_psych_master_db.py |
| `list_all_columns_to_txt.py` | 236 | 191 | ok | Export column names for documentation |
| `make_bigunit_from_excelname_v3.py` | 454 | 351 | ok | Unit / big-unit harmonization and backfill |
| `make_bigunit_from_filename_view.py` | 324 | 242 | ok | Unit / big-unit harmonization and backfill |
| `make_bigunit_from_folders_v2.py` | 265 | 210 | ok | Unit / big-unit harmonization and backfill |
| `make_bigunit_manual_v4.py` | 382 | 320 | ok | Unit / big-unit harmonization and backfill |
| `make_bigunit_manual_v5.py` | 372 | 305 | ok | Unit / big-unit harmonization and backfill |
| `make_bigunit_manual_v5_1.py` | 391 | 325 | ok | Unit / big-unit harmonization and backfill |
| `make_bigunit_manual_v5_2.py` | 505 | 379 | ok | Unit / big-unit harmonization and backfill |
| `make_bigunit_manual_v5_3.py` | 283 | 213 | ok | Unit / big-unit harmonization and backfill |
| `make_bigunit_manual_v5_3_1.py` | 243 | 189 | ok | Unit / big-unit harmonization and backfill |
| `make_dictionary_view.py` | 24 | 18 | ok | Auxiliary script retained for workflow provenance. |
| `make_drop24_rekey_v6.py` | 165 | 137 | ok | Auxiliary script retained for workflow provenance. |
| `make_highrisk_labels_for_schemeC.py` | 231 | 177 | ok | 为 SchemeC split 输出的 CSV 增加“高风险标签”列（不修改 SQLite DB） |
| `make_schemeC_unit_time_holdout.py` | 234 | 177 | ok | make_schemeC_unit_time_holdout.py |
| `make_schemeC_unit_time_holdout_v2.py` | 277 | 216 | ok | make_schemeC_unit_time_holdout_v2.py |
| `make_unit_filled_view.py` | 116 | 104 | ok | Auxiliary script retained for workflow provenance. |
| `patch_24Q4_bigunit_by_context_v1.py` | 279 | 224 | ok | Unit / big-unit harmonization and backfill |
| `patch_24Q4_bigunit_by_signature_v2.py` | 372 | 288 | ok | patch_24Q4_bigunit_by_signature_v2.py |
| `patch_24Q4_units_v1.py` | 161 | 128 | ok | Patch 2024-Q4 unit labels |
| `patch_24Q4_units_v2_1_from_rawfiles.py` | 475 | 381 | ok | Patch 2024-Q4 unit labels |
| `patch_24Q4_units_v2_from_rawfiles.py` | 252 | 215 | ok | Patch 2024-Q4 unit labels |
| `psych_3d_bigunit_dash_v4.py` | 965 | 793 | ERR SyntaxError('unexpected character after line continuation character', ('<unknown>', 549, 48, ' ws = ",".join([f"\'{str(w).replace(\\"\'\\",\\"\'\'\\")}\'" for w in waves])\n', 549, 0)) | Unit / big-unit harmonization and backfill |
| `psych_3d_bigunit_dash_v4_1.py` | 922 | 774 | ok | Unit / big-unit harmonization and backfill |
| `psych_3d_viz_dash.py` | 587 | 488 | ok | psych_3d_viz_dash.py |
| `psych_3d_viz_dash_v2.py` | 481 | 377 | ok | psych_3d_viz_dash_v2.py |
| `qc_highrisk_labels.py` | 179 | 146 | ok | High-risk label construction or QC |
| `qc_schemeC_test_event_counts.py` | 68 | 52 | ok | Scheme-C unit-time holdout or label construction |
| `qc_v6_24_only.py` | 116 | 96 | ok | Auxiliary script retained for workflow provenance. |
| `qc_v6_drop24_rekey.py` | 108 | 84 | ok | Auxiliary script retained for workflow provenance. |
| `repair_and_diagnose_db_v2.py` | 626 | 483 | ok | repair_and_diagnose_db_v2.py |
| `repair_and_diagnose_db_v3.py` | 585 | 482 | ok | repair_and_diagnose_db_v3.py (v3.1 FIXED) |
| `routeA_srcfile_bigunit_v1.py` | 316 | 253 | ok | Unit / big-unit harmonization and backfill |
| `routeA_v2_backfill_by_metafile_seq.py` | 297 | 238 | ok | Route-A unit backfill workflow |
| `routeA_v3_backfill_by_metaid_seq.py` | 312 | 262 | ok | RouteA v3: backfill BIG/SUB unit by (WAVE + META_ID + META_SEQ) |
| `routeA_v4_personfill_keyword.py` | 183 | 140 | ok | RouteA v4: BIG_UNIT 回填增强（不依赖 META_FILE/META_ID） |
| `routeA_v4_personfill_keyword_fast.py` | 198 | 159 | ok | Route-A unit backfill workflow |
| `routeA_v5_backfill_bigunit_from_raw.py` | 381 | 296 | ok | Unit / big-unit harmonization and backfill |
| `step0_efa_then_mi_auto.py` | 678 | 532 | ok | Auxiliary script retained for workflow provenance. |
| `step0_pcq24_theory_mi_fix.py` | 465 | 360 | ok | Auxiliary script retained for workflow provenance. |
| `step3_submit_dt_ctsem_res_cop.py` | 534 | 398 | ok | Continuous-time resource–coping dynamic analysis |
| `step4_H3_takeover_strong_v2.py` | 1048 | 745 | ok | Anxiety-dominant switching boundary test |
| `数据库修补.py` | 169 | 135 | ok | patch_24Q4_units_v1.py |
| `检查并更正.py` | 668 | 523 | ok | Phase 0: Audit & Fix FullAttendance_Database.xlsx (wide sheet) |

## Contact
For controlled-access data requests or questions about the analysis workflow, contact the corresponding author listed in the manuscript.
