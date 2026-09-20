# Stage 8 - Comparative Evaluation of Fraud-Detection Methods

All numbers are measured against the PLANTED answer key. Perfect or near-perfect scores mean 'catches the specific planted pattern', not 'works on real welfare fraud'.

## Answer key size
| pattern | core_entities | involved_entities |
|---|---|---|
| orphan_disbursement | 25 | 25 |
| shared_account | 916 | 1194 |
| duplicate_identity | 40 | 80 |
| multivariate_anomaly | 25 | 25 |

## Coverage matrix - recall on each planted pattern, by method
| planted_pattern | METHOD rule_based (validated 2) | METHOD rule_based (all 6) | METHOD fuzzy linkage | METHOD isolation_forest (standard) | METHOD isolation_forest (layered) | METHOD ENSEMBLE (rules2 + fuzzy + IF) |
|---|---|---|---|---|---|---|
| orphan_disbursement | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| shared_account | 1.000 | 1.000 | 0.001 | 0.371 | 0.000 | 1.000 |
| duplicate_identity | 0.000 | 0.650 | 0.975 | 0.200 | 0.025 | 1.000 |
| multivariate_anomaly | 0.000 | 0.040 | 0.000 | 0.160 | 0.800 | 0.800 |

## Method scorecard
| detector | n_flagged | any_fraud_precision | overall_core_recall |
|---|---|---|---|
| METHOD rule_based (validated 2) | 1103 | 1.000 | 0.935 |
| METHOD rule_based (all 6) | 2047 | 0.556 | 0.962 |
| METHOD fuzzy linkage | 82 | 0.951 | 0.040 |
| METHOD isolation_forest (standard) | 469 | 0.753 | 0.350 |
| METHOD isolation_forest (layered) | 412 | 0.051 | 0.021 |
| METHOD ENSEMBLE (rules2 + fuzzy + IF) | 1595 | 0.752 | 0.995 |

## Signal-level detail (P/R/F1 against each detector's own target pattern)
| detector | has_ground_truth | target | n_flagged | tp_core | core_size | precision | recall | f1 | precision_strict | any_fraud_precision |
|---|---|---|---|---|---|---|---|---|---|---|
| rule:orphaned_disbursement | True | orphan_disbursement | 25 | 25 | 25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rule:shared_bank_account | True | shared_account | 1079 | 916 | 916 | 0.999 | 1.000 | 1.000 | 0.849 | 1.000 |
| rule:duplicate_cnic_multiple_identities | False | - | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| rule:cnic_mismatch | False | - | 986 | n/a | n/a | n/a | n/a | n/a | n/a | 0.079 |
| rule:duplicate_cycle_payment | False | - | 596 | n/a | n/a | n/a | n/a | n/a | n/a | 1.000 |
| rule:payment_to_inactive_beneficiary | False | - | 346 | n/a | n/a | n/a | n/a | n/a | n/a | 1.000 |
| fuzzy:duplicate_identity | True | duplicate_identity | 82 | 39 | 40 | 0.951 | 0.975 | 0.963 | 0.476 | 0.951 |
| iforest:standard | True | multivariate_anomaly | 469 | 4 | 25 | 0.009 | 0.160 | 0.016 | 0.009 | 0.753 |
| iforest:layered | True | multivariate_anomaly | 412 | 20 | 25 | 0.049 | 0.800 | 0.092 | 0.049 | 0.051 |
| baseline:payment_count_gt_6 | True | multivariate_anomaly | 595 | 25 | 25 | 0.042 | 1.000 | 0.081 | 0.042 | 1.000 |

## What each method catches that the others miss (core planted entities)
| method | core_caught | caught_by_this_method_only | core_total |
|---|---|---|---|
| METHOD rule_based (validated 2) | 941 | 940 | 1006 |
| METHOD fuzzy linkage | 40 | 39 | 1006 |
| METHOD isolation_forest (layered) | 21 | 21 | 1006 |

## Flagged-set overlap (Jaccard)
| jaccard | METHOD rule_based (validated 2) | METHOD rule_based (all 6) | METHOD fuzzy linkage | METHOD isolation_forest (standard) | METHOD isolation_forest (layered) | baseline:payment_count_gt_6 |
|---|---|---|---|---|---|---|
| METHOD rule_based (validated 2) | 1.000 | 0.539 | 0.002 | 0.277 | 0.000 | 0.505 |
| METHOD rule_based (all 6) | 0.539 | 1.000 | 0.015 | 0.162 | 0.011 | 0.276 |
| METHOD fuzzy linkage | 0.002 | 0.015 | 1.000 | 0.013 | 0.000 | 0.001 |
| METHOD isolation_forest (standard) | 0.277 | 0.162 | 0.013 | 1.000 | 0.159 | 0.454 |
| METHOD isolation_forest (layered) | 0.000 | 0.011 | 0.000 | 0.159 | 1.000 | 0.020 |
| baseline:payment_count_gt_6 | 0.505 | 0.276 | 0.001 | 0.454 | 0.020 | 1.000 |

## Tie-out with Stage 6 (pair-level)
| pairs_flagged | pairs_true | tp | precision | recall |
|---|---|---|---|---|
| 41 | 40 | 39 | 0.951 | 0.975 |

## How to read this honestly
- `precision` counts hub-account owners (shared_account) and original beneficiaries (duplicate_identity) as legitimately involved; `precision_strict` counts only the logged culprits and is the number comparable with the earlier per-stage evaluators.
- `any_fraud_precision` credits a flag if the entity is part of ANY planted fraud. A detector with low target precision but high any-fraud precision is finding real fraud of a different kind.
- The four heuristic rules have no planted ground truth: their counts are informative, not validated. `cnic_mismatch` in particular fires on generator noise (national ID CNIC typos), so a low any-fraud precision there is not evidence the rule is wrong.
- `baseline:payment_count_gt_6` is a sanity reference. If a one-line count threshold matches Isolation Forest on the planted anomalies, that reflects how the anomalies were planted (an extra out-of-window payment), not necessarily that Isolation Forest is weak in general.
- The layered Isolation Forest excludes beneficiaries already flagged by Stage 5/6 before fitting, so it is designed to be complementary, not a like-for-like rival.