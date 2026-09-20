# Stage 8 - Comparative Evaluation of Fraud-Detection Methods

All numbers are measured against the PLANTED answer key. Perfect or near-perfect scores mean 'catches the specific planted pattern', not 'works on real welfare fraud'.

## Answer key size
| pattern | core_entities | involved_entities |
|---|---|---|
| orphan_disbursement | 25 | 25 |
| shared_account | 300 | 388 |
| duplicate_identity | 40 | 80 |
| multivariate_anomaly | 25 | 25 |

## Coverage matrix - recall on each planted pattern, by method
| planted_pattern | METHOD rule_based (validated 2) | METHOD rule_based (all 6) | METHOD fuzzy linkage | METHOD isolation_forest (standard) | METHOD isolation_forest (layered) | METHOD ENSEMBLE (rules2 + fuzzy + IF) |
|---|---|---|---|---|---|---|
| orphan_disbursement | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 |
| shared_account | 1.000 | 1.000 | 0.007 | 0.370 | 0.000 | 1.000 |
| duplicate_identity | 0.000 | 0.650 | 1.000 | 0.275 | 0.000 | 1.000 |
| multivariate_anomaly | 0.000 | 0.080 | 0.000 | 0.120 | 0.840 | 0.840 |

## Method scorecard
| detector | n_flagged | any_fraud_precision | overall_core_recall |
|---|---|---|---|
| METHOD rule_based (validated 2) | 375 | 1.000 | 0.833 |
| METHOD rule_based (all 6) | 699 | 0.581 | 0.905 |
| METHOD fuzzy linkage | 80 | 1.000 | 0.108 |
| METHOD isolation_forest (standard) | 159 | 0.792 | 0.321 |
| METHOD isolation_forest (layered) | 139 | 0.151 | 0.054 |
| METHOD ENSEMBLE (rules2 + fuzzy + IF) | 592 | 0.801 | 0.990 |

## Signal-level detail (P/R/F1 against each detector's own target pattern)
| detector | has_ground_truth | target | n_flagged | tp_core | core_size | precision | recall | f1 | precision_strict | any_fraud_precision |
|---|---|---|---|---|---|---|---|---|---|---|
| rule:orphaned_disbursement | True | orphan_disbursement | 25 | 25 | 25 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| rule:shared_bank_account | True | shared_account | 350 | 300 | 300 | 1.000 | 1.000 | 1.000 | 0.857 | 1.000 |
| rule:duplicate_cnic_multiple_identities | False | - | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| rule:cnic_mismatch | False | - | 322 | n/a | n/a | n/a | n/a | n/a | n/a | 0.090 |
| rule:duplicate_cycle_payment | False | - | 218 | n/a | n/a | n/a | n/a | n/a | n/a | 1.000 |
| rule:payment_to_inactive_beneficiary | False | - | 108 | n/a | n/a | n/a | n/a | n/a | n/a | 1.000 |
| fuzzy:duplicate_identity | True | duplicate_identity | 80 | 40 | 40 | 1.000 | 1.000 | 1.000 | 0.500 | 1.000 |
| iforest:standard | True | multivariate_anomaly | 159 | 3 | 25 | 0.019 | 0.120 | 0.033 | 0.019 | 0.792 |
| iforest:layered | True | multivariate_anomaly | 139 | 21 | 25 | 0.151 | 0.840 | 0.256 | 0.151 | 0.151 |
| baseline:payment_count_gt_6 | True | multivariate_anomaly | 217 | 25 | 25 | 0.115 | 1.000 | 0.207 | 0.115 | 1.000 |

## What each method catches that the others miss (core planted entities)
| method | core_caught | caught_by_this_method_only | core_total |
|---|---|---|---|
| METHOD rule_based (validated 2) | 325 | 323 | 390 |
| METHOD fuzzy linkage | 42 | 40 | 390 |
| METHOD isolation_forest (layered) | 21 | 21 | 390 |

## Flagged-set overlap (Jaccard)
| jaccard | METHOD rule_based (validated 2) | METHOD rule_based (all 6) | METHOD fuzzy linkage | METHOD isolation_forest (standard) | METHOD isolation_forest (layered) | baseline:payment_count_gt_6 |
|---|---|---|---|---|---|---|
| METHOD rule_based (validated 2) | 1.000 | 0.536 | 0.004 | 0.265 | 0.000 | 0.480 |
| METHOD rule_based (all 6) | 0.536 | 1.000 | 0.040 | 0.158 | 0.015 | 0.269 |
| METHOD fuzzy linkage | 0.004 | 0.040 | 1.000 | 0.048 | 0.000 | 0.000 |
| METHOD isolation_forest (standard) | 0.265 | 0.158 | 0.048 | 1.000 | 0.137 | 0.430 |
| METHOD isolation_forest (layered) | 0.000 | 0.015 | 0.000 | 0.137 | 1.000 | 0.063 |
| baseline:payment_count_gt_6 | 0.480 | 0.269 | 0.000 | 0.430 | 0.063 | 1.000 |

## Tie-out with Stage 6 (pair-level)
| pairs_flagged | pairs_true | tp | precision | recall |
|---|---|---|---|---|
| 40 | 40 | 40 | 1.000 | 1.000 |

## How to read this honestly
- `precision` counts hub-account owners (shared_account) and original beneficiaries (duplicate_identity) as legitimately involved; `precision_strict` counts only the logged culprits and is the number comparable with the earlier per-stage evaluators.
- `any_fraud_precision` credits a flag if the entity is part of ANY planted fraud. A detector with low target precision but high any-fraud precision is finding real fraud of a different kind.
- The four heuristic rules have no planted ground truth: their counts are informative, not validated. `cnic_mismatch` in particular fires on generator noise (national ID CNIC typos), so a low any-fraud precision there is not evidence the rule is wrong.
- `baseline:payment_count_gt_6` is a sanity reference. If a one-line count threshold matches Isolation Forest on the planted anomalies, that reflects how the anomalies were planted (an extra out-of-window payment), not necessarily that Isolation Forest is weak in general.
- The layered Isolation Forest excludes beneficiaries already flagged by Stage 5/6 before fitting, so it is designed to be complementary, not a like-for-like rival.