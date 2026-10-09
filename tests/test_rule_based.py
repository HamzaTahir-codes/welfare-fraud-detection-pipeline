"""Unit tests for the Stage 5 rules in src/fraud_detection/rule_based.py, on tiny hand-built frames."""

import pandas as pd

from src.fraud_detection.rule_based import (
    rule_cnic_mismatch,
    rule_duplicate_cnic,
    rule_duplicate_cycle_payment,
    rule_inactive_status_payment,
    rule_orphaned_disbursement,
    rule_shared_bank_account,
)


def flagged(result: pd.DataFrame) -> set:
    return set(result["entity_id"])


def test_duplicate_cnic_flags_cnic_shared_by_distinct_beneficiaries():
    ben = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "beneficiary_id": ["B1", "B2", "B3", "B3"],
            "cnic": ["111", "111", "222", "222"],
            "is_duplicate_row": [False, False, False, False],
        }
    )
    # B3 appearing twice with its own CNIC is not two identities
    assert flagged(rule_duplicate_cnic(ben)) == {"1", "2"}


def test_duplicate_cnic_ignores_exact_duplicate_rows():
    ben = pd.DataFrame(
        {
            "row_id": [1, 2],
            "beneficiary_id": ["B1", "B2"],
            "cnic": ["111", "111"],
            "is_duplicate_row": [False, True],
        }
    )
    assert rule_duplicate_cnic(ben).empty


def test_shared_bank_account_needs_distinct_beneficiaries():
    disb = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4, 5],
            "beneficiary_id": ["B1", "B1", "B1", "B2", "B3"],
            "bank_account_number": ["A", "A", "A", "X", "X"],
        }
    )
    # one beneficiary paid many times into their own account is legitimate
    assert flagged(rule_shared_bank_account(disb)) == {"4", "5"}


def test_orphaned_disbursement_passes_through_stage3_flag():
    disb = pd.DataFrame({"row_id": [1, 2], "orphaned_beneficiary_flag": [True, False]})
    result = rule_orphaned_disbursement(disb)
    assert flagged(result) == {"1"}
    assert (result["entity_type"] == "disbursement").all()


def test_cnic_mismatch_compares_against_linked_national_id():
    ben = pd.DataFrame({"row_id": [1, 2], "beneficiary_id": ["B1", "B2"], "cnic": ["111", "222"]})
    nid = pd.DataFrame({"linked_beneficiary_id": ["B1", "B2"], "national_id_no": ["111", "999"]})
    assert flagged(rule_cnic_mismatch(ben, nid)) == {"2"}


def test_duplicate_cycle_payment_ignores_full_row_copies():
    disb = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "beneficiary_id": ["B1", "B1", "B2", "B2"],
            "disbursement_cycle": ["C1", "C1", "C1", "C1"],
            # B1: two distinct payments in C1. B2: one payment plus an injected exact copy.
            "disbursement_id": ["D1", "D2", "D3", "D3"],
        }
    )
    assert flagged(rule_duplicate_cycle_payment(disb)) == {"1", "2"}


def test_inactive_status_payment_flags_inactive_and_suspended():
    ben = pd.DataFrame({"beneficiary_id": ["B1", "B2", "B3"], "status_clean": ["Active", "Inactive", "Suspended"]})
    disb = pd.DataFrame({"row_id": [1, 2, 3], "beneficiary_id": ["B1", "B2", "B3"]})
    assert flagged(rule_inactive_status_payment(ben, disb)) == {"2", "3"}


def test_rule_output_schema():
    disb = pd.DataFrame({"row_id": [7], "orphaned_beneficiary_flag": [True]})
    result = rule_orphaned_disbursement(disb)
    assert list(result.columns) == ["entity_id", "entity_type", "signal_type", "score"]
    assert result["entity_id"].map(type).eq(str).all()
