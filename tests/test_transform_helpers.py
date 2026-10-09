"""Unit tests for the Stage 3 cleaning helpers in src/etl/transform_stage.py."""

import pandas as pd

from transform_stage import (
    STATUS_MAPPING_BENEFICIARY,
    build_flag_summary,
    clean_numeric_with_cap,
    parse_date_flexible,
    standardize_categorical,
)


def test_standardize_categorical_maps_variants_and_keeps_unknowns():
    s = pd.Series([" ACTIVE ", "actve", "Suspended", "weird"])
    assert standardize_categorical(s, STATUS_MAPPING_BENEFICIARY).tolist() == [
        "Active",
        "Active",
        "Suspended",
        "weird",  # unmapped values stay visible rather than being dropped
    ]


def test_parse_date_flexible_normalises_formats_and_flags_garbage():
    formatted, failed = parse_date_flexible(pd.Series(["2023-01-15", "15/01/2023", "garbage", None]))
    assert formatted.tolist()[:2] == ["2023-01-15", "2023-01-15"]
    assert formatted.iloc[2:].isna().all()
    # a missing date is not a parse failure, only unparseable text is
    assert failed.tolist() == [False, False, True, False]


def test_clean_numeric_with_cap_strips_currency_and_rejects_implausible_values():
    s = pd.Series(["Rs. 1,500", "-5", "200000", "abc"], name="amount_pkr")
    cleaned, invalid = clean_numeric_with_cap(s, cap=100000)
    assert cleaned.iloc[0] == 1500
    assert cleaned.iloc[1:].isna().all()
    assert invalid.tolist() == [False, True, True, True]


def test_clean_numeric_with_cap_allows_negatives_when_asked():
    cleaned, invalid = clean_numeric_with_cap(pd.Series(["-5"], name="x"), cap=100, allow_negative=True)
    assert cleaned.iloc[0] == -5
    assert not invalid.iloc[0]


def test_build_flag_summary_lists_true_flags():
    df = pd.DataFrame({"a": [True, False, True], "b": [True, False, False]})
    assert build_flag_summary(df, ["a", "b"]).tolist() == ["a, b", "", "a"]
