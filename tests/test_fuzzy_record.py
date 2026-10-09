"""Unit tests for the pure helpers in src/fraud_detection/fuzzy_record.py."""

from datetime import date
from pathlib import Path
from random import Random

import pandas as pd
import pytest

import src.fraud_detection.fuzzy_record as fuzzy_record
from src.fraud_detection.fuzzy_record import (
    BENEFICIARIES_CSV_PATH,
    GROUND_TRUTH_PATH,
    add_blocking_keys,
    dedupe_exact_rows,
    dob_typo,
    evaluate_threshold,
    filter_dob_typo_pairs,
    find_best_threshold,
    generate_blocking_passes,
    generate_candidate_pairs,
    inject_dob_typos,
    name_initials_key,
    normalize_dob,
    run,
    score_pairs,
    strip_common_tokens,
    summarize_extra_pairs,
    union_candidate_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_strip_common_tokens_removes_stopwords():
    assert strip_common_tokens("House No. 12, Street 4", {"house", "no", "street"}) == "12, 4"


def test_strip_common_tokens_falls_back_when_everything_is_stripped():
    assert strip_common_tokens("Street", {"street"}) == "Street"


def test_dedupe_exact_rows_keeps_one_row_per_beneficiary():
    df = pd.DataFrame(
        {
            "beneficiary_id": ["B1", "B1", "B2", "B3"],
            "full_name": ["Ali", "Ali", "Sara", "Copy"],
            "is_duplicate_row": [False, False, False, True],
        }
    )
    out = dedupe_exact_rows(df)
    assert sorted(out.index) == ["B1", "B2"]


def test_evaluate_threshold_precision_recall_f1():
    scored = pd.DataFrame(
        {
            "beneficiary_id_a": ["B1", "B3", "B5"],
            "beneficiary_id_b": ["B2", "B4", "B6"],
            "combined_score": [0.95, 0.90, 0.40],
        }
    )
    truth = {frozenset({"B2", "B1"}), frozenset({"B7", "B8"})}  # pair order must not matter
    m = evaluate_threshold(scored, truth, threshold=0.85)
    assert m["n_flagged"] == 2
    assert m["n_true_positives"] == 1
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_evaluate_threshold_with_nothing_flagged():
    scored = pd.DataFrame({"beneficiary_id_a": ["B1"], "beneficiary_id_b": ["B2"], "combined_score": [0.1]})
    m = evaluate_threshold(scored, {frozenset({"B1", "B2"})}, threshold=0.9)
    assert m["precision"] == m["recall"] == m["f1"] == 0.0


# ── multi-pass blocking ──────────────────────────────────────────────────


def _beneficiaries(rows):
    """Hand-built beneficiaries frame in the shape dedupe_exact_rows() returns."""
    df = pd.DataFrame(rows, columns=["beneficiary_id", "full_name", "address_line", "city", "date_of_birth_clean"])
    df["is_duplicate_row"] = False
    return dedupe_exact_rows(df)


def _pair_keys(pairs):
    return {frozenset(p) for p in pairs}


# B2 is B1 re-registered with a name typo and a one-digit DOB typo in the day;
# B4 is B3 with the DOB typo in the year. The rest are distractors.
TYPO_FRAME = [
    ("B1", "Kimberly Johnson", "552 Ryan Mount", "Lahore", "1962-03-15"),
    ("B2", "Kimberyl Johnson", "552 Rayn Mount", "lahore ", "1962-03-16"),
    ("B3", "Daniel Ramos", "21249 Reynolds Hollow Apt. 557", "Quetta", "1965-05-31"),
    ("B4", "DANIEL RAOMS", "21249 Reynolds Hollow Atp. 557", "Quetta", "1966-05-31"),
    ("B5", "Kevin Jones", "9 Elm Court", "Lahore", "1962-03-15"),  # same DOB as B1, different person
    ("B6", "Kimberly Johnson", "77 Oak Lane", "Karachi", "1962-03-16"),  # same name as B1, other city
    ("B7", "Karen Jackson", "1 Pine Road", "Lahore", "1962-07-19"),  # same initials/city/year, DOB 2+ typos off
]


def test_multi_pass_finds_dob_typo_duplicates_that_dob_only_misses():
    df = _beneficiaries(TYPO_FRAME)
    dob_only = _pair_keys(generate_candidate_pairs(df, blocking="dob"))
    multi = _pair_keys(generate_candidate_pairs(df, blocking="multi"))

    for true_pair in (frozenset({"B1", "B2"}), frozenset({"B3", "B4"})):
        assert true_pair not in dob_only
        assert true_pair in multi

    assert dob_only <= multi  # the DOB block is never lost
    # the extra pass stays narrow: another city, or a DOB more than one typo off, never pairs
    assert frozenset({"B1", "B6"}) not in multi
    assert frozenset({"B1", "B7"}) not in multi


def test_dob_typo_duplicates_are_flagged_end_to_end():
    df = _beneficiaries(TYPO_FRAME)
    truth = {frozenset({"B1", "B2"}), frozenset({"B3", "B4"})}
    scored = score_pairs(df, generate_candidate_pairs(df, blocking="multi"))
    best = find_best_threshold(scored, truth)
    assert best["recall"] == 1.0
    assert best["precision"] == 1.0


def test_union_has_no_duplicate_or_self_pairs():
    df = _beneficiaries(TYPO_FRAME)
    first = pd.MultiIndex.from_tuples([("B2", "B1"), ("B5", "B1")])
    second = pd.MultiIndex.from_tuples([("B1", "B2"), ("B2", "B1"), ("B3", "B3"), ("B4", "B3")])
    union = union_candidate_pairs(df, first, second)
    assert list(union) == [("B2", "B1"), ("B5", "B1"), ("B4", "B3")]  # first pass keeps its order at the front


def test_generated_candidate_pairs_are_unique_unordered_pairs():
    # B8 shares B1's exact DOB, initials and city, so both passes produce (B1, B8)
    df = _beneficiaries(TYPO_FRAME + [("B8", "Kim Jones", "552 Ryan Mount", "Lahore", "1962-03-15")])
    pairs = list(generate_candidate_pairs(df, blocking="multi"))
    keys = [frozenset(p) for p in pairs]
    assert len(keys) == len(set(keys))
    assert all(a != b for a, b in pairs)
    assert frozenset({"B1", "B8"}) in keys


def test_blocking_passes_return_the_dob_block_and_the_union():
    df = _beneficiaries(TYPO_FRAME)
    dob_pairs, pairs = generate_blocking_passes(df, blocking="multi")
    assert _pair_keys(dob_pairs) == _pair_keys(generate_candidate_pairs(df, blocking="dob"))
    assert _pair_keys(pairs) == _pair_keys(generate_candidate_pairs(df, blocking="multi"))
    dob_only, same = generate_blocking_passes(df, blocking="dob")
    assert same is dob_only


def test_generate_candidate_pairs_rejects_unknown_mode():
    with pytest.raises(ValueError):
        generate_candidate_pairs(_beneficiaries(TYPO_FRAME), blocking="soundex")


def test_name_initials_key():
    assert name_initials_key("Kimberly  JOHNSON") == "kj"
    assert name_initials_key("Kimberyl Johnson") == "kj"
    assert name_initials_key("Muhammad Ali Khan") == "a"  # stopwords stripped
    assert name_initials_key("William Jones") != name_initials_key("John Williams")
    assert name_initials_key("  ") is None
    assert name_initials_key(None) is None


def test_normalize_dob_handles_strings_dates_and_junk():
    assert normalize_dob("1962-03-15") == "1962-03-15"
    assert normalize_dob(date(1962, 3, 15)) == "1962-03-15"
    assert normalize_dob("2962-03-15") == "2962-03-15"  # typo'd year outside pandas' datetime range
    assert normalize_dob(None) is None
    assert normalize_dob(float("nan")) is None
    assert normalize_dob("15/03/1962") is None


def test_filter_dob_typo_pairs_keeps_one_edit_only():
    df = _beneficiaries(
        [
            ("A", "x", "x", "c", "1962-03-15"),
            ("B", "x", "x", "c", "1962-03-16"),  # one substitution
            ("C", "x", "x", "c", "1962-03-51"),  # adjacent swap of A's day digits: one edit
            ("D", "x", "x", "c", "1972-03-16"),  # two edits from A
            ("E", "x", "x", "c", None),
        ]
    )
    keyed = add_blocking_keys(df)
    pairs = pd.MultiIndex.from_tuples([("B", "A"), ("C", "A"), ("D", "A"), ("E", "A")])
    assert list(filter_dob_typo_pairs(keyed, pairs)) == [("B", "A"), ("C", "A")]


def test_inject_dob_typos_changes_exactly_one_digit_to_a_valid_date():
    df = _beneficiaries(TYPO_FRAME)
    out = inject_dob_typos(df, {"B1", "B3"}, seed=0)
    for bid in ("B1", "B3"):
        before, after = df.loc[bid, "date_of_birth_clean"], out.loc[bid, "date_of_birth_clean"]
        assert sum(a != b for a, b in zip(before, after)) == 1
        date.fromisoformat(after)
    untouched = ["B2", "B4", "B5", "B6", "B7"]
    assert (out.loc[untouched, "date_of_birth_clean"] == df.loc[untouched, "date_of_birth_clean"]).all()
    assert out.equals(inject_dob_typos(df, {"B1", "B3"}, seed=0))  # deterministic per seed


def test_inject_dob_typos_skips_missing_and_non_iso_dobs():
    df = _beneficiaries(
        [
            ("A", "x", "x", "c", "15/03/1962"),
            ("B", "x", "x", "c", None),
            ("C", "x", "x", "c", "1962-02-30"),  # ISO-shaped but not a real date
            ("D", "x", "x", "c", "1962-03-15"),
        ]
    )
    out = inject_dob_typos(df, {"A", "B", "C", "D"}, seed=1)  # must return, not loop
    assert list(out.loc[["A", "C"], "date_of_birth_clean"]) == ["15/03/1962", "1962-02-30"]
    assert pd.isna(out.loc["B", "date_of_birth_clean"])
    assert out.loc["D", "date_of_birth_clean"] != "1962-03-15"


@pytest.mark.parametrize("bad", ["15/03/1962", "1962-02-30", "", "1962-3-15"])
def test_dob_typo_rejects_non_iso_dates(bad):
    with pytest.raises(ValueError):
        dob_typo(bad, Random(0))


def test_dob_typo_handles_leap_day():
    for seed in range(50):
        date.fromisoformat(dob_typo("1964-02-29", Random(seed)))


def test_union_rejects_labels_not_in_index():
    df = _beneficiaries(TYPO_FRAME)
    with pytest.raises(ValueError, match="not in df.index"):
        union_candidate_pairs(df, pd.MultiIndex.from_tuples([("B2", "B1"), ("B9", "B1")]))


def test_summarize_extra_pairs():
    scored = pd.DataFrame(
        {
            "beneficiary_id_a": ["B1", "B3", "B5", "B7"],
            "beneficiary_id_b": ["B2", "B4", "B6", "B8"],
            "combined_score": [0.95, 0.90, 0.61, 0.30],
        }
    )
    base = [("B2", "B1")]  # orientation must not matter
    truth = {frozenset({"B3", "B4"})}
    s = summarize_extra_pairs(scored, base, truth, threshold=0.62)
    assert s["n_extra"] == 3
    assert s["n_extra_true"] == 1
    assert s["top_stranger_score"] == pytest.approx(0.61)
    assert s["top_stranger_ids"] == ("B5", "B6")
    assert s["n_extra_stored"] == 2
    assert s["n_extra_strangers_flagged"] == 0


# ── offline path ─────────────────────────────────────────────────────────


@pytest.fixture
def no_db(monkeypatch):
    """Makes any DB access in fuzzy_record fail the test loudly."""

    def forbidden(*args, **kwargs):
        raise AssertionError("offline path touched the database")

    monkeypatch.setattr(fuzzy_record, "get_connection", forbidden)
    monkeypatch.setattr(fuzzy_record, "write_matches", forbidden)


@pytest.mark.parametrize("blocking", ["multi", "dob"])
def test_run_from_csv_never_touches_db_and_returns_best_threshold(monkeypatch, no_db, blocking):
    frame = pd.DataFrame(
        TYPO_FRAME, columns=["beneficiary_id", "full_name", "address_line", "city", "date_of_birth_clean"]
    )
    frame["is_duplicate_row"] = False
    truth = {frozenset({"B1", "B2"}), frozenset({"B3", "B4"})}
    monkeypatch.setattr(fuzzy_record, "load_beneficiaries_csv", lambda path: frame.copy())
    monkeypatch.setattr(fuzzy_record, "load_ground_truth_pairs", lambda path: truth)

    # dry_run=False on purpose: from_csv alone must keep the DB out of it
    best = run(blocking, from_csv="hand-built.csv", dry_run=False, ground_truth="unused.csv", simulate_dob_typos=None)

    deduped = dedupe_exact_rows(frame)
    expected = find_best_threshold(score_pairs(deduped, generate_candidate_pairs(deduped, blocking)), truth)
    assert best == expected
    assert set(best) == {"threshold", "precision", "recall", "f1", "n_flagged", "n_true_positives"}
    assert best["f1"] == (1.0 if blocking == "multi" else 0.0)


def test_run_on_tracked_15k_csv_multi_pass_does_not_lose_f1(monkeypatch, no_db):
    monkeypatch.chdir(REPO_ROOT)
    results = {
        blocking: run(blocking, str(BENEFICIARIES_CSV_PATH), False, str(GROUND_TRUTH_PATH), None)
        for blocking in ("dob", "multi")
    }
    assert results["multi"]["f1"] >= results["dob"]["f1"] - 0.01
