"""
fuzzy_linkage.py — Stage 6: Fuzzy record linkage for duplicate-identity
ghost beneficiaries.

Approach:
    1. Pull `beneficiaries` from Postgres.
    2. Drop exact full-row duplicates (is_duplicate_row) before comparing —
       those are Stage 3/5's data-quality artifact, not the fraud pattern
       this stage targets, and would contaminate results with trivial
       100%-similarity "matches" that are the same physical identity twice.
    3. Block candidate pairs on exact date_of_birth_clean — every planted
       duplicate_identity pair shares DOB by design, so this guarantees no
       true pair is blocked out, while still cutting comparisons down from
       O(n^2) to same-birthday groups only.
    4. Score each candidate pair with RapidFuzz token_sort_ratio on
       full_name and address_line, combined as a weighted average.
    5. Tune the flagging threshold against the planted ground truth
       (duplicate_identity rows in ground_truth_fraud_ids.csv) by sweeping
       thresholds and picking the one that maximizes F1 — mirrors the
       validation rigor already applied to Stage 5's rule-based signals.
    6. Write candidate pairs scoring above MIN_SCORE_TO_STORE to
       record_linkage_matches, with is_flagged set by the tuned threshold.

>>> CHECK BEFORE RUNNING <<<
- Import path for get_connection() is a guess based on the stated
  connection.py/config.py pattern — adjust to match your actual module.
- GROUND_TRUTH_PATH assumes project-root location, per your Stage 1 setup.
"""

import uuid
from pathlib import Path

import pandas as pd
import recordlinkage
from psycopg2.extras import execute_values
from rapidfuzz import fuzz

from src.database.connection import get_connection  # ADJUST if your actual path differs

# ── CONFIG ────────────────────────────────────────────────────────────────
GROUND_TRUTH_PATH = Path("data") / "ground_truth_fraud_ids.csv"

NAME_WEIGHT = 0.6
ADDRESS_WEIGHT = 0.4

MIN_SCORE_TO_STORE = 0.50          # don't clutter the table with near-zero scores
THRESHOLD_SEARCH_RANGE = [round(x * 0.01, 2) for x in range(50, 100)]  # 0.50 .. 0.99

MATCH_METHOD = "fuzzy_duplicate_identity"

# Tokens that appear across many unrelated records and inflate token_sort_ratio
# without carrying identity signal. Strip these before scoring so similarity
# reflects the distinctive part of the name/address, not shared boilerplate.
# ADJUST this list based on what you actually see in your data (see the
# diagnostic step in main() before trusting these defaults).
NAME_STOPWORDS = {
    "muhammad", "mohammad", "mohammed", "syed", "sheikh", "hafiz",
    "bibi", "begum", "khan", "bin", "binte", "s/o", "d/o", "w/o",
}
ADDRESS_STOPWORDS = {
    "house", "street", "st", "road", "rd", "block", "colony",
    "sector", "phase", "town", "society", "plot", "no", "number",
}


def strip_common_tokens(text: str, stopwords: set) -> str:
    tokens = [t for t in str(text).lower().split() if t.strip(",.") not in stopwords]
    return " ".join(tokens) if tokens else str(text)  # fall back to original if everything was stripped


# ── DATA LOADING ─────────────────────────────────────────────────────────

def load_beneficiaries(conn) -> pd.DataFrame:
    query = """
        SELECT row_id, beneficiary_id, full_name, address_line,
               date_of_birth_clean, is_duplicate_row
        FROM beneficiaries
        WHERE beneficiary_id IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def load_ground_truth_pairs(path: Path) -> set:
    """Returns a set of frozenset({beneficiary_id_a, beneficiary_id_b})
    for every planted duplicate_identity pair — used only for threshold
    tuning/evaluation, never fed into the matching algorithm itself.
    """
    gt = pd.read_csv(path)
    dup = gt[gt["fraud_type"] == "duplicate_identity"]
    return {
        frozenset({row["original_beneficiary_id"], row["duplicate_beneficiary_id"]})
        for _, row in dup.iterrows()
    }


# ── CANDIDATE GENERATION & SCORING ──────────────────────────────────────

def dedupe_exact_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Stage 3/5's exact full-row duplicate artifacts before fuzzy
    comparison — keeps one physical row per beneficiary_id so beneficiary_id
    is a reliable unique key for this stage's candidate generation and so
    exact duplicates don't show up as trivial 100%-similarity false matches.
    """
    clean = df[df["is_duplicate_row"] != True].copy()  # noqa: E712
    clean = clean.drop_duplicates(subset="beneficiary_id", keep="first")
    return clean.set_index("beneficiary_id", drop=False)


def generate_candidate_pairs(df: pd.DataFrame) -> pd.MultiIndex:
    """Blocks on exact date_of_birth_clean. Uses the SINGLE-argument
    recordlinkage.Index().index(df) deduplication mode — this is a
    self-linkage task (one dataset compared to itself), not a link between
    two different datasets. The single-arg form is what actually excludes
    self-pairs (a row matched to itself) and duplicate orderings ((A,B) and
    (B,A) both appearing) — passing df twice as index(df, df) does neither,
    since that tells recordlinkage to treat them as two separate datasets.
    """
    indexer = recordlinkage.Index()
    indexer.block(left_on="date_of_birth_clean")
    return indexer.index(df)


def score_pairs(df: pd.DataFrame, pairs: pd.MultiIndex) -> pd.DataFrame:
    records = []
    for idx_a, idx_b in pairs:
        row_a, row_b = df.loc[idx_a], df.loc[idx_b]

        name_a = strip_common_tokens(row_a["full_name"], NAME_STOPWORDS)
        name_b = strip_common_tokens(row_b["full_name"], NAME_STOPWORDS)
        addr_a = strip_common_tokens(row_a["address_line"], ADDRESS_STOPWORDS)
        addr_b = strip_common_tokens(row_b["address_line"], ADDRESS_STOPWORDS)

        name_score = fuzz.token_sort_ratio(name_a, name_b) / 100.0
        address_score = fuzz.token_sort_ratio(addr_a, addr_b) / 100.0
        combined = NAME_WEIGHT * name_score + ADDRESS_WEIGHT * address_score

        records.append({
            "beneficiary_id_a": row_a["beneficiary_id"],
            "beneficiary_id_b": row_b["beneficiary_id"],
            "name_score": name_score,
            "address_score": address_score,
            "combined_score": combined,
        })
    return pd.DataFrame(records)


# ── THRESHOLD TUNING AGAINST GROUND TRUTH ───────────────────────────────

def evaluate_threshold(scored: pd.DataFrame, ground_truth_pairs: set, threshold: float) -> dict:
    flagged = scored[scored["combined_score"] >= threshold]
    flagged_pairs = {
        frozenset({row["beneficiary_id_a"], row["beneficiary_id_b"]})
        for _, row in flagged.iterrows()
    }

    true_positives = flagged_pairs & ground_truth_pairs
    precision = len(true_positives) / len(flagged_pairs) if flagged_pairs else 0.0
    recall = len(true_positives) / len(ground_truth_pairs) if ground_truth_pairs else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_flagged": len(flagged_pairs),
        "n_true_positives": len(true_positives),
    }


def find_best_threshold(scored: pd.DataFrame, ground_truth_pairs: set) -> dict:
    results = [evaluate_threshold(scored, ground_truth_pairs, t) for t in THRESHOLD_SEARCH_RANGE]
    return max(results, key=lambda r: r["f1"])


def print_score_separation_report(scored: pd.DataFrame, ground_truth_pairs: set):
    """Prints score distribution for known true-duplicate pairs vs all other
    candidate pairs, so you can SEE separation before trusting any threshold
    — if these ranges still overlap heavily, the scoring approach needs more
    work, not just a different cutoff.
    """
    pair_key = scored.apply(lambda r: frozenset({r["beneficiary_id_a"], r["beneficiary_id_b"]}), axis=1)
    is_true_pair = pair_key.isin(ground_truth_pairs)

    true_scores = scored.loc[is_true_pair, "combined_score"]
    false_scores = scored.loc[~is_true_pair, "combined_score"]

    print("\n--- Score separation report ---")
    print(f"True duplicate pairs found in candidates: {len(true_scores)}/{len(ground_truth_pairs)}")
    print(f"True pair scores -> min={true_scores.min():.3f} median={true_scores.median():.3f} max={true_scores.max():.3f}")
    print(f"Non-match scores -> min={false_scores.min():.3f} median={false_scores.median():.3f} "
          f"p90={false_scores.quantile(0.9):.3f} max={false_scores.max():.3f}")
    print("--- end report ---\n")


# ── WRITE RESULTS ────────────────────────────────────────────────────────

def write_matches(conn, scored: pd.DataFrame, threshold: float):
    to_store = scored[scored["combined_score"] >= MIN_SCORE_TO_STORE].copy()
    to_store["is_flagged"] = to_store["combined_score"] >= threshold

    rows = [
        (
            r["beneficiary_id_a"],
            r["beneficiary_id_b"],
            MATCH_METHOD,
            round(float(r["combined_score"]), 4),
            bool(r["is_flagged"]),
        )
        for _, r in to_store.iterrows()
    ]

    insert_query = """
        INSERT INTO record_linkage_matches
            (beneficiary_id, matched_record_id, match_method, confidence_score, is_flagged)
        VALUES %s
    """
    with conn.cursor() as cur:
        execute_values(cur, insert_query, rows)
    conn.commit()
    return len(rows)


# ── MAIN ─────────────────────────────────────────────────────────────────

def main():
    conn = get_connection()
    try:
        beneficiaries = load_beneficiaries(conn)
        deduped = dedupe_exact_rows(beneficiaries)

        pairs = generate_candidate_pairs(deduped)
        print(f"Candidate pairs after DOB blocking: {len(pairs)}")

        scored = score_pairs(deduped, pairs)

        ground_truth_pairs = load_ground_truth_pairs(GROUND_TRUTH_PATH)
        print_score_separation_report(scored, ground_truth_pairs)

        best = find_best_threshold(scored, ground_truth_pairs)
        print(
            f"Best threshold: {best['threshold']} "
            f"(precision={best['precision']:.3f}, recall={best['recall']:.3f}, f1={best['f1']:.3f}, "
            f"flagged={best['n_flagged']}, true_positives={best['n_true_positives']}/{len(ground_truth_pairs)})"
        )

        n_written = write_matches(conn, scored, threshold=best["threshold"])
        print(f"Wrote {n_written} candidate pairs to record_linkage_matches "
              f"(scored >= {MIN_SCORE_TO_STORE}, flagged at >= {best['threshold']}).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()