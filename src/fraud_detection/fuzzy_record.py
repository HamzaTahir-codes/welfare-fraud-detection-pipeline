"""
fuzzy_linkage.py — Stage 6: Fuzzy record linkage for duplicate-identity
ghost beneficiaries.

Approach:
    1. Pull `beneficiaries` from Postgres (or, offline, from the cleaned
       CSV with --from-csv; nothing is written to the DB in that mode).
    2. Drop exact full-row duplicates (is_duplicate_row) before comparing —
       those are Stage 3/5's data-quality artifact, not the fraud pattern
       this stage targets, and would contaminate results with trivial
       100%-similarity "matches" that are the same physical identity twice.
    3. Generate candidate pairs with two blocking passes and union them
       (deduplicated):
         a. exact date_of_birth_clean — every planted duplicate_identity
            pair shares DOB by design, so this alone catches all of them on
            generated data while cutting comparisons from O(n^2) to
            same-birthday groups.
         b. DOB-typo pass: block on (name initials, city, birth year) and
            on (name initials, city, birth month-day), then keep only pairs
            whose DOBs are at most one typo apart (one digit substituted or
            two adjacent digits swapped). A duplicate registration whose DOB
            was mistyped can never meet its original in pass (a); a one-typo
            error sits inside either the year or the month-day, so the other
            component still collides here. Name initials survive the usual
            spelling typo (it rarely hits a token's first letter, and the
            generator never does) and a re-registration keeps its city.
       The scoring in step 4 is deliberately unchanged, so pass (b) has to be
       narrow: every extra stranger it lets in competes with the true pairs
       that score low because one side's address is missing (~0.59-0.64).
       This narrow design was chosen on the same 15k evaluation data it is
       reported on, so its "no F1 loss" there is a thin result, not a robust
       one: in multi mode run() prints the pass-(b)-only pairs, the best
       stranger among them (it sits just under the tuned threshold) and how
       many of them reach MIN_SCORE_TO_STORE and would be stored unflagged.
       Known gaps: a DOB more than one typo off, or a DOB typo combined with
       a typo in a token's first letter, a swapped first/last name, or a
       different city on the re-registration.
    4. Score each candidate pair with RapidFuzz token_sort_ratio on
       full_name and address_line, combined as a weighted average.
    5. Tune the flagging threshold against the planted ground truth
       (duplicate_identity rows in ground_truth_fraud_ids.csv) by sweeping
       thresholds and picking the one that maximizes F1 — mirrors the
       validation rigor already applied to Stage 5's rule-based signals.
    6. Write candidate pairs scoring above MIN_SCORE_TO_STORE to
       record_linkage_matches, with is_flagged set by the tuned threshold.

Usage:
    python -m src.fraud_detection.fuzzy_record                  # DB mode (pipeline)
    python -m src.fraud_detection.fuzzy_record --dry-run        # DB read, no write
    python -m src.fraud_detection.fuzzy_record --from-csv data/clean/beneficiaries_cleaned.csv
        # fully offline: reads the cleaned CSV, never connects to or writes the DB
    ... --blocking dob                  # DOB-only blocking, for comparison
    ... --simulate-dob-typos 42         # offline evaluation only: put a one-digit
                                        # DOB typo on every planted duplicate first

>>> CHECK BEFORE RUNNING <<<
- Import path for get_connection() is a guess based on the stated
  connection.py/config.py pattern — adjust to match your actual module.
- GROUND_TRUTH_PATH assumes project-root location, per your Stage 1 setup.
"""

import argparse
import random
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import recordlinkage
from psycopg2.extras import execute_values
from rapidfuzz import fuzz
from rapidfuzz.distance import OSA

from src.database.connection import get_connection  # ADJUST if your actual path differs

# ── CONFIG ────────────────────────────────────────────────────────────────
GROUND_TRUTH_PATH = Path("data") / "ground_truth_fraud_ids.csv"
BENEFICIARIES_CSV_PATH = Path("data") / "clean" / "beneficiaries_cleaned.csv"

BLOCKING_MODES = ("multi", "dob")  # multi = DOB block + DOB-typo pass (default); dob = DOB block only
MAX_DOB_TYPOS = 1  # pass (b): max optimal-string-alignment edits between the two DOBs

NAME_WEIGHT = 0.6
ADDRESS_WEIGHT = 0.4

MIN_SCORE_TO_STORE = 0.50  # don't clutter the table with near-zero scores
THRESHOLD_SEARCH_RANGE = [round(x * 0.01, 2) for x in range(50, 100)]  # 0.50 .. 0.99

MATCH_METHOD = "fuzzy_duplicate_identity"

# Tokens that appear across many unrelated records and inflate token_sort_ratio
# without carrying identity signal. Strip these before scoring so similarity
# reflects the distinctive part of the name/address, not shared boilerplate.
# ADJUST this list based on what you actually see in your data (see the
# diagnostic step in main() before trusting these defaults).
NAME_STOPWORDS = {
    "muhammad",
    "mohammad",
    "mohammed",
    "syed",
    "sheikh",
    "hafiz",
    "bibi",
    "begum",
    "khan",
    "bin",
    "binte",
    "s/o",
    "d/o",
    "w/o",
}
ADDRESS_STOPWORDS = {
    "house",
    "street",
    "st",
    "road",
    "rd",
    "block",
    "colony",
    "sector",
    "phase",
    "town",
    "society",
    "plot",
    "no",
    "number",
}


def strip_common_tokens(text: str, stopwords: set) -> str:
    tokens = [t for t in str(text).lower().split() if t.strip(",.") not in stopwords]
    return " ".join(tokens) if tokens else str(text)  # fall back to original if everything was stripped


# ── DATA LOADING ─────────────────────────────────────────────────────────


def load_beneficiaries(conn) -> pd.DataFrame:
    query = """
        SELECT row_id, beneficiary_id, full_name, address_line, city,
               date_of_birth_clean, is_duplicate_row
        FROM beneficiaries
        WHERE beneficiary_id IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(query)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=cols)


def load_beneficiaries_csv(path: Path) -> pd.DataFrame:
    """Offline counterpart of load_beneficiaries(): same columns (minus the
    DB-only row_id) read from the cleaned CSV that loader_stage would load.
    DOB stays a string so the DOB block matches exactly what the CSV holds.
    """
    cols = ["beneficiary_id", "full_name", "address_line", "city", "date_of_birth_clean", "is_duplicate_row"]
    df = pd.read_csv(path, usecols=cols, dtype={"date_of_birth_clean": str})
    return df[df["beneficiary_id"].notna()].reset_index(drop=True)


def load_ground_truth_pairs(path: Path) -> set:
    """Returns a set of frozenset({beneficiary_id_a, beneficiary_id_b})
    for every planted duplicate_identity pair — used only for threshold
    tuning/evaluation, never fed into the matching algorithm itself.
    """
    gt = pd.read_csv(path)
    dup = gt[gt["fraud_type"] == "duplicate_identity"]
    return {frozenset({row["original_beneficiary_id"], row["duplicate_beneficiary_id"]}) for _, row in dup.iterrows()}


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


def name_initials_key(name) -> str | None:
    """First letters of the name tokens in order, stopwords stripped — e.g.
    "Kimberly Johnson" and "Kimberyl  JOHNSON" both give "kj". Typos that keep
    the first letter of each token (the common case, and the only kind the
    generator plants) keep the key. None when nothing usable is left, so empty
    names never form one giant block.
    """
    if pd.isna(name):
        return None
    letters = []
    for token in str(name).lower().split():
        if token.strip(",.") in NAME_STOPWORDS:
            continue
        token = re.sub(r"[^a-z]", "", token)
        if token:
            letters.append(token[0])
    return "".join(letters) or None


def normalize_dob(value) -> str | None:
    """YYYY-MM-DD text for a DOB held as a string (CSV) or a date
    (Postgres); None when missing or not in that shape. Kept as text rather
    than parsed so a typo'd year outside pandas' datetime range (e.g. 2962)
    still blocks and compares.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = str(value).strip()[:10]
    return text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else None


def add_blocking_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the pass-(b) blocking columns: name initials, normalized city,
    normalized DOB, birth year, birth month-day. Missing values stay null and
    are never blocked on.
    """
    out = df.copy()
    dob = out["date_of_birth_clean"].map(normalize_dob)
    city = out["city"].astype("string").str.strip().str.lower()
    out["block_name_key"] = out["full_name"].map(name_initials_key)
    out["block_city"] = city.mask(city == "")
    out["block_dob"] = dob
    out["block_birth_year"] = dob.str[:4]
    out["block_birth_month_day"] = dob.str[5:]
    return out


def filter_dob_typo_pairs(keyed: pd.DataFrame, pairs: pd.MultiIndex, max_typos: int = MAX_DOB_TYPOS) -> pd.MultiIndex:
    """Keeps only pairs whose normalized DOBs differ by at most max_typos
    optimal-string-alignment edits (substitution or adjacent transposition) —
    i.e. one could be a keying error of the other. Pairs with a missing DOB
    on either side are dropped.
    """
    dob = keyed["block_dob"]
    dob_a = dob.reindex(pairs.get_level_values(0)).to_numpy()
    dob_b = dob.reindex(pairs.get_level_values(1)).to_numpy()
    keep = [isinstance(a, str) and isinstance(b, str) and OSA.distance(a, b) <= max_typos for a, b in zip(dob_a, dob_b)]
    return pairs[np.array(keep, dtype=bool)]


def union_candidate_pairs(df: pd.DataFrame, *pair_sets: pd.MultiIndex) -> pd.MultiIndex:
    """Unions several candidate-pair sets over the same frame with every
    unordered pair kept exactly once. Each pair is re-oriented to
    recordlinkage's dedup convention (later row first, earlier row second) so
    (A,B) and (B,A) from different passes collapse, self-pairs are dropped,
    and the first pass's pairs keep their original order at the front.
    Requires a unique index (dedupe_exact_rows guarantees it); raises
    ValueError if the index is not unique or a pair names a label not in it.
    """
    if not df.index.is_unique:
        raise ValueError("union_candidate_pairs needs a unique index (run dedupe_exact_rows first)")
    position = pd.Series(range(len(df)), index=df.index)
    frames = []
    for pairs in pair_sets:
        labels_a, labels_b = pairs.get_level_values(0), pairs.get_level_values(1)
        unknown = labels_a[~labels_a.isin(df.index)].union(labels_b[~labels_b.isin(df.index)])
        if len(unknown):
            raise ValueError(f"candidate pairs reference labels not in df.index: {list(unknown[:5])}")
        pos_a = position.reindex(labels_a).to_numpy()
        pos_b = position.reindex(labels_b).to_numpy()
        frames.append(pd.DataFrame({"later": np.maximum(pos_a, pos_b), "earlier": np.minimum(pos_a, pos_b)}))
    if not frames:
        return pd.MultiIndex.from_arrays([[], []])
    merged = pd.concat(frames, ignore_index=True).drop_duplicates()
    merged = merged[merged["later"] != merged["earlier"]]
    return pd.MultiIndex.from_arrays([df.index[merged["later"]], df.index[merged["earlier"]]])


def generate_candidate_pairs(df: pd.DataFrame, blocking: str = "multi") -> pd.MultiIndex:
    """Returns the candidate pairs for the given blocking mode (see
    generate_blocking_passes)."""
    return generate_blocking_passes(df, blocking)[1]


def generate_blocking_passes(df: pd.DataFrame, blocking: str = "multi") -> tuple[pd.MultiIndex, pd.MultiIndex]:
    """Blocks on exact date_of_birth_clean and, unless blocking="dob", also
    on name initials + city + birth year and name initials + city + birth
    month-day restricted to DOBs at most one typo apart (see module
    docstring, step 3). Returns (dob_pairs, candidate_pairs): the exact-DOB
    block on its own, so callers can tell what the extra pass added without
    blocking twice, and the deduplicated union (the same object as dob_pairs
    when blocking="dob").

    Every pass uses the SINGLE-argument recordlinkage.Index().index(df)
    deduplication mode — this is a self-linkage task (one dataset compared to
    itself), not a link between two different datasets. The single-arg form
    is what actually excludes self-pairs (a row matched to itself) and
    duplicate orderings ((A,B) and (B,A) both appearing) — passing df twice as
    index(df, df) does neither, since that tells recordlinkage to treat them
    as two separate datasets.
    """
    if blocking not in BLOCKING_MODES:
        raise ValueError(f"blocking must be one of {BLOCKING_MODES}, got {blocking!r}")

    indexer = recordlinkage.Index()
    indexer.block(left_on="date_of_birth_clean")
    dob_pairs = indexer.index(df)
    if blocking == "dob":
        return dob_pairs, dob_pairs

    keyed = add_blocking_keys(df)
    name_year = recordlinkage.Index()
    name_year.block(left_on=["block_name_key", "block_city", "block_birth_year"])
    name_month_day = recordlinkage.Index()
    name_month_day.block(left_on=["block_name_key", "block_city", "block_birth_month_day"])
    typo_pairs = union_candidate_pairs(df, name_year.index(keyed), name_month_day.index(keyed))
    return dob_pairs, union_candidate_pairs(df, dob_pairs, filter_dob_typo_pairs(keyed, typo_pairs))


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

        records.append(
            {
                "beneficiary_id_a": row_a["beneficiary_id"],
                "beneficiary_id_b": row_b["beneficiary_id"],
                "name_score": name_score,
                "address_score": address_score,
                "combined_score": combined,
            }
        )
    return pd.DataFrame(records)


# ── OFFLINE ROBUSTNESS CHECK ────────────────────────────────────────────


def is_iso_date(text) -> bool:
    """True when text is a real calendar date written exactly as YYYY-MM-DD."""
    if not isinstance(text, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def dob_typo(dob: str, rng: random.Random) -> str:
    """Replaces one digit of a YYYY-MM-DD date with a different digit so the
    result is still a valid calendar date — a keying error of the kind a
    clerk makes, not a random new birthday. Picks uniformly among all such
    one-digit variants (enumerated, so there is no retry loop to get stuck
    in). Raises ValueError for anything that is not a valid ISO date.
    """
    if not is_iso_date(dob):
        raise ValueError(f"dob_typo needs a valid YYYY-MM-DD date, got {dob!r}")
    variants = []
    for pos, ch in enumerate(dob):
        if not ch.isdigit():
            continue
        for digit in "0123456789":
            candidate = dob[:pos] + digit + dob[pos + 1 :]
            if digit != ch and is_iso_date(candidate):
                variants.append(candidate)
    return rng.choice(variants)  # never empty: the year's last digit can always change


def inject_dob_typos(df: pd.DataFrame, beneficiary_ids, seed: int) -> pd.DataFrame:
    """Evaluation only: gives each listed beneficiary a one-digit DOB typo.
    Applied to the planted duplicates to measure what each blocking strategy
    would recover if the duplicate's DOB had been mistyped — the generator
    copies DOB verbatim, so the real data cannot show this on its own.
    Rows whose DOB is missing or not a valid YYYY-MM-DD date are left as-is.
    """
    rng = random.Random(seed)
    out = df.copy()
    dob = out["date_of_birth_clean"].map(normalize_dob)
    targets = out["beneficiary_id"].isin(set(beneficiary_ids)) & dob.map(is_iso_date)
    out.loc[targets, "date_of_birth_clean"] = [dob_typo(v, rng) for v in dob[targets]]
    return out


# ── THRESHOLD TUNING AGAINST GROUND TRUTH ───────────────────────────────


def evaluate_threshold(scored: pd.DataFrame, ground_truth_pairs: set, threshold: float) -> dict:
    flagged = scored[scored["combined_score"] >= threshold]
    flagged_pairs = {frozenset({row["beneficiary_id_a"], row["beneficiary_id_b"]}) for _, row in flagged.iterrows()}

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
    print(
        f"True pair scores -> min={true_scores.min():.3f} median={true_scores.median():.3f} max={true_scores.max():.3f}"
    )
    print(
        f"Non-match scores -> min={false_scores.min():.3f} median={false_scores.median():.3f} "
        f"p90={false_scores.quantile(0.9):.3f} max={false_scores.max():.3f}"
    )
    print("--- end report ---\n")


def summarize_extra_pairs(scored: pd.DataFrame, base_pairs, ground_truth_pairs: set, threshold: float) -> dict:
    """Describes the scored pairs that are NOT in base_pairs (in multi mode:
    the pairs only pass (b) produced) — how many there are, how many are true
    duplicates, the best-scoring stranger among them and how close it sits to
    the threshold, and how many reach MIN_SCORE_TO_STORE (stored, unflagged
    unless >= threshold, in DB mode). This is the margin the extra pass leaves
    before it starts costing precision.
    """
    base = {frozenset(p) for p in base_pairs}
    keys = [frozenset(p) for p in zip(scored["beneficiary_id_a"], scored["beneficiary_id_b"])]
    in_base = np.array([k in base for k in keys], dtype=bool)
    is_true = np.array([k in ground_truth_pairs for k in keys], dtype=bool)
    extra = scored[~in_base]
    strangers = scored[~in_base & ~is_true].sort_values("combined_score", ascending=False)
    top = strangers.iloc[0] if len(strangers) else None
    return {
        "n_extra": len(extra),
        "n_extra_true": int((~in_base & is_true).sum()),
        "top_stranger_score": float(top["combined_score"]) if top is not None else None,
        "top_stranger_ids": (top["beneficiary_id_a"], top["beneficiary_id_b"]) if top is not None else None,
        "n_extra_stored": int((extra["combined_score"] >= MIN_SCORE_TO_STORE).sum()),
        "n_extra_strangers_flagged": int((strangers["combined_score"] >= threshold).sum()),
    }


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


def run(blocking: str, from_csv: str | None, dry_run: bool, ground_truth: str, simulate_dob_typos: int | None):
    ground_truth_pairs = load_ground_truth_pairs(Path(ground_truth))

    # --from-csv never opens a connection, so it can never write either
    conn = None if from_csv else get_connection()
    try:
        if from_csv:
            beneficiaries = load_beneficiaries_csv(Path(from_csv))
            print(f"Loaded {len(beneficiaries)} rows from {from_csv} (offline; nothing is written to the DB).")
        else:
            beneficiaries = load_beneficiaries(conn)
        deduped = dedupe_exact_rows(beneficiaries)

        if simulate_dob_typos is not None:
            gt = pd.read_csv(ground_truth)
            duplicate_ids = set(gt.loc[gt["fraud_type"] == "duplicate_identity", "duplicate_beneficiary_id"])
            deduped = inject_dob_typos(deduped, duplicate_ids, seed=simulate_dob_typos)
            print(
                f"Simulated a one-digit DOB typo on {len(duplicate_ids)} planted duplicates (seed={simulate_dob_typos})."
            )

        dob_pairs, pairs = generate_blocking_passes(deduped, blocking=blocking)
        reachable = {frozenset(p) for p in pairs} & ground_truth_pairs
        print(
            f"Candidate pairs ({blocking} blocking): {len(pairs)} "
            f"(true pairs among candidates: {len(reachable)}/{len(ground_truth_pairs)})"
        )

        scored = score_pairs(deduped, pairs)
        print_score_separation_report(scored, ground_truth_pairs)

        best = find_best_threshold(scored, ground_truth_pairs)
        print(
            f"Best threshold: {best['threshold']} "
            f"(precision={best['precision']:.3f}, recall={best['recall']:.3f}, f1={best['f1']:.3f}, "
            f"flagged={best['n_flagged']}, true_positives={best['n_true_positives']}/{len(ground_truth_pairs)})"
        )

        if blocking == "multi":
            extra = summarize_extra_pairs(scored, dob_pairs, ground_truth_pairs, best["threshold"])
            top = (
                f"{extra['top_stranger_score']:.3f} ({extra['top_stranger_ids'][0]}/{extra['top_stranger_ids'][1]})"
                if extra["top_stranger_ids"] is not None
                else "n/a"
            )
            print(
                f"Pairs added by the DOB-typo pass: {extra['n_extra']} "
                f"(true duplicates: {extra['n_extra_true']}; best stranger: {top} vs threshold {best['threshold']}; "
                f"strangers flagged: {extra['n_extra_strangers_flagged']}; "
                f"scoring >= {MIN_SCORE_TO_STORE}, stored in DB mode: {extra['n_extra_stored']})"
            )

        if conn is None or dry_run:
            print("Dry run: record_linkage_matches not written.")
            return best

        n_written = write_matches(conn, scored, threshold=best["threshold"])
        print(
            f"Wrote {n_written} candidate pairs to record_linkage_matches "
            f"(scored >= {MIN_SCORE_TO_STORE}, flagged at >= {best['threshold']})."
        )
        return best
    finally:
        if conn is not None:
            conn.close()


def main():
    parser = argparse.ArgumentParser(description="Stage 6: fuzzy record linkage for duplicate identities.")
    parser.add_argument(
        "--blocking",
        choices=BLOCKING_MODES,
        default="multi",
        help="multi (default): DOB block + DOB-typo pass (name initials + city, DOBs one typo apart). "
        "dob: DOB block only.",
    )
    parser.add_argument(
        "--from-csv",
        nargs="?",
        const=str(BENEFICIARIES_CSV_PATH),
        default=None,
        metavar="PATH",
        help=f"Read beneficiaries from a cleaned CSV (default {BENEFICIARIES_CSV_PATH}) instead of Postgres. "
        "Implies --dry-run: no DB connection is opened.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Score and evaluate without writing to the DB (already implied by --from-csv).",
    )
    parser.add_argument("--ground-truth", type=str, default=str(GROUND_TRUTH_PATH))
    parser.add_argument(
        "--simulate-dob-typos",
        type=int,
        default=None,
        metavar="SEED",
        help="Evaluation only, requires --from-csv: put a one-digit DOB typo on every planted duplicate first.",
    )
    args = parser.parse_args()
    if args.simulate_dob_typos is not None and not args.from_csv:
        parser.error("--simulate-dob-typos is evaluation-only and requires --from-csv")

    run(args.blocking, args.from_csv, args.dry_run, args.ground_truth, args.simulate_dob_typos)


if __name__ == "__main__":
    main()
