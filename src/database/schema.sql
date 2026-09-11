/*
    This SQL script defines the schema for the welfare fraud detection database. 
    It includes tables for beneficiaries, national IDs, and disbursements, along with their respective fields and constraints.
*/

-- =========================================================
-- Stage 4 Schema: Multi-Source Data Pipeline & Fraud Detection
-- =========================================================

CREATE TABLE beneficiaries (
    row_id                   BIGSERIAL PRIMARY KEY,
    beneficiary_id           TEXT,
    full_name                TEXT,
    father_or_husband_name   TEXT,
    gender                   TEXT,
    date_of_birth            TEXT,
    cnic                     TEXT,
    phone_number             TEXT,
    marital_status           TEXT,
    household_size           INTEGER,
    monthly_income_pkr       TEXT,
    address_line             TEXT,
    city                     TEXT,
    province                 TEXT,
    bank_account_number      TEXT,
    welfare_program          TEXT,
    registration_date        TEXT,
    registration_channel     TEXT,
    status                   TEXT,
    gender_clean             TEXT,
    status_clean             TEXT,
    date_of_birth_clean      DATE,
    dob_parse_failed         BOOLEAN DEFAULT FALSE,
    registration_date_clean  DATE,
    reg_date_parse_failed    BOOLEAN DEFAULT FALSE,
    monthly_income_clean     NUMERIC(12,2),
    income_flag              BOOLEAN DEFAULT FALSE,
    phone_missing            BOOLEAN DEFAULT FALSE,
    address_missing          BOOLEAN DEFAULT FALSE,
    is_duplicate_row         BOOLEAN DEFAULT FALSE,
    data_quality_flags       TEXT[]
);

CREATE INDEX idx_beneficiaries_bank_account ON beneficiaries (bank_account_number);
CREATE INDEX idx_beneficiaries_cnic ON beneficiaries (cnic);


CREATE TABLE national_id_records (
    national_id_no           TEXT PRIMARY KEY,
    full_name                TEXT,
    gender                   TEXT,
    date_of_birth            TEXT,
    address                  TEXT,
    id_issue_date            TEXT,
    id_status                TEXT,
    linked_beneficiary_id    TEXT,
    gender_clean             TEXT,
    date_of_birth_clean      DATE,
    dob_parse_failed         BOOLEAN DEFAULT FALSE,
    id_issue_date_clean      DATE,
    issue_date_parse_failed  BOOLEAN DEFAULT FALSE,
    address_missing          BOOLEAN DEFAULT FALSE,
    id_status_missing        BOOLEAN DEFAULT FALSE,
    no_linked_beneficiary    BOOLEAN DEFAULT FALSE,
    data_quality_flags       TEXT[]
);

CREATE INDEX idx_national_id_linked_ben ON national_id_records (linked_beneficiary_id);


CREATE TABLE disbursements (
    row_id                   BIGSERIAL PRIMARY KEY,
    disbursement_id          TEXT,
    beneficiary_id           TEXT,
    bank_account_number      TEXT,
    amount_pkr               TEXT,
    disbursement_cycle       TEXT,
    disbursement_date        TEXT,
    payment_method           TEXT,
    branch_or_agent_code     TEXT,
    status                   TEXT,
    status_clean             TEXT,
    disbursement_date_clean  DATE,
    disb_date_parse_failed   BOOLEAN DEFAULT FALSE,
    amount_pkr_clean         NUMERIC(12,2),
    amount_flag              BOOLEAN DEFAULT FALSE,
    payment_method_missing   BOOLEAN DEFAULT FALSE,
    data_quality_flags       TEXT[],
    shared_bank_account_flag BOOLEAN DEFAULT FALSE,
    orphaned_beneficiary_flag BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_disbursements_beneficiary ON disbursements (beneficiary_id);
CREATE INDEX idx_disbursements_bank_account ON disbursements (bank_account_number);


-- ---------- Fraud Detection Output Tables ----------

CREATE TABLE record_linkage_matches (
    match_id          SERIAL PRIMARY KEY,
    beneficiary_id    TEXT,
    national_id_no    TEXT,
    match_method      TEXT,
    confidence_score  NUMERIC(5,4),
    is_flagged        BOOLEAN DEFAULT FALSE,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_matches_beneficiary ON record_linkage_matches (beneficiary_id);
CREATE INDEX idx_matches_national_id ON record_linkage_matches (national_id_no);


CREATE TABLE fraud_signals (
    signal_id     SERIAL PRIMARY KEY,
    entity_id     TEXT,
    entity_type   TEXT,
    method        TEXT,
    signal_type   TEXT,
    score         NUMERIC(6,4),
    run_id        TEXT,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_fraud_signals_entity ON fraud_signals (entity_id);