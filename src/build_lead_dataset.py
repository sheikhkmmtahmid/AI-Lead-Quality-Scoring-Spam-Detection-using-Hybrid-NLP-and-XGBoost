import os
import re
import random
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 1. CONFIG

BASE_PATH = r"D:\AI-Powered Lead and Contact Form Quality Scoring System\Dataset\final databases"
OUTPUT_PATH = r"D:\AI-Powered Lead and Contact Form Quality Scoring System\outputs"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

TARGET_LEADS = 30000
COMPANY_SAMPLE_SIZE = 20000
MESSAGE_SAMPLE_SIZE = 30000

SHORT_TEXT_MAX_TOKENS = 256
LONG_TEXT_MAX_TOKENS = 512

# 2. GENERIC HELPERS

def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    encodings = [None, "utf-8", "latin1", "cp1252"]
    last_error = None

    for enc in encodings:
        try:
            if enc is None:
                return pd.read_csv(path, **kwargs)
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Failed to read {path}. Last error: {last_error}")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_token_count(text: object) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    return len(text.split())


def has_url(text: object) -> int:
    text = normalize_text(text).lower()
    return int(bool(re.search(r"(http[s]?://|www\.)", text)))


def has_email(text: object) -> int:
    text = normalize_text(text)
    return int(bool(re.search(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", text)))


def has_phone(text: object) -> int:
    text = normalize_text(text)
    return int(bool(re.search(r"(\+?\d[\d\-\s\(\)]{7,}\d)", text)))


def urgency_score(text: object) -> int:
    text = normalize_text(text).lower()
    keywords = [
        "urgent", "immediately", "asap", "now", "limited time",
        "act now", "final notice", "deadline", "important",
        "click now", "verify", "confirm", "update"
    ]
    return sum(1 for kw in keywords if kw in text)


def clean_enron_message(text: object) -> str:
    text = normalize_text(text)

    split_markers = [
        "-----Original Message-----",
        "From:",
        "Sent:",
        "To:",
        "Subject:",
        "Forwarded by",
        "----- Forwarded by"
    ]

    for marker in split_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    text = re.sub(r"\s+", " ", text).strip()
    return text


def bucket_company_size(value: object) -> str:
    if pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()

    known_labels = {
        "1-10", "11-50", "51-200", "201-500", "501-1000",
        "1001-5000", "5001-10000", "10000+",
        "small", "medium", "large", "enterprise"
    }
    if text in known_labels:
        return text

    nums = re.findall(r"\d+", text)
    if nums:
        num = int(nums[0])
        if num <= 10:
            return "1-10"
        if num <= 50:
            return "11-50"
        if num <= 200:
            return "51-200"
        if num <= 500:
            return "201-500"
        if num <= 1000:
            return "501-1000"
        if num <= 5000:
            return "1001-5000"
        if num <= 10000:
            return "5001-10000"
        return "10000+"

    return text if text else "unknown"


def parse_date_safe(series: pd.Series, dayfirst: bool = True) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)


# 3. LOAD COMPANIES HOUSE


def load_companies_house(part_filename: str) -> pd.DataFrame:
    path = os.path.join(BASE_PATH, part_filename)
    df = safe_read_csv(path)
    df = clean_columns(df)

    keep_cols = [
        "CompanyName",
        "CompanyNumber",
        "CompanyCategory",
        "CompanyStatus",
        "RegAddress.Country",
        "RegAddress.County",
        "RegAddress.PostTown",
        "RegAddress.PostCode",
        "CountryOfOrigin",
        "IncorporationDate",
        "DissolutionDate",
        "SICCode.SicText_1",
        "SICCode.SicText_2",
        "SICCode.SicText_3",
        "SICCode.SicText_4",
        "Accounts.AccountCategory"
    ]

    existing_cols = [c for c in keep_cols if c in df.columns]
    df = df[existing_cols].copy()

    rename_map = {
        "CompanyName": "company_name",
        "CompanyNumber": "company_number",
        "CompanyCategory": "company_category",
        "CompanyStatus": "company_status",
        "RegAddress.Country": "country",
        "RegAddress.County": "county",
        "RegAddress.PostTown": "post_town",
        "RegAddress.PostCode": "post_code",
        "CountryOfOrigin": "country_of_origin",
        "IncorporationDate": "incorporation_date",
        "DissolutionDate": "dissolution_date",
        "SICCode.SicText_1": "industry_1",
        "SICCode.SicText_2": "industry_2",
        "SICCode.SicText_3": "industry_3",
        "SICCode.SicText_4": "industry_4",
        "Accounts.AccountCategory": "account_category"
    }

    df = df.rename(columns=rename_map)

    df["company_name"] = df["company_name"].map(normalize_text)
    df["incorporation_date"] = parse_date_safe(df["incorporation_date"], dayfirst=True)
    df["dissolution_date"] = parse_date_safe(df["dissolution_date"], dayfirst=True)

    today = pd.Timestamp.today().normalize()
    df["company_age_years"] = ((today - df["incorporation_date"]).dt.days / 365.25).round(1)
    df["company_age_years"] = df["company_age_years"].fillna(0)

    df["primary_industry"] = (
        df[["industry_1", "industry_2", "industry_3", "industry_4"]]
        .bfill(axis=1)
        .iloc[:, 0]
        .fillna("unknown")
    )

    text_cols = [
        "company_name", "company_category", "company_status", "country",
        "county", "post_town", "post_code", "country_of_origin",
        "industry_1", "industry_2", "industry_3", "industry_4",
        "primary_industry", "account_category"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].map(normalize_text)

    df = df[df["company_name"] != ""].copy()
    df = df.drop_duplicates(subset=["company_name", "company_number"]).reset_index(drop=True)
    df["company_id"] = np.arange(1, len(df) + 1)

    return df


# 4. LOAD ENRICHMENT DATA

def load_linkedin_companies() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, "LinkedIn-company-info.csv")
    df = safe_read_csv(path)
    df = clean_columns(df)

    keep = ["name", "industries", "company_size", "country_code", "formatted_locations", "founded", "website"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    df = df.rename(columns={
        "name": "company_name",
        "industries": "linkedin_industries",
        "company_size": "linkedin_company_size",
        "country_code": "linkedin_country_code",
        "formatted_locations": "linkedin_locations",
        "founded": "linkedin_founded",
        "website": "linkedin_website"
    })

    for col in df.columns:
        df[col] = df[col].map(normalize_text)

    df = df[df["company_name"] != ""].drop_duplicates(subset=["company_name"]).reset_index(drop=True)
    return df


def load_zoominfo_companies() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, "Zoominfo-companies-information.csv")
    df = safe_read_csv(path)
    df = clean_columns(df)

    keep = ["name", "industry", "employees", "total_employees", "revenue", "headquarters", "website", "tech_stack"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    df = df.rename(columns={
        "name": "company_name",
        "industry": "zoominfo_industry",
        "employees": "zoominfo_employees",
        "total_employees": "zoominfo_total_employees",
        "revenue": "zoominfo_revenue",
        "headquarters": "zoominfo_headquarters",
        "website": "zoominfo_website",
        "tech_stack": "zoominfo_tech_stack"
    })

    for col in df.columns:
        df[col] = df[col].map(normalize_text)

    df = df[df["company_name"] != ""].drop_duplicates(subset=["company_name"]).reset_index(drop=True)
    return df


def load_slintel_companies() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, "Slintel-6sense-company-information.csv")
    df = safe_read_csv(path)
    df = clean_columns(df)

    keep = ["name", "industries", "num_employees", "location", "region", "country_code", "website", "techstack_arr"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    df = df.rename(columns={
        "name": "company_name",
        "industries": "slintel_industries",
        "num_employees": "slintel_num_employees",
        "location": "slintel_location",
        "region": "slintel_region",
        "country_code": "slintel_country_code",
        "website": "slintel_website",
        "techstack_arr": "slintel_techstack"
    })

    for col in df.columns:
        df[col] = df[col].map(normalize_text)

    df = df[df["company_name"] != ""].drop_duplicates(subset=["company_name"]).reset_index(drop=True)
    return df


# 5. BUILD COMPANY TABLE

def build_companies_table() -> pd.DataFrame:
    companies_house = load_companies_house("BasicCompanyData-2026-03-02-part3_7.csv")
    linkedin = load_linkedin_companies()
    zoominfo = load_zoominfo_companies()
    slintel = load_slintel_companies()

    companies = companies_house.copy()
    companies = companies.merge(linkedin, on="company_name", how="left")
    companies = companies.merge(zoominfo, on="company_name", how="left")
    companies = companies.merge(slintel, on="company_name", how="left")

    companies["industry"] = companies["primary_industry"]

    companies["company_size_raw"] = companies["linkedin_company_size"].replace("", np.nan)
    if "zoominfo_total_employees" in companies.columns:
        companies["company_size_raw"] = companies["company_size_raw"].fillna(companies["zoominfo_total_employees"])
    if "zoominfo_employees" in companies.columns:
        companies["company_size_raw"] = companies["company_size_raw"].fillna(companies["zoominfo_employees"])
    if "slintel_num_employees" in companies.columns:
        companies["company_size_raw"] = companies["company_size_raw"].fillna(companies["slintel_num_employees"])

    companies["company_size"] = companies["company_size_raw"].map(bucket_company_size)

    companies["location"] = companies["post_town"].replace("", np.nan)
    if "zoominfo_headquarters" in companies.columns:
        companies["location"] = companies["location"].fillna(companies["zoominfo_headquarters"])
    if "slintel_location" in companies.columns:
        companies["location"] = companies["location"].fillna(companies["slintel_location"])
    if "linkedin_locations" in companies.columns:
        companies["location"] = companies["location"].fillna(companies["linkedin_locations"])
    companies["location"] = companies["location"].fillna("unknown")

    final_cols = [
        "company_id",
        "company_name",
        "company_number",
        "company_status",
        "company_category",
        "country",
        "county",
        "post_town",
        "post_code",
        "country_of_origin",
        "company_age_years",
        "industry",
        "company_size",
        "location",
        "account_category",
        "linkedin_website",
        "zoominfo_website",
        "slintel_website"
    ]

    final_cols = [c for c in final_cols if c in companies.columns]
    companies = companies[final_cols].copy()

    for col in companies.columns:
        if companies[col].dtype == "object":
            companies[col] = companies[col].fillna("unknown").map(normalize_text)

    return companies


# 6. LOAD MESSAGE DATA

def load_sms_spam_collection() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, "SMSSpamCollection.csv")
    df = safe_read_csv(path, header=None, names=["label", "message_text"])
    df = clean_columns(df)

    df["source_dataset"] = "sms_spam_collection"
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message_text"] = df["message_text"].map(normalize_text)
    return df


def load_smishing_dataset(filename: str, source_name: str) -> pd.DataFrame:
    path = os.path.join(BASE_PATH, filename)
    df = safe_read_csv(path)
    df = clean_columns(df)

    df = df.rename(columns={
        "LABEL": "label",
        "TEXT": "message_text",
        "URL": "url_flag_raw",
        "EMAIL": "email_flag_raw",
        "PHONE": "phone_flag_raw"
    })

    keep = ["label", "message_text", "url_flag_raw", "email_flag_raw", "phone_flag_raw"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    df["source_dataset"] = source_name
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message_text"] = df["message_text"].map(normalize_text)

    for raw_col, clean_col in [
        ("url_flag_raw", "has_url_dataset"),
        ("email_flag_raw", "has_email_dataset"),
        ("phone_flag_raw", "has_phone_dataset")
    ]:
        if raw_col in df.columns:
            df[clean_col] = pd.to_numeric(df[raw_col], errors="coerce").fillna(0).astype(int)
        else:
            df[clean_col] = 0

    return df


def load_enron_messages() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, "enron_spam_data.csv")
    df = safe_read_csv(path)
    df = clean_columns(df)

    keep = ["Message ID", "Subject", "Message", "Spam/Ham", "Date"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    df = df.rename(columns={
        "Message ID": "raw_message_id",
        "Subject": "subject",
        "Message": "message_text",
        "Spam/Ham": "label",
        "Date": "message_date"
    })

    df["subject"] = df["subject"].map(normalize_text)
    df["message_text"] = df["message_text"].map(clean_enron_message)
    df["message_text"] = np.where(
        df["message_text"] == "",
        df["subject"],
        df["subject"] + " " + df["message_text"]
    )
    df["message_text"] = pd.Series(df["message_text"]).map(normalize_text)

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["source_dataset"] = "enron"

    df = df[df["message_text"] != ""].copy()
    return df


def build_messages_table() -> pd.DataFrame:
    sms = load_sms_spam_collection()
    ds_10191 = load_smishing_dataset("Dataset_10191.csv", "dataset_10191")
    ds_5971 = load_smishing_dataset("Dataset_5971.csv", "dataset_5971")
    enron = load_enron_messages()

    messages = pd.concat([sms, ds_10191, ds_5971, enron], ignore_index=True, sort=False)
    messages["message_text"] = messages["message_text"].map(normalize_text)
    messages = messages[messages["message_text"] != ""].copy()

    label_map = {
        "ham": "ham",
        "spam": "spam",
        "smishing": "phishing",
        "phishing": "phishing"
    }
    messages["label"] = messages["label"].map(lambda x: label_map.get(str(x).lower(), str(x).lower()))

    messages["message_length_chars"] = messages["message_text"].str.len()
    messages["token_count"] = messages["message_text"].map(simple_token_count)

    messages["has_url_regex"] = messages["message_text"].map(has_url)
    messages["has_email_regex"] = messages["message_text"].map(has_email)
    messages["has_phone_regex"] = messages["message_text"].map(has_phone)
    messages["urgency_score"] = messages["message_text"].map(urgency_score)

    messages["has_url"] = messages[["has_url_dataset", "has_url_regex"]].max(axis=1)
    messages["has_email"] = messages[["has_email_dataset", "has_email_regex"]].max(axis=1)
    messages["has_phone"] = messages[["has_phone_dataset", "has_phone_regex"]].max(axis=1)

    messages["needs_sliding_window"] = (messages["token_count"] > SHORT_TEXT_MAX_TOKENS).astype(int)
    messages["too_long_for_single_512_pass"] = (messages["token_count"] > LONG_TEXT_MAX_TOKENS).astype(int)

    messages = messages.reset_index(drop=True)
    messages["message_id"] = np.arange(1, len(messages) + 1)

    final_cols = [
        "message_id",
        "source_dataset",
        "label",
        "message_text",
        "message_length_chars",
        "token_count",
        "has_url",
        "has_email",
        "has_phone",
        "urgency_score",
        "needs_sliding_window",
        "too_long_for_single_512_pass"
    ]
    return messages[final_cols].copy()


# 7. LOAD REVIEW DATA

def load_yelp_reviews() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, "Yelp-businesses-reviews.csv")
    df = safe_read_csv(path)
    df = clean_columns(df)

    keep = ["business_name", "Rating", "Content", "Date"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    df = df.rename(columns={
        "business_name": "company_name",
        "Rating": "review_rating",
        "Content": "review_text",
        "Date": "review_date"
    })

    df["source_dataset"] = "yelp"
    df["company_name"] = df["company_name"].map(normalize_text)
    df["review_text"] = df["review_text"].map(normalize_text)
    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    df["review_token_count"] = df["review_text"].map(simple_token_count)

    df = df[(df["company_name"] != "") & (df["review_text"] != "")].copy()
    return df


def load_trustpilot_reviews() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, "Trustpilot-business-reviews.csv")
    df = safe_read_csv(path)
    df = clean_columns(df)

    keep = [
        "company_name", "review_rating", "review_title", "review_content",
        "review_date", "company_overall_rating", "company_total_reviews",
        "company_country", "company_category"
    ]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    df = df.rename(columns={
        "review_content": "review_text",
        "company_overall_rating": "company_overall_rating_snapshot",
        "company_total_reviews": "company_total_reviews_snapshot",
        "company_country": "review_company_country",
        "company_category": "review_company_category"
    })

    df["source_dataset"] = "trustpilot"
    df["company_name"] = df["company_name"].map(normalize_text)
    df["review_text"] = df["review_text"].map(normalize_text)
    df["review_title"] = df["review_title"].map(normalize_text)
    df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
    df["company_overall_rating_snapshot"] = pd.to_numeric(df["company_overall_rating_snapshot"], errors="coerce")
    df["company_total_reviews_snapshot"] = pd.to_numeric(df["company_total_reviews_snapshot"], errors="coerce")
    df["review_token_count"] = df["review_text"].map(simple_token_count)

    df = df[(df["company_name"] != "") & (df["review_text"] != "")].copy()
    return df


def build_reviews_table() -> pd.DataFrame:
    yelp = load_yelp_reviews()
    trustpilot = load_trustpilot_reviews()

    reviews = pd.concat([yelp, trustpilot], ignore_index=True, sort=False)
    reviews = reviews.reset_index(drop=True)
    reviews["review_id"] = np.arange(1, len(reviews) + 1)
    return reviews


def build_company_review_features(reviews: pd.DataFrame) -> pd.DataFrame:
    grp = reviews.groupby("company_name", dropna=False).agg(
        avg_review_score=("review_rating", "mean"),
        review_count=("review_id", "count"),
        avg_review_length_tokens=("review_token_count", "mean"),
    ).reset_index()

    grp["avg_review_score"] = grp["avg_review_score"].round(2)
    grp["avg_review_length_tokens"] = grp["avg_review_length_tokens"].round(1)
    return grp


# 8. BUILD FINAL LEADS

def generate_email_from_company(company_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "", company_name.lower())
    slug = slug[:20] if slug else "company"
    domains = ["gmail.com", "outlook.com", "yahoo.com", "company.com", "business.co.uk"]
    return f"contact@{random.choice(domains)}" if slug == "company" else f"info@{slug}.com"


def domain_type_from_email(email: str) -> str:
    if not email or "@" not in email:
        return "unknown"
    domain = email.split("@")[-1].lower()
    free_domains = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com"}
    return "free" if domain in free_domains else "corporate"


def build_leads_table(companies: pd.DataFrame, messages: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    company_review_features = build_company_review_features(reviews)

    companies_enriched = companies.merge(
        company_review_features,
        on="company_name",
        how="left"
    )

    companies_enriched["avg_review_score"] = companies_enriched["avg_review_score"].fillna(3.0)
    companies_enriched["review_count"] = companies_enriched["review_count"].fillna(0).astype(int)
    companies_enriched["avg_review_length_tokens"] = companies_enriched["avg_review_length_tokens"].fillna(0)

    companies_sample = companies_enriched.sample(
        n=min(COMPANY_SAMPLE_SIZE, len(companies_enriched)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    messages_sample = messages.sample(
        n=min(MESSAGE_SAMPLE_SIZE, len(messages)),
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    picked_company_ids = np.random.choice(companies_sample["company_id"], size=TARGET_LEADS, replace=True)
    picked_message_ids = np.random.choice(messages_sample["message_id"], size=TARGET_LEADS, replace=True)

    leads = pd.DataFrame({
        "lead_id": np.arange(1, TARGET_LEADS + 1),
        "company_id": picked_company_ids,
        "message_id": picked_message_ids
    })

    leads = leads.merge(companies_sample, on="company_id", how="left")
    leads = leads.merge(messages_sample, on="message_id", how="left")

    leads["generated_contact_email"] = leads["company_name"].map(generate_email_from_company)
    leads["domain_type"] = leads["generated_contact_email"].map(domain_type_from_email)

    leads["spam_score_rule"] = np.select(
        [
            leads["label"].eq("spam"),
            leads["label"].eq("phishing"),
            leads["label"].eq("ham")
        ],
        [0.85, 1.00, 0.05],
        default=0.20
    )

    leads["risk_score_rule"] = (
        0.35 * leads["has_url"] +
        0.20 * leads["has_email"] +
        0.20 * leads["has_phone"] +
        0.10 * (leads["urgency_score"] > 0).astype(int) +
        0.15 * (leads["company_status"].str.lower() != "active").astype(int)
    ).round(3)

    leads["quality_score_rule"] = (
        0.25 * (leads["label"] == "ham").astype(int) +
        0.20 * (leads["company_status"].str.lower() == "active").astype(int) +
        0.20 * (leads["company_age_years"] >= 3).astype(int) +
        0.20 * (leads["avg_review_score"] >= 3.5).astype(int) +
        0.15 * (leads["domain_type"] == "corporate").astype(int)
    ).round(3)

    leads["recommended_action"] = np.select(
        [
            leads["risk_score_rule"] >= 0.75,
            (leads["spam_score_rule"] >= 0.80) & (leads["risk_score_rule"] < 0.75),
            leads["quality_score_rule"] >= 0.70
        ],
        [
            "manual_risk_review",
            "block_or_quarantine",
            "send_to_sales"
        ],
        default="needs_review"
    )

    final_cols = [
        "lead_id",
        "company_id",
        "message_id",
        "company_name",
        "company_number",
        "company_status",
        "company_category",
        "industry",
        "company_size",
        "country",
        "location",
        "company_age_years",
        "avg_review_score",
        "review_count",
        "message_text",
        "label",
        "source_dataset",
        "message_length_chars",
        "token_count",
        "has_url",
        "has_email",
        "has_phone",
        "urgency_score",
        "needs_sliding_window",
        "too_long_for_single_512_pass",
        "generated_contact_email",
        "domain_type",
        "spam_score_rule",
        "risk_score_rule",
        "quality_score_rule",
        "recommended_action"
    ]

    return leads[final_cols].copy()


# 9. MAIN

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Loading companies...")
    companies = build_companies_table()
    print(f"Companies shape: {companies.shape}")

    print("Loading messages...")
    messages = build_messages_table()
    print(f"Messages shape: {messages.shape}")

    print("Loading reviews...")
    reviews = build_reviews_table()
    print(f"Reviews shape: {reviews.shape}")

    print("Building leads...")
    leads = build_leads_table(companies, messages, reviews)
    print(f"Leads shape: {leads.shape}")

    companies.to_csv(os.path.join(OUTPUT_PATH, "companies_clean.csv"), index=False)
    messages.to_csv(os.path.join(OUTPUT_PATH, "messages_clean.csv"), index=False)
    reviews.to_csv(os.path.join(OUTPUT_PATH, "reviews_clean.csv"), index=False)
    leads.to_csv(os.path.join(OUTPUT_PATH, "leads_final.csv"), index=False)

    print("\nSaved:")
    print(os.path.join(OUTPUT_PATH, "companies_clean.csv"))
    print(os.path.join(OUTPUT_PATH, "messages_clean.csv"))
    print(os.path.join(OUTPUT_PATH, "reviews_clean.csv"))
    print(os.path.join(OUTPUT_PATH, "leads_final.csv"))

    print("\nSliding window need summary:")
    print(leads["needs_sliding_window"].value_counts(dropna=False))

    print("\nLabel distribution:")
    print(leads["label"].value_counts(dropna=False))

    print("\nRecommended action distribution:")
    print(leads["recommended_action"].value_counts(dropna=False))


if __name__ == "__main__":
    main()