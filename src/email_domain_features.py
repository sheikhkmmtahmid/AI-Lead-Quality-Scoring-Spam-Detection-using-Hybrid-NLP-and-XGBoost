import re
import smtplib
import socket
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import dns.resolver


# CONFIG

EMAIL_REGEX = re.compile(
    r"^[A-Z0-9._%+\-']+@[A-Z0-9.\-]+\.[A-Z]{2,}$",
    re.IGNORECASE,
)

EMAIL_IN_TEXT_REGEX = re.compile(
    r"\b[A-Z0-9._%+\-']+@[A-Z0-9.\-]+\.[A-Z]{2,}\b",
    re.IGNORECASE,
)

URL_IN_TEXT_REGEX = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+)",
    re.IGNORECASE,
)

FREE_EMAIL_PROVIDERS = {
    "gmail.com",
    "googlemail.com",
    "yahoo.com",
    "yahoo.co.uk",
    "hotmail.com",
    "outlook.com",
    "live.com",
    "icloud.com",
    "aol.com",
    "protonmail.com",
    "proton.me",
    "gmx.com",
    "gmx.co.uk",
    "mail.com",
    "yandex.com",
    "yandex.ru",
    "zoho.com",
}

DISPOSABLE_EMAIL_PROVIDERS = {
    "mailinator.com",
    "10minutemail.com",
    "guerrillamail.com",
    "tempmail.com",
    "yopmail.com",
    "trashmail.com",
    "sharklasers.com",
    "dispostable.com",
    "getnada.com",
    "temp-mail.org",
}

DEFAULT_SMTP_TIMEOUT = 8
DEFAULT_DNS_TIMEOUT = 5


# DATA MODEL

@dataclass
class EmailDomainFeatures:
    input_contact_email: Optional[str]
    input_website_url: Optional[str]
    extracted_email_from_text: Optional[str]
    extracted_url_from_text: Optional[str]

    resolved_email: Optional[str]
    resolved_domain: Optional[str]

    email_syntax_valid: int
    domain_present: int
    domain_has_mx: int
    domain_has_a: int
    smtp_reachable: int
    smtp_mailbox_accepted: int

    is_free_provider: int
    is_disposable_provider: int
    domain_matches_company_name: int

    domain_type: str
    domain_trust_score: float

    smtp_response_code: Optional[int]
    smtp_response_message: Optional[str]
    smtp_exception: Optional[str]

    mx_hosts: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# LOW-LEVEL HELPERS

def safe_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    return value if value else None


def normalize_email(email: Optional[str]) -> Optional[str]:
    email = safe_str(email)
    if not email:
        return None
    return email.strip().lower()


def normalize_domain(domain: Optional[str]) -> Optional[str]:
    domain = safe_str(domain)
    if not domain:
        return None

    domain = domain.lower().strip()

    if domain.startswith("http://"):
        domain = domain[len("http://"):]
    elif domain.startswith("https://"):
        domain = domain[len("https://"):]

    if domain.startswith("www."):
        domain = domain[len("www."):]

    domain = domain.split("/", 1)[0]
    domain = domain.split("?", 1)[0]
    domain = domain.split("#", 1)[0]
    domain = domain.strip(" .")

    return domain if domain else None


def extract_email_from_text(text: Optional[str]) -> Optional[str]:
    text = safe_str(text)
    if not text:
        return None

    match = EMAIL_IN_TEXT_REGEX.search(text)
    return normalize_email(match.group(0)) if match else None


def extract_url_from_text(text: Optional[str]) -> Optional[str]:
    text = safe_str(text)
    if not text:
        return None

    match = URL_IN_TEXT_REGEX.search(text)
    return match.group(0).strip() if match else None


def extract_domain_from_email(email: Optional[str]) -> Optional[str]:
    email = normalize_email(email)
    if not email or "@" not in email:
        return None
    return normalize_domain(email.split("@", 1)[1])


def extract_domain_from_url(url: Optional[str]) -> Optional[str]:
    url = safe_str(url)
    if not url:
        return None
    return normalize_domain(url)


def is_valid_email_syntax(email: Optional[str]) -> int:
    email = normalize_email(email)
    if not email:
        return 0
    return int(bool(EMAIL_REGEX.fullmatch(email)))


def is_free_provider(domain: Optional[str]) -> int:
    domain = normalize_domain(domain)
    return int(domain in FREE_EMAIL_PROVIDERS) if domain else 0


def is_disposable_provider(domain: Optional[str]) -> int:
    domain = normalize_domain(domain)
    return int(domain in DISPOSABLE_EMAIL_PROVIDERS) if domain else 0


def classify_domain_type(domain: Optional[str]) -> str:
    if not domain:
        return "unknown"
    if domain in FREE_EMAIL_PROVIDERS:
        return "free"
    return "corporate"


def tokenize_company_name(company_name: Optional[str]) -> List[str]:
    company_name = safe_str(company_name)
    if not company_name:
        return []

    cleaned = re.sub(r"[^a-zA-Z0-9 ]+", " ", company_name.lower())
    tokens = [t for t in cleaned.split() if len(t) >= 3]

    stopwords = {
        "ltd", "limited", "llp", "plc", "inc", "corp", "co", "company",
        "group", "holdings", "services", "solutions", "international",
    }
    return [t for t in tokens if t not in stopwords]


def domain_matches_company_name(company_name: Optional[str], domain: Optional[str]) -> int:
    domain = normalize_domain(domain)
    if not company_name or not domain:
        return 0

    left_part = domain.split(".", 1)[0]
    left_part = re.sub(r"[^a-zA-Z0-9]+", "", left_part.lower())

    tokens = tokenize_company_name(company_name)
    if not tokens:
        return 0

    for token in tokens:
        if token in left_part or left_part in token:
            return 1
    return 0


# DNS HELPERS

def build_resolver(timeout: int = DEFAULT_DNS_TIMEOUT) -> dns.resolver.Resolver:
    resolver = dns.resolver.Resolver()
    resolver.timeout = timeout
    resolver.lifetime = timeout
    return resolver


def resolve_mx_hosts(domain: Optional[str], timeout: int = DEFAULT_DNS_TIMEOUT) -> List[str]:
    domain = normalize_domain(domain)
    if not domain:
        return []

    resolver = build_resolver(timeout=timeout)

    try:
        answers = resolver.resolve(domain, "MX")
        hosts = []
        for record in answers:
            exchange = str(record.exchange).rstrip(".").lower()
            if exchange:
                hosts.append(exchange)
        return sorted(set(hosts))
    except Exception:
        return []


def resolve_a_record_exists(domain: Optional[str], timeout: int = DEFAULT_DNS_TIMEOUT) -> int:
    domain = normalize_domain(domain)
    if not domain:
        return 0

    resolver = build_resolver(timeout=timeout)

    try:
        resolver.resolve(domain, "A")
        return 1
    except Exception:
        try:
            resolver.resolve(domain, "AAAA")
            return 1
        except Exception:
            return 0


# SMTP CHECK

def smtp_mailbox_check(
    email: Optional[str],
    company_name: Optional[str],
    timeout: int = DEFAULT_SMTP_TIMEOUT,
    helo_host: str = "localhost",
    mail_from: str = "validator@example.com",
) -> Dict[str, Any]:
    email = normalize_email(email)

    result = {
        "smtp_reachable": 0,
        "smtp_mailbox_accepted": 0,
        "smtp_response_code": None,
        "smtp_response_message": None,
        "smtp_exception": None,
    }

    if not email or not is_valid_email_syntax(email):
        result["smtp_exception"] = "invalid_email_syntax"
        return result

    domain = extract_domain_from_email(email)
    mx_hosts = resolve_mx_hosts(domain)

    if not mx_hosts:
        result["smtp_exception"] = "no_mx_record"
        return result

    smtp_host = mx_hosts[0]

    try:
        with smtplib.SMTP(timeout=timeout) as server:
            server.connect(smtp_host, 25)
            server.helo(helo_host)

            try:
                server.mail(mail_from)
                code, message = server.rcpt(email)
            except smtplib.SMTPServerDisconnected:
                result["smtp_exception"] = "server_disconnected"
                return result
            except smtplib.SMTPException as exc:
                result["smtp_exception"] = str(exc)
                return result

            result["smtp_reachable"] = 1
            result["smtp_response_code"] = int(code)
            result["smtp_response_message"] = (
                message.decode("utf-8", errors="ignore")
                if isinstance(message, bytes)
                else str(message)
            )

            if 200 <= int(code) < 300:
                result["smtp_mailbox_accepted"] = 1

            return result

    except (socket.timeout, TimeoutError):
        result["smtp_exception"] = "smtp_timeout"
        return result
    except Exception as exc:
        result["smtp_exception"] = str(exc)
        return result


# FEATURE BUILDING

def compute_domain_trust_score(
    email_syntax_valid: int,
    domain_present: int,
    domain_has_mx: int,
    domain_has_a: int,
    smtp_reachable: int,
    smtp_mailbox_accepted: int,
    is_free: int,
    is_disposable: int,
    company_match: int,
) -> float:
    score = 0.0

    if email_syntax_valid:
        score += 0.10
    if domain_present:
        score += 0.10
    if domain_has_mx:
        score += 0.20
    if domain_has_a:
        score += 0.10
    if smtp_reachable:
        score += 0.15
    if smtp_mailbox_accepted:
        score += 0.20
    if company_match:
        score += 0.15

    if is_free:
        score -= 0.10
    if is_disposable:
        score -= 0.40

    return round(max(0.0, min(1.0, score)), 4)


def build_email_domain_features(
    company_name: Optional[str],
    contact_email: Optional[str],
    website_url: Optional[str],
    message_text: Optional[str],
    enable_smtp_check: bool = True,
    smtp_timeout: int = DEFAULT_SMTP_TIMEOUT,
) -> Dict[str, Any]:
    input_contact_email = normalize_email(contact_email)
    input_website_url = safe_str(website_url)

    extracted_email = extract_email_from_text(message_text)
    extracted_url = extract_url_from_text(message_text)

    resolved_email = input_contact_email or extracted_email

    email_domain = extract_domain_from_email(resolved_email)
    url_domain = extract_domain_from_url(input_website_url or extracted_url)

    resolved_domain = email_domain or url_domain

    email_syntax_valid = is_valid_email_syntax(resolved_email)
    domain_present = int(bool(resolved_domain))
    mx_hosts = resolve_mx_hosts(resolved_domain)
    domain_has_mx = int(len(mx_hosts) > 0)
    domain_has_a = resolve_a_record_exists(resolved_domain)
    free_provider_flag = is_free_provider(resolved_domain)
    disposable_provider_flag = is_disposable_provider(resolved_domain)
    company_match_flag = domain_matches_company_name(company_name, resolved_domain)

    smtp_info = {
        "smtp_reachable": 0,
        "smtp_mailbox_accepted": 0,
        "smtp_response_code": None,
        "smtp_response_message": None,
        "smtp_exception": None,
    }

    if enable_smtp_check and resolved_email and email_syntax_valid:
        smtp_info = smtp_mailbox_check(
            email=resolved_email,
            company_name=company_name,
            timeout=smtp_timeout,
        )

    domain_type = classify_domain_type(resolved_domain)
    domain_trust_score = compute_domain_trust_score(
        email_syntax_valid=email_syntax_valid,
        domain_present=domain_present,
        domain_has_mx=domain_has_mx,
        domain_has_a=domain_has_a,
        smtp_reachable=int(smtp_info["smtp_reachable"]),
        smtp_mailbox_accepted=int(smtp_info["smtp_mailbox_accepted"]),
        is_free=free_provider_flag,
        is_disposable=disposable_provider_flag,
        company_match=company_match_flag,
    )

    features = EmailDomainFeatures(
        input_contact_email=input_contact_email,
        input_website_url=input_website_url,
        extracted_email_from_text=extracted_email,
        extracted_url_from_text=extracted_url,
        resolved_email=resolved_email,
        resolved_domain=resolved_domain,
        email_syntax_valid=email_syntax_valid,
        domain_present=domain_present,
        domain_has_mx=domain_has_mx,
        domain_has_a=domain_has_a,
        smtp_reachable=int(smtp_info["smtp_reachable"]),
        smtp_mailbox_accepted=int(smtp_info["smtp_mailbox_accepted"]),
        is_free_provider=free_provider_flag,
        is_disposable_provider=disposable_provider_flag,
        domain_matches_company_name=company_match_flag,
        domain_type=domain_type,
        domain_trust_score=domain_trust_score,
        smtp_response_code=smtp_info["smtp_response_code"],
        smtp_response_message=smtp_info["smtp_response_message"],
        smtp_exception=smtp_info["smtp_exception"],
        mx_hosts=";".join(mx_hosts) if mx_hosts else None,
    )

    return features.to_dict()


# DATAFRAME HELPERS

def infer_contact_email_from_row(row: Dict[str, Any]) -> Optional[str]:
    for key in [
        "contact_email",
        "email",
        "company_email",
        "generated_email",
    ]:
        value = row.get(key)
        if value and safe_str(value):
            return normalize_email(value)
    return None


def infer_website_url_from_row(row: Dict[str, Any]) -> Optional[str]:
    for key in [
        "website_url",
        "website",
        "company_website",
        "domain_url",
    ]:
        value = row.get(key)
        if value and safe_str(value):
            return safe_str(value)
    return None


def generate_email_domain_feature_frame(
    df,
    company_name_col: str = "company_name",
    message_text_col: str = "message_text",
    lead_id_col: Optional[str] = "lead_id",
    message_id_col: Optional[str] = "message_id",
    enable_smtp_check: bool = True,
) -> "pd.DataFrame":
    

    rows = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        feature_row = {
            "lead_id": row_dict.get(lead_id_col) if lead_id_col in row_dict else None,
            "message_id": row_dict.get(message_id_col) if message_id_col in row_dict else None,
        }

        computed = build_email_domain_features(
            company_name=row_dict.get(company_name_col),
            contact_email=infer_contact_email_from_row(row_dict),
            website_url=infer_website_url_from_row(row_dict),
            message_text=row_dict.get(message_text_col),
            enable_smtp_check=enable_smtp_check,
        )

        feature_row.update(computed)
        rows.append(feature_row)

    return pd.DataFrame(rows)