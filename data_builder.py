import math
from dataclasses import dataclass
from typing import Iterable
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from config import (
    EMBEDDING_DIM,
    ITEM_EMBEDDINGS_PATH,
    ITEM_ID_ORDER_PATH,
    RANDOM_SEED,
    RETRIEVAL_TOP_K,
    SVD_PATH,
    VECTORIZER_PATH,
)
from database import write_df


CATEGORY_TO_SKILLS = {
    "data": ["sql", "python", "pandas", "statistics", "tableau", "ab testing", "airflow"],
    "ml": ["python", "machine learning", "pytorch", "tensorflow", "feature engineering", "mlops", "experimentation"],
    "backend": ["python", "java", "apis", "microservices", "postgres", "redis", "docker"],
    "frontend": ["javascript", "react", "typescript", "css", "ui", "web performance", "testing"],
    "product": ["roadmapping", "analytics", "stakeholder management", "sql", "experimentation", "communication", "prioritization"],
    "design": ["figma", "user research", "prototyping", "interaction design", "ux writing", "visual design", "accessibility"],
    "marketing": ["seo", "content", "analytics", "campaigns", "copywriting", "crm", "growth"],
    "sales": ["crm", "pipeline", "communication", "lead generation", "negotiation", "forecasting", "customer success"],
}

CATEGORY_TO_TITLES = {
    "data": ["Data Analyst", "Senior Data Analyst", "Analytics Engineer", "Business Intelligence Analyst"],
    "ml": ["Machine Learning Engineer", "Applied Scientist", "Recommendation Scientist", "ML Engineer"],
    "backend": ["Backend Engineer", "Software Engineer", "Platform Engineer", "API Engineer"],
    "frontend": ["Frontend Engineer", "Web Engineer", "UI Engineer", "React Developer"],
    "product": ["Product Manager", "Growth Product Manager", "Platform Product Manager", "Product Analyst"],
    "design": ["Product Designer", "UX Designer", "Interaction Designer", "UI Designer"],
    "marketing": ["Growth Marketer", "Product Marketing Manager", "Content Strategist", "SEO Specialist"],
    "sales": ["Account Executive", "Sales Development Rep", "Customer Success Manager", "Revenue Operations Analyst"],
}

LOCATIONS = [
    "San Francisco",
    "New York",
    "Seattle",
    "Austin",
    "Los Angeles",
    "Remote",
]

COMPANIES = [
    "Acme Labs",
    "Nova Systems",
    "Pioneer AI",
    "BrightPath",
    "Nimbus Tech",
    "BluePeak",
    "Vertex Works",
    "Northstar",
    "Atlas Cloud",
    "Signal House",
    "Orbit Data",
    "CraftFlow",
]

CATEGORY_BASE_SALARY = {
    "data": 90000,
    "ml": 140000,
    "backend": 130000,
    "frontend": 120000,
    "product": 135000,
    "design": 110000,
    "marketing": 95000,
    "sales": 100000,
}


@dataclass
class RetrievalArtifacts:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD
    item_embeddings: np.ndarray
    item_id_order: np.ndarray


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def to_skill_set(text: str) -> set[str]:
    return {token.strip() for token in text.split(",") if token.strip()}


def make_users(n_users: int = 80, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = list(CATEGORY_TO_SKILLS.keys())
    rows = []

    for user_id in range(1, n_users + 1):
        primary = rng.choice(categories)
        secondary = rng.choice([c for c in categories if c != primary])
        preferred_categories = [primary] if rng.random() < 0.7 else [primary, secondary]
        base_skills = []
        for cat in preferred_categories:
            base_skills.extend(rng.choice(CATEGORY_TO_SKILLS[cat], size=3, replace=False).tolist())
        base_skills = sorted(set(base_skills))
        wants_remote = int(rng.random() < 0.45)
        preferred_location = "Remote" if wants_remote and rng.random() < 0.55 else rng.choice(LOCATIONS[:-1])
        years_experience = int(rng.integers(0, 13))
        profile_text = (
            f"User wants {', '.join(preferred_categories)} jobs. "
            f"Skills: {', '.join(base_skills)}. "
            f"Location: {preferred_location}. Remote preference: {bool(wants_remote)}. "
            f"Experience: {years_experience} years."
        )
        rows.append(
            {
                "user_id": user_id,
                "years_experience": years_experience,
                "preferred_location": preferred_location,
                "preferred_categories": ", ".join(preferred_categories),
                "skills_text": ", ".join(base_skills),
                "wants_remote": wants_remote,
                "profile_text": profile_text,
            }
        )
    return pd.DataFrame(rows)


def make_items(n_items: int = 500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    categories = list(CATEGORY_TO_SKILLS.keys())
    rows = []

    for item_id in range(1, n_items + 1):
        category = rng.choice(categories)
        title = rng.choice(CATEGORY_TO_TITLES[category])
        company = rng.choice(COMPANIES)
        is_remote = int(rng.random() < 0.35)
        location = "Remote" if is_remote and rng.random() < 0.7 else rng.choice(LOCATIONS[:-1])
        required_years = int(rng.integers(0, 11))
        base_salary = CATEGORY_BASE_SALARY[category] + required_years * 5000
        salary_spread = int(rng.integers(15000, 40000))
        salary_min = base_salary + int(rng.integers(-10000, 10000))
        salary_max = salary_min + salary_spread
        days_since_posted = int(rng.integers(0, 61))
        is_active = int(rng.random() > 0.04)
        skills = rng.choice(CATEGORY_TO_SKILLS[category], size=4, replace=False).tolist()
        desc = (
            f"{title} at {company}. Looking for {category} experience with {', '.join(skills)}. "
            f"This role is based in {location}. Requires {required_years}+ years experience."
        )
        item_text = (
            f"{title}. {category}. {company}. Skills {', '.join(skills)}. "
            f"Location {location}. Remote {bool(is_remote)}. {desc}"
        )
        rows.append(
            {
                "item_id": item_id,
                "title": title,
                "company": company,
                "category": category,
                "location": location,
                "salary_min": float(salary_min),
                "salary_max": float(salary_max),
                "is_remote": is_remote,
                "is_active": is_active,
                "days_since_posted": days_since_posted,
                "required_years": required_years,
                "skills_text": ", ".join(skills),
                "description": desc,
                "item_text": item_text,
            }
        )
    return pd.DataFrame(rows)


def fit_retrieval_artifacts(items: pd.DataFrame) -> RetrievalArtifacts:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
    item_tfidf = vectorizer.fit_transform(items["item_text"])
    n_components = min(EMBEDDING_DIM, max(2, item_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    item_dense = svd.fit_transform(item_tfidf)
    item_embeddings = normalize(item_dense)
    item_id_order = items["item_id"].to_numpy()

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(svd, SVD_PATH)
    np.save(ITEM_EMBEDDINGS_PATH, item_embeddings)
    np.save(ITEM_ID_ORDER_PATH, item_id_order)

    return RetrievalArtifacts(
        vectorizer=vectorizer,
        svd=svd,
        item_embeddings=item_embeddings,
        item_id_order=item_id_order,
    )


def make_query_text(user_row: pd.Series, rng: np.random.Generator) -> str:
    skills = user_row["skills_text"].split(", ")
    preferred_categories = user_row["preferred_categories"]
    location = user_row["preferred_location"]
    query_templates = [
        f"{preferred_categories} jobs in {location}",
        f"{preferred_categories} roles with {skills[0]} and {skills[1]}",
        f"{preferred_categories} openings for {user_row['years_experience']} years experience",
        f"remote {preferred_categories} opportunities with {skills[0]}",
    ]
    return rng.choice(query_templates)


def embed_query(text: str, artifacts: RetrievalArtifacts) -> np.ndarray:
    query_tfidf = artifacts.vectorizer.transform([text])
    query_dense = artifacts.svd.transform(query_tfidf)
    return normalize(query_dense)[0]


def retrieve_top_candidates(
    query_text: str,
    items: pd.DataFrame,
    artifacts: RetrievalArtifacts,
    top_k: int = RETRIEVAL_TOP_K,
) -> pd.DataFrame:
    query_vec = embed_query(query_text, artifacts)
    scores = artifacts.item_embeddings @ query_vec
    top_idx = np.argpartition(scores, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    candidates = items.iloc[top_idx].copy().reset_index(drop=True)
    candidates["retrieval_score"] = scores[top_idx]
    return candidates


def expected_salary(years_experience: int, preferred_categories: str) -> float:
    primary = preferred_categories.split(", ")[0]
    return CATEGORY_BASE_SALARY[primary] + years_experience * 6000


def true_relevance_probability(user_row: pd.Series, item_row: pd.Series) -> float:
    user_skills = to_skill_set(user_row["skills_text"])
    item_skills = to_skill_set(item_row["skills_text"])
    overlap = len(user_skills & item_skills)
    union = max(len(user_skills | item_skills), 1)
    skill_overlap = overlap / max(len(item_skills), 1)
    skill_jaccard = overlap / union
    category_match = int(item_row["category"] in user_row["preferred_categories"].split(", "))
    location_match = int(item_row["location"] == user_row["preferred_location"])
    remote_match = int(bool(user_row["wants_remote"]) and bool(item_row["is_remote"]))
    experience_gap = abs(int(user_row["years_experience"]) - int(item_row["required_years"]))
    exp_fit = math.exp(-experience_gap / 4.0)
    expected = expected_salary(int(user_row["years_experience"]), user_row["preferred_categories"])
    salary_center = (float(item_row["salary_min"]) + float(item_row["salary_max"])) / 2.0
    salary_fit = 1.0 - min(abs(salary_center - expected) / max(expected, 1.0), 1.0)
    freshness = 1.0 - min(float(item_row["days_since_posted"]) / 60.0, 1.0)
    inactive_penalty = -2.0 if int(item_row["is_active"]) == 0 else 0.0

    logit = (
        -2.0
        + 2.0 * skill_overlap
        + 1.2 * skill_jaccard
        + 1.0 * category_match
        + 0.5 * location_match
        + 0.7 * remote_match
        + 0.6 * exp_fit
        + 0.4 * salary_fit
        + 0.3 * freshness
        + inactive_penalty
    )
    return float(sigmoid(logit))


def softmax(x: Iterable[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float) / max(temperature, 1e-6)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    return exp / exp.sum()


def sample_logged_ranking(
    candidates: pd.DataFrame,
    user_row: pd.Series,
    rng: np.random.Generator,
    display_k: int = 10,
) -> pd.DataFrame:
    work = candidates.head(30).copy().reset_index(drop=True)
    user_categories = set(user_row["preferred_categories"].split(", "))

    category_match = work["category"].isin(user_categories).astype(float)
    freshness = 1.0 - work["days_since_posted"] / 60.0
    noisy_score = (
        0.75 * work["retrieval_score"].to_numpy()
        + 0.25 * category_match.to_numpy()
        + 0.15 * freshness.to_numpy()
        + rng.normal(0.0, 0.15, size=len(work))
    )
    work["logging_score"] = noisy_score

    chosen_rows = []
    remaining = work.copy()
    for position in range(1, display_k + 1):
        probs = softmax(remaining["logging_score"], temperature=0.75)
        pick_idx = int(rng.choice(np.arange(len(remaining)), p=probs))
        picked = remaining.iloc[pick_idx].copy()
        picked["position"] = position
        picked["propensity"] = float(probs[pick_idx])
        chosen_rows.append(picked)
        remaining = remaining.drop(remaining.index[pick_idx]).reset_index(drop=True)

    ranked = pd.DataFrame(chosen_rows).reset_index(drop=True)
    click_probs = []
    clicks = []
    applies = []
    for _, row in ranked.iterrows():
        p = true_relevance_probability(user_row, row)
        p = float(sigmoid(-1.4 + 3.5 * p - 0.22 * row["position"]))
        click = int(rng.random() < p)
        apply = int(click and rng.random() < min(0.45, p))
        click_probs.append(p)
        clicks.append(click)
        applies.append(apply)

    ranked["oracle_click_prob"] = click_probs
    ranked["clicked"] = clicks
    ranked["applied"] = applies
    return ranked


def build_logged_data(
    users: pd.DataFrame,
    items: pd.DataFrame,
    artifacts: RetrievalArtifacts,
    requests_per_user: int = 4,
    seed: int = RANDOM_SEED,
) -> None:
    rng = np.random.default_rng(seed + 2)

    request_rows = []
    impression_rows = []
    click_rows = []
    interaction_rows = []
    label_rows = []

    start = pd.Timestamp("2025-01-01")
    request_counter = 0
    total_requests = len(users) * requests_per_user
    train_cut = int(total_requests * 0.6)
    valid_cut = int(total_requests * 0.8)

    for _, user_row in users.iterrows():
        for _ in range(requests_per_user):
            request_counter += 1
            request_id = str(uuid4())
            ts = start + pd.Timedelta(minutes=request_counter * 5)
            query_text = make_query_text(user_row, rng)
            retrieval_text = f"{user_row['profile_text']} {query_text}"
            candidates = retrieve_top_candidates(retrieval_text, items, artifacts)
            logged = sample_logged_ranking(candidates, user_row, rng)

            if request_counter <= train_cut:
                split_name = "train"
            elif request_counter <= valid_cut:
                split_name = "valid"
            else:
                split_name = "test"

            request_rows.append(
                {
                    "request_id": request_id,
                    "user_id": int(user_row["user_id"]),
                    "query_text": query_text,
                    "request_ts": ts.isoformat(),
                    "split_name": split_name,
                    "retrieved_count": len(candidates),
                    "ranked_count": len(logged),
                    "retrieval_ms": np.nan,
                    "ranking_ms": np.nan,
                    "source": "simulator",
                }
            )

            for _, row in logged.iterrows():
                impression_rows.append(
                    {
                        "request_id": request_id,
                        "user_id": int(user_row["user_id"]),
                        "item_id": int(row["item_id"]),
                        "position": int(row["position"]),
                        "retrieval_score": float(row["retrieval_score"]),
                        "ranking_score": float(row["logging_score"]),
                        "final_score": float(row["logging_score"]),
                        "propensity": float(row["propensity"]),
                        "impression_ts": ts.isoformat(),
                        "source": "simulator",
                    }
                )
                label_rows.append(
                    {
                        "request_id": request_id,
                        "user_id": int(user_row["user_id"]),
                        "item_id": int(row["item_id"]),
                        "label": int(row["clicked"]),
                        "source": "simulator_click",
                        "created_ts": ts.isoformat(),
                    }
                )
                interaction_rows.append(
                    {
                        "request_id": request_id,
                        "user_id": int(user_row["user_id"]),
                        "item_id": int(row["item_id"]),
                        "event_type": "impression",
                        "value": 1.0,
                        "event_ts": ts.isoformat(),
                        "source": "simulator",
                    }
                )
                if int(row["clicked"]) == 1:
                    click_rows.append(
                        {
                            "request_id": request_id,
                            "user_id": int(user_row["user_id"]),
                            "item_id": int(row["item_id"]),
                            "position": int(row["position"]),
                            "click_ts": ts.isoformat(),
                            "source": "simulator",
                        }
                    )
                    interaction_rows.append(
                        {
                            "request_id": request_id,
                            "user_id": int(user_row["user_id"]),
                            "item_id": int(row["item_id"]),
                            "event_type": "click",
                            "value": 1.0,
                            "event_ts": ts.isoformat(),
                            "source": "simulator",
                        }
                    )
                if int(row["applied"]) == 1:
                    interaction_rows.append(
                        {
                            "request_id": request_id,
                            "user_id": int(user_row["user_id"]),
                            "item_id": int(row["item_id"]),
                            "event_type": "apply",
                            "value": 1.0,
                            "event_ts": ts.isoformat(),
                            "source": "simulator",
                        }
                    )

    write_df("users", users, if_exists="append")
    write_df("items", items, if_exists="append")
    write_df("request_logs", pd.DataFrame(request_rows), if_exists="append")
    write_df("impressions", pd.DataFrame(impression_rows), if_exists="append")
    write_df("clicks", pd.DataFrame(click_rows), if_exists="append")
    write_df("interactions", pd.DataFrame(interaction_rows), if_exists="append")
    write_df("labels", pd.DataFrame(label_rows), if_exists="append")


def build_demo_data() -> None:
    users = make_users()
    items = make_items()
    artifacts = fit_retrieval_artifacts(items)
    build_logged_data(users, items, artifacts)