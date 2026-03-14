"""
Clustering Benchmark v2 — Public Datasets with DEV/TEST Splits

Uses 20 Newsgroups (sklearn) in two settings:
  - 20NG-diverse: 5 well-separated categories (easy)
  - 20NG-similar: 5 confusable comp.* categories (hard)

Supports:
  - Validator sweep on DEV split
  - Frozen evaluation on TEST split
  - Multiple baselines: algorithm-only, rules, random perturbation
  - Multiple validator configs: permissive, conservative, tuned
"""
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    homogeneity_score,
)

from ..core.validator import BaseValidator, ValidationResult
from ..datasets.newsgroups import Document


# ─── Configuration ───

TFIDF_MAX_FEATURES = 200
DOCS_PER_CATEGORY = 6      # 30 docs/instance for 5 categories
MAX_DOC_WORDS = 150         # Truncate document text in LLM prompt

# Validator sweep grid
SWEEP_BOOST_BOUNDS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0]
SWEEP_MAX_MODIFICATIONS = [10, 20, 30, 50, 100, 500]

# Predefined validator configs
VALIDATOR_CONFIGS = {
    "conservative": {"max_feature_boost": 0.3, "max_features_modified": 30},
    "permissive":   {"max_feature_boost": 2.0, "max_features_modified": 500},
}


# ─── Context Pools ───

DIVERSE_CONTEXTS = [
    ("Group these documents by their primary subject area. The collection "
     "contains posts from online forums covering sports, space science, "
     "computer graphics, political discussions, and religious topics."),
    ("Organize these forum posts for a content moderation system. Separate "
     "technical discussions from social and opinion-based discussions."),
    ("A digital library needs to categorize these texts. Focus on separating "
     "factual and scientific content from community and opinion content."),
    ("Cluster these documents for an information retrieval system. Users "
     "should find related posts grouped together."),
    ("Group these texts to detect distinct online communities. Each cluster "
     "should represent a unique interest area."),
]

SIMILAR_CONTEXTS = [
    ("These documents come from computer-related newsgroups. Group them by "
     "their specific sub-topic: graphics, Windows OS, IBM hardware, Mac "
     "hardware, or X Window System."),
    ("Organize these technical posts by the computing platform or technology "
     "they primarily discuss."),
    ("Cluster these forum messages for a tech support system. Users asking "
     "about different hardware and software systems should be separated."),
    ("Group these computer-related discussions by their most specific topic. "
     "Distinguish between operating systems, hardware platforms, and graphics."),
    ("Sort these posts into the most precise categories possible. Even though "
     "all are about computing, they discuss distinct sub-areas."),
]

EMPTY_CONTEXT = "Cluster these documents into groups based on content similarity."


# ─── Data Classes ───

@dataclass
class ValidatorConfig:
    max_feature_boost: float = 0.3
    max_features_modified: int = 30

    def to_dict(self) -> dict:
        return {"max_feature_boost": self.max_feature_boost,
                "max_features_modified": self.max_features_modified}


@dataclass
class ClusteringInstanceV2:
    instance_id: str
    documents: list[Document]
    n_clusters: int
    context: str
    context_type: str
    true_labels: list[int]
    seed: int = 0
    dataset_setting: str = "diverse"
    split: str = "test"


# ─── Instance Generation ───

def generate_instances(documents: list[Document],
                       n_instances: int,
                       n_clusters: int,
                       docs_per_category: int,
                       base_seed: int,
                       dataset_setting: str,
                       split: str,
                       context_pool: list[str]) -> list[ClusteringInstanceV2]:
    """Generate clustering instances by sampling documents.

    Each instance samples docs_per_category documents from each category,
    creating a balanced subset for clustering.
    """
    # Group documents by category
    by_category = {}
    for doc in documents:
        by_category.setdefault(doc.category_id, []).append(doc)

    categories = sorted(by_category.keys())
    min_docs = min(len(by_category[c]) for c in categories)
    if min_docs < docs_per_category:
        docs_per_category = min_docs

    instances = []
    for i in range(n_instances):
        seed = base_seed + i
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        # Sample documents
        sampled = []
        true_labels = []
        for cat_idx, cat_id in enumerate(categories):
            pool = by_category[cat_id]
            indices = np_rng.choice(len(pool), size=docs_per_category, replace=False)
            for idx in indices:
                sampled.append(pool[idx])
                true_labels.append(cat_idx)

        # Shuffle documents (so category blocks aren't contiguous)
        combined = list(zip(sampled, true_labels))
        rng.shuffle(combined)
        sampled, true_labels = zip(*combined)
        sampled = list(sampled)
        true_labels = list(true_labels)

        # Select context
        r = rng.random()
        if r < 0.7:
            ctx_type = "normal"
            context = rng.choice(context_pool)
        elif r < 0.8:
            ctx_type = "empty"
            context = EMPTY_CONTEXT
        elif r < 0.9:
            ctx_type = "noisy"
            base_ctx = rng.choice(context_pool)
            context = (
                f"{base_ctx} Also, the weather forecast says rain tomorrow. "
                f"Random number: {rng.randint(1, 9999)}. "
                "Consider document length as a factor. Colors may be relevant."
            )
        else:
            ctx_type = "contradictory"
            context = (
                "Group all technical documents together regardless of sub-topic. "
                "Also, make sure each specific technology gets its own cluster."
            )

        instances.append(ClusteringInstanceV2(
            instance_id=f"cluster_{split}_{i:04d}",
            documents=sampled,
            n_clusters=n_clusters,
            context=context,
            context_type=ctx_type,
            true_labels=true_labels,
            seed=seed,
            dataset_setting=dataset_setting,
            split=split,
        ))

    return instances


# ─── Feature Extraction ───

def extract_features(documents: list[Document],
                     max_features: int = TFIDF_MAX_FEATURES
                     ) -> tuple[np.ndarray, TfidfVectorizer]:
    """Extract TF-IDF features from documents.

    Returns (feature_matrix, fitted_vectorizer).
    """
    texts = [doc.text for doc in documents]
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    features = vectorizer.fit_transform(texts).toarray()
    return features, vectorizer


def get_vocab(vectorizer: TfidfVectorizer) -> dict[int, str]:
    """Get feature index → word mapping."""
    return {i: word for word, i in vectorizer.vocabulary_.items()}


# ─── Clustering ───

def run_kmeans(features: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(features)


# ─── Evaluation ───

def evaluate_clustering(pred_labels: np.ndarray,
                        true_labels: list[int],
                        features: np.ndarray) -> dict:
    """Compute all clustering quality metrics."""
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    homog = homogeneity_score(true_labels, pred_labels)
    n_pred_clusters = len(set(pred_labels))
    sil = silhouette_score(features, pred_labels) if n_pred_clusters > 1 else 0.0

    return {
        "adjusted_rand_index": round(ari, 4),
        "normalized_mutual_info": round(nmi, 4),
        "homogeneity": round(homog, 4),
        "silhouette_score": round(sil, 4),
        "n_clusters_found": n_pred_clusters,
    }


# ─── Validator V2 ───

class ClusteringValidatorV2(BaseValidator):
    """Configurable clustering validator with sweepable bounds."""

    def __init__(self, strength: str = "full",
                 config: Optional[ValidatorConfig] = None):
        super().__init__(name="clustering_validator_v2", strength=strength)
        self.config = config or ValidatorConfig()

    def _check_schema(self, proposal: dict) -> ValidationResult | None:
        if not isinstance(proposal, dict):
            return ValidationResult(
                decision="rejected", original_proposal=proposal,
                validated_proposal=None, reasons=["proposal is not a dict"])
        if "feature_boosts" not in proposal:
            return ValidationResult(
                decision="rejected", original_proposal=proposal,
                validated_proposal=None, reasons=["missing 'feature_boosts' key"])
        if not isinstance(proposal["feature_boosts"], list):
            return ValidationResult(
                decision="rejected", original_proposal=proposal,
                validated_proposal=None, reasons=["'feature_boosts' is not a list"])
        return None

    def _full_validate(self, proposal: dict, context: dict) -> ValidationResult:
        n_articles = context.get("n_articles", 0)
        n_features = context.get("n_features", 0)
        max_boost = self.config.max_feature_boost
        max_mods = self.config.max_features_modified

        valid_boosts = []
        reasons = []
        clamped = []

        for boost in proposal["feature_boosts"]:
            if not isinstance(boost, dict):
                reasons.append(f"skipped non-dict boost: {boost}")
                continue
            article_idx = boost.get("article_index")
            feature_idx = boost.get("feature_index")
            value = boost.get("boost_value", 0)

            if article_idx is None or feature_idx is None:
                reasons.append("skipped boost with missing index")
                continue
            if not isinstance(article_idx, int) or not isinstance(feature_idx, int):
                reasons.append(f"skipped non-integer indices: ({article_idx}, {feature_idx})")
                continue
            if article_idx < 0 or article_idx >= n_articles:
                reasons.append(f"rejected out-of-range article_index: {article_idx}")
                continue
            if feature_idx < 0 or feature_idx >= n_features:
                reasons.append(f"rejected out-of-range feature_index: {feature_idx}")
                continue
            if not isinstance(value, (int, float)):
                reasons.append(f"rejected non-numeric boost for ({article_idx}, {feature_idx})")
                continue

            original_value = value
            value = max(-max_boost, min(max_boost, value))
            if value != original_value:
                clamped.append(f"({article_idx},{feature_idx}): {original_value:.3f} -> {value:.3f}")

            valid_boosts.append({
                "article_index": article_idx,
                "feature_index": feature_idx,
                "boost_value": round(value, 4),
            })

        if len(valid_boosts) > max_mods:
            reasons.append(f"truncated from {len(valid_boosts)} to {max_mods} boosts")
            valid_boosts = valid_boosts[:max_mods]
            clamped.append(f"truncated to {max_mods}")

        if not valid_boosts:
            return ValidationResult(
                decision="rejected", original_proposal=proposal,
                validated_proposal=None,
                reasons=reasons + ["no valid boosts after validation"])

        decision = "clamped" if clamped else "accepted"
        return ValidationResult(
            decision=decision, original_proposal=proposal,
            validated_proposal={"feature_boosts": valid_boosts},
            reasons=reasons, clamped_fields=clamped)


# ─── Apply Boosts ───

def apply_boosts(features: np.ndarray,
                 validated_proposal: dict) -> np.ndarray:
    """Apply validated feature boosts to the feature matrix."""
    boosted = features.copy()
    for b in validated_proposal.get("feature_boosts", []):
        ai = b.get("article_index")
        fi = b.get("feature_index")
        bv = b.get("boost_value", 0)
        if (ai is not None and fi is not None and
            isinstance(ai, int) and isinstance(fi, int) and
            0 <= ai < features.shape[0] and 0 <= fi < features.shape[1] and
            isinstance(bv, (int, float))):
            boosted[ai, fi] += bv
    return boosted


# ─── Validate Proposal Offline ───

def validate_proposal_offline(raw_proposal: dict,
                              n_articles: int,
                              n_features: int,
                              config: ValidatorConfig) -> ValidationResult:
    """Apply a validator config to a raw proposal without making an API call.

    This is used for the validator sweep: run LLM once, apply many configs.
    """
    validator = ClusteringValidatorV2(strength="full", config=config)
    ctx = {"n_articles": n_articles, "n_features": n_features}
    return validator.validate(raw_proposal, ctx)


# ─── Tool Schema ───

CLUSTERING_TOOL_V2 = {
    "name": "propose_feature_boosts",
    "description": (
        "Propose feature boosts to improve document clustering quality. "
        "Adjust TF-IDF feature values for specific documents to encourage "
        "better grouping aligned with the clustering objective. "
        "Positive boosts increase a feature's weight for a document; "
        "negative boosts decrease it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "feature_boosts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "article_index": {
                            "type": "integer",
                            "description": "0-based document index"
                        },
                        "feature_index": {
                            "type": "integer",
                            "description": "0-based TF-IDF feature index (see vocabulary)"
                        },
                        "boost_value": {
                            "type": "number",
                            "description": "Value to add to the feature weight"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Brief reason for this boost"
                        },
                    },
                    "required": ["article_index", "feature_index", "boost_value"],
                },
                "description": "List of feature boosts to improve clustering",
            },
        },
        "required": ["feature_boosts"],
    },
}


# ─── Prompts ───

SYSTEM_PROMPT_V2 = (
    "You are a clustering optimization assistant. Given documents and a "
    "clustering objective, you propose adjustments to TF-IDF feature weights "
    "to improve cluster quality. You are shown the TF-IDF vocabulary so you "
    "can make informed decisions about which features to boost or suppress "
    "for which documents. Use the propose_feature_boosts tool."
)


def make_user_prompt_v2(instance: ClusteringInstanceV2,
                        n_features: int,
                        vocab: dict[int, str]) -> str:
    """Build user prompt including documents and TF-IDF vocabulary."""
    # Format vocabulary (top features)
    vocab_lines = []
    for idx in sorted(vocab.keys())[:n_features]:
        vocab_lines.append(f"  {idx}: {vocab[idx]}")
    vocab_str = "\n".join(vocab_lines)

    # Format documents (truncated)
    doc_lines = []
    for i, doc in enumerate(instance.documents):
        words = doc.text.split()[:MAX_DOC_WORDS]
        truncated = " ".join(words)
        if len(words) == MAX_DOC_WORDS:
            truncated += "..."
        doc_lines.append(f"  [{i}] {truncated}")
    docs_str = "\n".join(doc_lines)

    return (
        f"Clustering Objective: {instance.context}\n\n"
        f"Number of target clusters: {instance.n_clusters}\n"
        f"Number of documents: {len(instance.documents)}\n"
        f"Document indices: 0 to {len(instance.documents)-1}\n"
        f"Feature indices: 0 to {n_features-1}\n\n"
        f"TF-IDF Vocabulary (feature_index: word):\n{vocab_str}\n\n"
        f"Documents:\n{docs_str}\n\n"
        f"Propose feature boosts to improve clustering quality based on "
        f"the objective above."
    )


# ─── Baselines ───

def run_baseline_algorithm_only(instance: ClusteringInstanceV2) -> dict:
    """Pure KMeans on raw TF-IDF features."""
    features, _ = extract_features(instance.documents)
    labels = run_kmeans(features, instance.n_clusters, instance.seed)
    return evaluate_clustering(labels, instance.true_labels, features)


def run_baseline_rules(instance: ClusteringInstanceV2,
                       dataset_setting: str) -> dict:
    """Keyword-based feature reweighting from context."""
    features, vectorizer = extract_features(instance.documents)
    vocab = get_vocab(vectorizer)
    ctx_lower = instance.context.lower()

    # Extract keywords from context
    ctx_words = set(ctx_lower.split())
    boosted = features.copy()

    # Boost features whose TF-IDF word appears in the context
    for fi, word in vocab.items():
        if word.lower() in ctx_words:
            boosted[:, fi] *= 1.1  # 10% boost for context-mentioned terms

    labels = run_kmeans(boosted, instance.n_clusters, instance.seed)
    return evaluate_clustering(labels, instance.true_labels, boosted)


def run_baseline_random_perturbation(instance: ClusteringInstanceV2,
                                     n_boosts: int = 30,
                                     boost_magnitude: float = 0.5) -> dict:
    """Random feature perturbations matching LLM proposal statistics."""
    features, _ = extract_features(instance.documents)
    rng = np.random.RandomState(instance.seed)

    boosted = features.copy()
    n_docs, n_feats = features.shape
    for _ in range(n_boosts):
        ai = rng.randint(0, n_docs)
        fi = rng.randint(0, n_feats)
        bv = rng.uniform(-boost_magnitude, boost_magnitude)
        boosted[ai, fi] += bv

    labels = run_kmeans(boosted, instance.n_clusters, instance.seed)
    return evaluate_clustering(labels, instance.true_labels, boosted)


# ─── LLM Integration ───

def run_algogen(instance: ClusteringInstanceV2,
                proposal_client,
                validator_config: ValidatorConfig,
                validator_strength: str = "full") -> tuple[dict, dict | None]:
    """Run LLM proposal → validate → cluster → evaluate.

    Returns (metrics_dict, raw_proposal_or_None).
    The raw_proposal is returned for offline validator sweep.
    """
    features, vectorizer = extract_features(instance.documents)
    vocab = get_vocab(vectorizer)
    n_features = features.shape[1]

    user_prompt = make_user_prompt_v2(instance, n_features, vocab)

    raw_proposal, log = proposal_client.propose(
        benchmark=f"clustering_{instance.dataset_setting}",
        instance_id=instance.instance_id,
        system_prompt=SYSTEM_PROMPT_V2,
        user_prompt=user_prompt,
        tool_definition=CLUSTERING_TOOL_V2,
        tool_name="propose_feature_boosts",
        prompt_version="v2",
        schema_version="v2",
    )

    # Validate
    validator = ClusteringValidatorV2(strength=validator_strength,
                                      config=validator_config)
    ctx = {"n_articles": len(instance.documents), "n_features": n_features}
    vresult = validator.validate(raw_proposal, ctx)
    log.validator_decision = vresult.decision

    # Apply or fallback
    if vresult.decision in ("accepted", "clamped") and vresult.validated_proposal:
        boosted = apply_boosts(features, vresult.validated_proposal)
        labels = run_kmeans(boosted, instance.n_clusters, instance.seed)
        metrics = evaluate_clustering(labels, instance.true_labels, boosted)
    else:
        labels = run_kmeans(features, instance.n_clusters, instance.seed)
        metrics = evaluate_clustering(labels, instance.true_labels, features)

    metrics["validator_decision"] = vresult.decision
    metrics["proposal_accepted"] = vresult.decision in ("accepted", "clamped")
    metrics["n_boosts_proposed"] = (
        len(raw_proposal.get("feature_boosts", [])) if raw_proposal else 0
    )
    metrics["n_boosts_validated"] = (
        len(vresult.validated_proposal["feature_boosts"])
        if vresult.validated_proposal else 0
    )

    return metrics, raw_proposal


def apply_proposal_with_config(instance: ClusteringInstanceV2,
                               raw_proposal: dict | None,
                               config: ValidatorConfig) -> dict:
    """Apply a raw proposal with a specific validator config (offline sweep).

    No API call — reuses a previously collected raw proposal.
    """
    features, _ = extract_features(instance.documents)
    n_features = features.shape[1]

    if raw_proposal is None:
        # No proposal available (API error) — fallback to baseline
        labels = run_kmeans(features, instance.n_clusters, instance.seed)
        metrics = evaluate_clustering(labels, instance.true_labels, features)
        metrics["validator_decision"] = "error_fallback"
        metrics["proposal_accepted"] = False
        metrics["n_boosts_proposed"] = 0
        metrics["n_boosts_validated"] = 0
        return metrics

    vresult = validate_proposal_offline(
        raw_proposal, len(instance.documents), n_features, config)

    if vresult.decision in ("accepted", "clamped") and vresult.validated_proposal:
        boosted = apply_boosts(features, vresult.validated_proposal)
        labels = run_kmeans(boosted, instance.n_clusters, instance.seed)
        metrics = evaluate_clustering(labels, instance.true_labels, boosted)
    else:
        labels = run_kmeans(features, instance.n_clusters, instance.seed)
        metrics = evaluate_clustering(labels, instance.true_labels, features)

    metrics["validator_decision"] = vresult.decision
    metrics["proposal_accepted"] = vresult.decision in ("accepted", "clamped")
    metrics["n_boosts_proposed"] = len(raw_proposal.get("feature_boosts", []))
    metrics["n_boosts_validated"] = (
        len(vresult.validated_proposal["feature_boosts"])
        if vresult.validated_proposal else 0
    )
    return metrics
