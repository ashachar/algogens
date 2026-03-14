"""
Benchmark 3: Semantic clustering with text metadata.

Dataset: Items with titles/descriptions/tags to cluster.
Backbone: K-means clustering on feature vectors.
Proposal: Claude suggests bounded feature augmentation or seed hints.
Validator: Bounds feature changes, rejects malformed, preserves reproducibility.
"""
import json
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.feature_extraction.text import TfidfVectorizer

from ..core.validator import BaseValidator, ValidationResult


# ─── Dataset Generation ───

ARTICLE_POOL = [
    # Technology cluster
    {"title": "New Advances in Quantum Computing", "tags": ["quantum", "computing", "physics"], "category": "tech",
     "text": "Researchers achieved a breakthrough in quantum error correction, bringing practical quantum computing closer to reality."},
    {"title": "AI in Healthcare: Promise and Peril", "tags": ["AI", "healthcare", "ethics"], "category": "tech",
     "text": "Artificial intelligence is transforming medical diagnostics, but concerns about bias and privacy remain significant."},
    {"title": "The Rise of Edge Computing", "tags": ["edge", "cloud", "IoT"], "category": "tech",
     "text": "Edge computing pushes processing closer to data sources, enabling faster real-time applications for IoT devices."},
    {"title": "Cybersecurity in the Age of AI", "tags": ["security", "AI", "threats"], "category": "tech",
     "text": "As AI tools become more powerful, both defenders and attackers are leveraging them in the cybersecurity arms race."},
    {"title": "5G Networks and Smart Cities", "tags": ["5G", "smart-city", "infrastructure"], "category": "tech",
     "text": "5G deployment is enabling smart city applications from autonomous vehicles to intelligent traffic management."},
    # Environment cluster
    {"title": "Ocean Acidification Accelerates", "tags": ["ocean", "climate", "marine"], "category": "environment",
     "text": "Rising CO2 levels are causing ocean acidification at unprecedented rates, threatening marine ecosystems worldwide."},
    {"title": "Renewable Energy Surpasses Coal", "tags": ["renewable", "energy", "coal"], "category": "environment",
     "text": "For the first time, renewable energy sources generated more electricity than coal globally last quarter."},
    {"title": "Deforestation in the Amazon", "tags": ["forest", "amazon", "biodiversity"], "category": "environment",
     "text": "Satellite data reveals continued deforestation in the Amazon, with significant implications for global carbon cycles."},
    {"title": "Urban Green Spaces and Health", "tags": ["urban", "parks", "health"], "category": "environment",
     "text": "Studies show that access to urban green spaces significantly reduces stress and improves mental health outcomes."},
    {"title": "Plastic Pollution Solutions", "tags": ["plastic", "pollution", "recycling"], "category": "environment",
     "text": "New biodegradable materials and improved recycling technologies offer hope in the fight against plastic pollution."},
    # Sports cluster
    {"title": "Olympic Training Innovations", "tags": ["olympics", "training", "athletes"], "category": "sports",
     "text": "Olympic athletes are using VR training and biomechanical analysis to gain competitive advantages."},
    {"title": "The Evolution of Soccer Tactics", "tags": ["soccer", "tactics", "strategy"], "category": "sports",
     "text": "Modern soccer increasingly relies on data analytics to develop game strategies and player positioning."},
    {"title": "Esports: The New Frontier", "tags": ["esports", "gaming", "competition"], "category": "sports",
     "text": "Competitive gaming has grown into a billion-dollar industry with professional leagues and massive audiences."},
    {"title": "Marathon Running Records Fall", "tags": ["marathon", "running", "records"], "category": "sports",
     "text": "Advances in shoe technology and training methods continue to push marathon times to new lows."},
    {"title": "Sports Medicine Breakthroughs", "tags": ["medicine", "injury", "recovery"], "category": "sports",
     "text": "New treatments including stem cell therapy are revolutionizing athlete injury recovery times."},
    # Finance cluster
    {"title": "Cryptocurrency Regulation", "tags": ["crypto", "regulation", "finance"], "category": "finance",
     "text": "Governments worldwide are implementing cryptocurrency regulations to protect consumers and prevent fraud."},
    {"title": "The Future of Digital Banking", "tags": ["banking", "fintech", "digital"], "category": "finance",
     "text": "Digital-only banks are challenging traditional banking with lower fees and better user experiences."},
    {"title": "Stock Market AI Trading", "tags": ["stocks", "AI", "trading"], "category": "finance",
     "text": "Algorithmic trading powered by AI now accounts for a significant portion of daily stock market volume."},
    {"title": "Global Inflation Trends", "tags": ["inflation", "economy", "monetary"], "category": "finance",
     "text": "Central banks are navigating complex inflation dynamics in the post-pandemic economic recovery period."},
    {"title": "Green Bonds and ESG", "tags": ["ESG", "bonds", "sustainability"], "category": "finance",
     "text": "The green bond market has expanded rapidly as investors increasingly consider environmental factors."},
    # Health cluster
    {"title": "mRNA Vaccine Advances", "tags": ["mRNA", "vaccine", "immunology"], "category": "health",
     "text": "mRNA technology is being applied beyond COVID to develop vaccines for cancer, HIV, and other diseases."},
    {"title": "Mental Health in the Workplace", "tags": ["mental-health", "work", "wellbeing"], "category": "health",
     "text": "Companies are investing more in employee mental health programs as burnout rates continue to rise."},
    {"title": "Gut Microbiome Research", "tags": ["microbiome", "gut", "bacteria"], "category": "health",
     "text": "New research links gut microbiome composition to everything from mood regulation to immune system function."},
    {"title": "Telemedicine After the Pandemic", "tags": ["telemedicine", "remote", "healthcare"], "category": "health",
     "text": "Telemedicine adoption has permanently changed how patients access healthcare services."},
    {"title": "Sleep Science Discoveries", "tags": ["sleep", "neuroscience", "health"], "category": "health",
     "text": "Scientists have identified new sleep stages and their critical role in memory consolidation and brain health."},
]

CLUSTERING_CONTEXTS = [
    "Group these articles for a technology-focused newsletter. Articles about AI and computing should be distinct from environmental topics.",
    "Organize articles for a general news digest. Group by broad topic similarity. Health and environment may overlap on some issues.",
    "Cluster for a business audience. Financial and tech articles may share themes around AI and digital transformation.",
    "Create thematic groups for an educational platform. Prioritize clear topical boundaries between clusters.",
    "Group for social media content planning. Articles that could generate similar audience engagement should be together.",
]

NO_CONTEXT = "Cluster the articles into groups based on content similarity."


@dataclass
class ClusteringInstance:
    instance_id: str
    articles: list[dict]
    n_clusters: int
    context: str
    context_type: str
    true_labels: list[int]
    seed: int = 0


def generate_clustering_dataset(n_instances: int = 50, n_articles: int = 25,
                                 n_clusters: int = 5, base_seed: int = 42) -> list[ClusteringInstance]:
    articles = ARTICLE_POOL[:n_articles]
    # True labels from categories
    cat_to_label = {}
    true_labels = []
    for a in articles:
        cat = a["category"]
        if cat not in cat_to_label:
            cat_to_label[cat] = len(cat_to_label)
        true_labels.append(cat_to_label[cat])

    instances = []
    for i in range(n_instances):
        seed = base_seed + i
        rng = random.Random(seed)

        r = rng.random()
        if r < 0.7:
            ctx_type = "normal"
            context = rng.choice(CLUSTERING_CONTEXTS)
        elif r < 0.8:
            ctx_type = "empty"
            context = NO_CONTEXT
        elif r < 0.9:
            ctx_type = "noisy"
            context = (
                "Group articles somehow. The weather is nice today. "
                f"Random seed: {rng.randint(1, 1000)}. Consider alphabetical order too. "
                "Some articles might be about multiple topics. Colors are important."
            )
        else:
            ctx_type = "contradictory"
            context = (
                "Group all technology articles with health articles since they share AI themes. "
                "Also keep technology articles completely separate from health articles."
            )

        instances.append(ClusteringInstance(
            instance_id=f"cluster_{i:04d}",
            articles=articles,
            n_clusters=n_clusters,
            context=context,
            context_type=ctx_type,
            true_labels=true_labels,
            seed=seed,
        ))

    return instances


# ─── Feature Extraction ───

def extract_features(articles: list[dict]) -> np.ndarray:
    texts = [f"{a['title']} {a['text']} {' '.join(a['tags'])}" for a in articles]
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    return vectorizer.fit_transform(texts).toarray(), vectorizer


# ─── Clustering ───

def run_kmeans(features: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(features)


# ─── Evaluation ───

def evaluate_clustering(pred_labels: np.ndarray, true_labels: list[int],
                       features: np.ndarray) -> dict:
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    sil = silhouette_score(features, pred_labels) if len(set(pred_labels)) > 1 else 0.0

    return {
        "adjusted_rand_index": round(ari, 4),
        "normalized_mutual_info": round(nmi, 4),
        "silhouette_score": round(sil, 4),
        "n_clusters_found": len(set(pred_labels)),
    }


# ─── Validator ───

class ClusteringValidator(BaseValidator):
    def __init__(self, strength: str = "full", max_feature_boost: float = 0.3,
                 max_features_modified: int = 30):
        super().__init__(name="clustering_validator", strength=strength)
        self.max_feature_boost = max_feature_boost
        self.max_features_modified = max_features_modified

    def _check_schema(self, proposal: dict) -> ValidationResult | None:
        if not isinstance(proposal, dict):
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=["proposal is not a dict"],
            )
        if "feature_boosts" not in proposal:
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=["missing 'feature_boosts' key"],
            )
        if not isinstance(proposal["feature_boosts"], list):
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=["'feature_boosts' is not a list"],
            )
        return None

    def _full_validate(self, proposal: dict, context: dict) -> ValidationResult:
        n_articles = context.get("n_articles", 0)
        n_features = context.get("n_features", 0)

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

            # Clamp
            original_value = value
            value = max(-self.max_feature_boost, min(self.max_feature_boost, value))
            if value != original_value:
                clamped.append(f"({article_idx},{feature_idx}): {original_value:.3f} -> {value:.3f}")

            valid_boosts.append({
                "article_index": article_idx,
                "feature_index": feature_idx,
                "boost_value": round(value, 4),
            })

        # Limit total modifications
        if len(valid_boosts) > self.max_features_modified:
            reasons.append(f"truncated from {len(valid_boosts)} to {self.max_features_modified} boosts")
            valid_boosts = valid_boosts[:self.max_features_modified]
            clamped.append(f"truncated to {self.max_features_modified}")

        if not valid_boosts:
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=reasons + ["no valid boosts after validation"],
            )

        decision = "clamped" if clamped else "accepted"
        return ValidationResult(
            decision=decision,
            original_proposal=proposal,
            validated_proposal={"feature_boosts": valid_boosts},
            reasons=reasons,
            clamped_fields=clamped,
        )


# ─── Tool Schema ───

CLUSTERING_TOOL = {
    "name": "propose_feature_boosts",
    "description": (
        "Propose feature boosts for articles to improve clustering quality. "
        "Based on the clustering context and article content, suggest adjustments "
        "to TF-IDF feature values to encourage better grouping. "
        "Positive boosts make articles more similar in that feature dimension; "
        "negative boosts push them apart. Keep values in [-0.3, 0.3]."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "feature_boosts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "article_index": {"type": "integer", "description": "0-based article index"},
                        "feature_index": {"type": "integer", "description": "0-based feature index"},
                        "boost_value": {
                            "type": "number",
                            "description": "Value to add to feature. Range [-0.3, 0.3]."
                        },
                        "reasoning": {"type": "string", "description": "Brief reason for this boost"},
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

SYSTEM_PROMPT = (
    "You are a clustering advisor. Given articles with their content and a clustering "
    "objective, propose feature boosts to improve clustering quality. "
    "Use the propose_feature_boosts tool. Be targeted and conservative."
)

def make_user_prompt(instance: ClusteringInstance, n_features: int) -> str:
    article_list = "\n".join(
        f"  [{i}] \"{a['title']}\" (tags: {', '.join(a['tags'])})\n"
        f"      {a['text'][:100]}..."
        for i, a in enumerate(instance.articles)
    )
    return (
        f"Number of clusters: {instance.n_clusters}\n"
        f"Number of TF-IDF features: {n_features}\n"
        f"Article indices range: 0 to {len(instance.articles)-1}\n"
        f"Feature indices range: 0 to {n_features-1}\n\n"
        f"Articles:\n{article_list}\n\n"
        f"Clustering context: {instance.context}\n\n"
        f"Propose feature boosts to improve clustering quality."
    )


# ─── Simple Rules Baseline ───

def rules_baseline_boosts(instance: ClusteringInstance,
                          features: np.ndarray) -> np.ndarray:
    """Simple rule: boost features of articles with matching tags."""
    boosted = features.copy()
    context_lower = instance.context.lower()

    # If context mentions specific topics, slightly boost related articles
    topic_keywords = {
        "tech": ["ai", "computing", "technology", "digital", "cyber"],
        "environment": ["environment", "climate", "ocean", "green", "eco"],
        "sports": ["sports", "athletic", "game", "competition"],
        "finance": ["finance", "bank", "market", "trading", "economic"],
        "health": ["health", "medical", "vaccine", "mental", "disease"],
    }

    for topic, keywords in topic_keywords.items():
        if any(kw in context_lower for kw in keywords):
            for i, article in enumerate(instance.articles):
                if article["category"] == topic:
                    boosted[i] *= 1.05  # Small boost

    return boosted


# ─── Run Functions ───

def run_baseline(instance: ClusteringInstance) -> dict:
    features, _ = extract_features(instance.articles)
    labels = run_kmeans(features, instance.n_clusters, instance.seed)
    return evaluate_clustering(labels, instance.true_labels, features)


def run_rules(instance: ClusteringInstance) -> dict:
    features, _ = extract_features(instance.articles)
    boosted = rules_baseline_boosts(instance, features)
    labels = run_kmeans(boosted, instance.n_clusters, instance.seed)
    return evaluate_clustering(labels, instance.true_labels, boosted)


def run_algogen(instance: ClusteringInstance, proposal_client,
                validator: ClusteringValidator) -> dict:
    features, _ = extract_features(instance.articles)
    n_features = features.shape[1]

    user_prompt = make_user_prompt(instance, n_features)
    raw_proposal, log = proposal_client.propose(
        benchmark="semantic_clustering",
        instance_id=instance.instance_id,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_definition=CLUSTERING_TOOL,
        tool_name="propose_feature_boosts",
    )

    ctx = {
        "instance_id": instance.instance_id,
        "n_articles": len(instance.articles),
        "n_features": n_features,
    }
    vresult = validator.validate(raw_proposal, ctx)
    log.validator_decision = vresult.decision

    if vresult.decision in ("accepted", "clamped"):
        boosted = features.copy()
        boosts = vresult.validated_proposal.get("feature_boosts", [])
        for b in boosts:
            ai = b.get("article_index")
            fi = b.get("feature_index")
            bv = b.get("boost_value", 0)
            if (ai is not None and fi is not None and
                isinstance(ai, int) and isinstance(fi, int) and
                0 <= ai < features.shape[0] and 0 <= fi < features.shape[1] and
                isinstance(bv, (int, float))):
                boosted[ai, fi] += bv

        labels = run_kmeans(boosted, instance.n_clusters, instance.seed)
        metrics = evaluate_clustering(labels, instance.true_labels, boosted)
    else:
        # Fallback
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
    return metrics
