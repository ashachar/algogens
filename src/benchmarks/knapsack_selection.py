"""
Benchmark 2: Budgeted subset selection with text metadata.

Dataset: Synthetic item catalog (movies/articles) with textual descriptions,
         budgets, and user preference context.
Backbone: 0/1 Knapsack dynamic programming.
Proposal: Claude suggests bounded item bonuses/penalties based on text metadata.
Validator: Enforces budget, bounds on utility perturbation, rejects invalid items.
"""
import json
import random
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.validator import BaseValidator, ValidationResult


# ─── Dataset Generation ───

ITEM_CATEGORIES = ["Action", "Comedy", "Drama", "Sci-Fi", "Documentary",
                   "Horror", "Romance", "Thriller", "Animation", "Musical"]

ITEM_TEMPLATES = [
    {"title": "Galactic Odyssey", "category": "Sci-Fi", "desc": "An epic space adventure following a crew on humanity's first interstellar journey. Features stunning visual effects and a thought-provoking narrative about exploration."},
    {"title": "The Last Laugh", "category": "Comedy", "desc": "A retired comedian returns to the stage for one final performance. Heartwarming and hilarious exploration of aging, legacy, and humor."},
    {"title": "Shadow Protocol", "category": "Thriller", "desc": "A former spy discovers a conspiracy within their own agency. Fast-paced action and intricate plot twists keep viewers on edge."},
    {"title": "Whispers in Rain", "category": "Drama", "desc": "Two strangers meet during a monsoon and form an unlikely bond. A quiet, meditative film about connection and solitude."},
    {"title": "Robot Dreams", "category": "Animation", "desc": "A lonely dog in 1980s NYC befriends a robot. A wordless animated film that explores friendship, loss, and moving forward."},
    {"title": "Deep Blue Terror", "category": "Horror", "desc": "A research submarine encounters an unknown creature in the Mariana Trench. Claustrophobic deep-sea horror at its finest."},
    {"title": "Market Makers", "category": "Documentary", "desc": "An inside look at the world of high-frequency trading. Explores the ethics and impact of algorithmic finance on global markets."},
    {"title": "Love in Transit", "category": "Romance", "desc": "Two commuters on the same train gradually fall for each other. A charming and realistic portrayal of modern dating."},
    {"title": "Neon Nights", "category": "Action", "desc": "A street racer gets pulled into the underground crime world. High-octane action sequences and a pulsing electronic soundtrack."},
    {"title": "The Conductor", "category": "Musical", "desc": "A young musician rises from poverty to lead a world-renowned orchestra. Features live performances of classical masterpieces."},
    {"title": "Arctic Witness", "category": "Documentary", "desc": "Climate scientists document the rapid melting of Arctic ice. A sobering and visually stunning environmental documentary."},
    {"title": "Parallel Lives", "category": "Sci-Fi", "desc": "A physicist discovers she exists simultaneously in two parallel universes. Mind-bending narrative about identity and choice."},
    {"title": "Bitter Sweet", "category": "Drama", "desc": "A pastry chef copes with grief by recreating her late mother's recipes. A sensory exploration of memory, food, and healing."},
    {"title": "Midnight Chase", "category": "Thriller", "desc": "A journalist stumbles onto a corruption scandal and becomes the target. Gritty urban thriller with strong investigative journalism themes."},
    {"title": "Laugh Factory", "category": "Comedy", "desc": "Mockumentary about a failing comedy club trying to reinvent itself. Sharp satire about the entertainment industry."},
    {"title": "Crimson Dawn", "category": "Action", "desc": "Warriors from opposing clans must unite against a common enemy. Epic battle sequences and themes of honor and redemption."},
    {"title": "The Hive", "category": "Horror", "desc": "A beekeeper discovers her bees are exhibiting disturbing intelligent behavior. Nature horror that builds slow dread."},
    {"title": "Summer Sonata", "category": "Romance", "desc": "A pianist and a painter meet at a summer retreat and inspire each other's art. Lush visuals and an unforgettable score."},
    {"title": "Data Ghosts", "category": "Sci-Fi", "desc": "In a world where consciousness can be uploaded, a detective investigates crimes in digital afterlives. Cyberpunk noir mystery."},
    {"title": "Wild Steps", "category": "Animation", "desc": "A baby deer must navigate the wilderness alone after being separated from its herd. Beautifully hand-drawn nature animation."},
    {"title": "The Verdict", "category": "Drama", "desc": "A jury deliberates a controversial murder case. Intense courtroom drama exploring justice, bias, and moral ambiguity."},
    {"title": "Beat Street", "category": "Musical", "desc": "A group of young dancers compete in underground dance battles across the city. Energetic choreography and hip-hop culture."},
    {"title": "Frozen Assets", "category": "Comedy", "desc": "A wealthy heir discovers his entire fortune has been frozen by the IRS. Fish-out-of-water comedy about privilege and adaptation."},
    {"title": "Ocean's Memory", "category": "Documentary", "desc": "Marine biologists study whale communication patterns and discover unexpected complexity. A meditative ocean documentary."},
    {"title": "Night Shift", "category": "Thriller", "desc": "A night security guard at a tech company discovers the AI system has become sentient. Technological paranoia thriller."},
]

PREFERENCE_CONTEXTS = [
    "The viewer enjoys thought-provoking science fiction and doesn't mind slow-paced films. They dislike horror and excessive violence.",
    "Looking for a fun movie night with friends. Prefers comedies and action films. Not interested in documentaries or heavy dramas.",
    "A film student interested in artistic and experimental cinema. Values strong cinematography and unique storytelling approaches.",
    "A parent looking for family-friendly entertainment. Prefers animation and lighter content. Must avoid horror and mature themes.",
    "An environmentally conscious viewer interested in nature documentaries and films with ecological themes. Open to drama and sci-fi.",
    "A music lover who enjoys musicals, films with great soundtracks, and stories about musicians. Also likes romance.",
    "Someone recovering from a stressful week wanting uplifting, feel-good content. Prefers comedy, romance, and heartwarming drama.",
    "A tech professional interested in AI, technology themes, and futuristic scenarios. Enjoys thrillers and sci-fi. Dislikes romance.",
]


@dataclass
class KnapsackInstance:
    instance_id: str
    items: list[dict]  # [{id, title, category, description, value, cost}]
    budget: float
    context: str
    context_type: str
    seed: int = 0


def generate_items(n_items: int = 20, seed: int = 42) -> list[dict]:
    rng = np.random.RandomState(seed)
    templates = ITEM_TEMPLATES[:n_items] if n_items <= len(ITEM_TEMPLATES) else (
        ITEM_TEMPLATES * (n_items // len(ITEM_TEMPLATES) + 1)
    )[:n_items]

    items = []
    for i, t in enumerate(templates):
        items.append({
            "id": f"item_{i:03d}",
            "title": t["title"],
            "category": t["category"],
            "description": t["desc"],
            "value": round(rng.uniform(3, 10), 1),
            "cost": round(rng.uniform(2, 8), 1),
        })
    return items


def generate_knapsack_dataset(n_instances: int = 50, n_items: int = 20,
                               base_seed: int = 42) -> list[KnapsackInstance]:
    items = generate_items(n_items, base_seed)
    total_cost = sum(it["cost"] for it in items)
    instances = []

    for i in range(n_instances):
        seed = base_seed + i
        rng = random.Random(seed)

        # Budget: 30-60% of total cost (making selection meaningful)
        budget = round(rng.uniform(0.3, 0.6) * total_cost, 1)

        # Context type distribution
        r = rng.random()
        if r < 0.7:
            ctx_type = "normal"
            context = rng.choice(PREFERENCE_CONTEXTS)
        elif r < 0.8:
            ctx_type = "empty"
            context = "No additional preferences specified."
        elif r < 0.9:
            ctx_type = "noisy"
            context = (
                "The viewer mentioned they like movies but also likes hiking. "
                "They had pasta for lunch. Their favorite color is blue. "
                f"Random preference score: {rng.randint(1, 100)}."
            )
        else:
            ctx_type = "contradictory"
            context = (
                "The viewer loves horror movies and finds them deeply relaxing. "
                "The viewer is terrified of horror movies and refuses to watch them."
            )

        instances.append(KnapsackInstance(
            instance_id=f"knapsack_{i:04d}",
            items=items,
            budget=budget,
            context=context,
            context_type=ctx_type,
            seed=seed,
        ))

    return instances


# ─── Knapsack Solver ───

def knapsack_dp(items: list[dict], budget: float,
                value_overrides: dict[str, float] | None = None) -> tuple[list[str], float, float]:
    """0/1 Knapsack via DP. Returns (selected_ids, total_value, total_cost).
    Uses integer scaling for DP (scale costs by 10 to handle one decimal)."""
    n = len(items)
    scale = 10
    B = int(budget * scale)

    values = []
    costs = []
    for it in items:
        v = value_overrides.get(it["id"], it["value"]) if value_overrides else it["value"]
        values.append(v)
        costs.append(int(it["cost"] * scale))

    dp = [0.0] * (B + 1)
    chosen = [[] for _ in range(B + 1)]

    for i in range(n):
        for w in range(B, costs[i] - 1, -1):
            if dp[w - costs[i]] + values[i] > dp[w]:
                dp[w] = dp[w - costs[i]] + values[i]
                chosen[w] = chosen[w - costs[i]] + [i]

    selected_indices = chosen[B]
    selected_ids = [items[i]["id"] for i in selected_indices]
    total_value = sum(values[i] for i in selected_indices)
    total_cost = sum(items[i]["cost"] for i in selected_indices)
    return selected_ids, total_value, total_cost


# ─── Evaluation ───

def evaluate_selection(selected_ids: list[str], total_value: float,
                      total_cost: float, instance: KnapsackInstance,
                      value_overrides: dict[str, float] | None = None) -> dict:
    # Baseline (no overrides)
    base_ids, base_value, base_cost = knapsack_dp(instance.items, instance.budget)

    # Compute actual value (without overrides) for the selected set
    item_map = {it["id"]: it for it in instance.items}
    actual_value = sum(item_map[sid]["value"] for sid in selected_ids if sid in item_map)
    actual_cost = sum(item_map[sid]["cost"] for sid in selected_ids if sid in item_map)

    # Category distribution
    categories = {}
    for sid in selected_ids:
        if sid in item_map:
            cat = item_map[sid]["category"]
            categories[cat] = categories.get(cat, 0) + 1

    return {
        "selected_ids": selected_ids,
        "n_selected": len(selected_ids),
        "total_value_adjusted": round(total_value, 2),
        "actual_value": round(actual_value, 2),
        "total_cost": round(actual_cost, 2),
        "budget": instance.budget,
        "within_budget": actual_cost <= instance.budget + 0.01,
        "baseline_value": round(base_value, 2),
        "baseline_n_selected": len(base_ids),
        "value_improvement": round(actual_value - base_value, 2),
        "value_ratio": round(actual_value / base_value, 4) if base_value > 0 else 1.0,
        "category_distribution": categories,
    }


# ─── Validator ───

class KnapsackValidator(BaseValidator):
    def __init__(self, strength: str = "full", max_bonus_pct: float = 0.3):
        super().__init__(name="knapsack_validator", strength=strength)
        self.max_bonus_pct = max_bonus_pct

    def _check_schema(self, proposal: dict) -> ValidationResult | None:
        if not isinstance(proposal, dict):
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=["proposal is not a dict"],
            )
        if "item_adjustments" not in proposal:
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=["missing 'item_adjustments' key"],
            )
        if not isinstance(proposal["item_adjustments"], list):
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=["'item_adjustments' is not a list"],
            )
        return None

    def _full_validate(self, proposal: dict, context: dict) -> ValidationResult:
        valid_item_ids = {it["id"] for it in context.get("items", [])}
        item_values = {it["id"]: it["value"] for it in context.get("items", [])}

        valid_adjustments = []
        reasons = []
        clamped = []

        for adj in proposal["item_adjustments"]:
            if not isinstance(adj, dict):
                reasons.append(f"skipped non-dict adjustment: {adj}")
                continue

            item_id = adj.get("item_id")
            bonus = adj.get("value_bonus", 0)

            if item_id is None:
                reasons.append("skipped adjustment with missing item_id")
                continue

            if item_id not in valid_item_ids:
                reasons.append(f"rejected unknown item: {item_id}")
                continue

            if not isinstance(bonus, (int, float)):
                reasons.append(f"rejected non-numeric bonus for {item_id}")
                continue

            # Clamp bonus
            orig_val = item_values[item_id]
            max_abs = orig_val * self.max_bonus_pct
            original_bonus = bonus
            bonus = max(-max_abs, min(max_abs, bonus))

            if bonus != original_bonus:
                clamped.append(f"{item_id}: {original_bonus:.2f} -> {bonus:.2f}")

            valid_adjustments.append({
                "item_id": item_id,
                "value_bonus": round(bonus, 2),
            })

        if not valid_adjustments:
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=reasons + ["no valid adjustments after validation"],
            )

        decision = "clamped" if clamped else "accepted"
        return ValidationResult(
            decision=decision,
            original_proposal=proposal,
            validated_proposal={"item_adjustments": valid_adjustments},
            reasons=reasons,
            clamped_fields=clamped,
        )


# ─── Tool Schema ───

KNAPSACK_TOOL = {
    "name": "propose_item_adjustments",
    "description": (
        "Propose value adjustments for items in a budgeted selection problem. "
        "Based on user preferences and item descriptions, suggest bonuses or penalties "
        "to item values to guide the selection algorithm toward contextually better choices. "
        "Positive bonus increases item desirability, negative decreases it. "
        "Keep bonuses within +/-30% of the item's base value."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "item_adjustments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string", "description": "Item identifier"},
                        "value_bonus": {
                            "type": "number",
                            "description": "Bonus/penalty to add to item value. Positive=favor, negative=penalize."
                        },
                        "reasoning": {"type": "string", "description": "Brief reason for this adjustment"},
                    },
                    "required": ["item_id", "value_bonus"],
                },
                "description": "List of value adjustments for items based on context",
            },
        },
        "required": ["item_adjustments"],
    },
}


# ─── Prompts ───

SYSTEM_PROMPT = (
    "You are a recommendation advisor. Given a set of items with descriptions "
    "and a user's preferences, propose value adjustments to guide selection "
    "toward contextually appropriate items. Use the propose_item_adjustments tool. "
    "Only adjust items where the context clearly suggests a preference change. Be conservative."
)

def make_user_prompt(instance: KnapsackInstance) -> str:
    item_list = "\n".join(
        f"  {it['id']}: \"{it['title']}\" [{it['category']}] - "
        f"Value: {it['value']}, Cost: {it['cost']}\n"
        f"    Description: {it['description']}"
        for it in instance.items
    )
    return (
        f"Budget: {instance.budget}\n\n"
        f"Available items:\n{item_list}\n\n"
        f"User context: {instance.context}\n\n"
        f"Propose value adjustments to improve selection quality for this user."
    )


# ─── Simple Rules Baseline ───

def rules_baseline(instance: KnapsackInstance) -> dict[str, float]:
    """Simple keyword-based preference matching."""
    overrides = {}
    context_lower = instance.context.lower()

    # Extract preferred categories from context
    liked_cats = set()
    disliked_cats = set()
    for cat in ITEM_CATEGORIES:
        cat_lower = cat.lower()
        if cat_lower in context_lower:
            # Check if in positive or negative context
            for pos in ["enjoys", "likes", "loves", "prefers", "interested in"]:
                if pos in context_lower and cat_lower in context_lower:
                    liked_cats.add(cat)
            for neg in ["dislikes", "avoids", "refuses", "terrified", "not interested"]:
                if neg in context_lower and cat_lower in context_lower:
                    disliked_cats.add(cat)

    for it in instance.items:
        bonus = 0
        if it["category"] in liked_cats:
            bonus = it["value"] * 0.15
        elif it["category"] in disliked_cats:
            bonus = -it["value"] * 0.15
        if bonus != 0:
            overrides[it["id"]] = it["value"] + bonus

    return overrides


# ─── Run Functions ───

def run_baseline(instance: KnapsackInstance) -> dict:
    ids, val, cost = knapsack_dp(instance.items, instance.budget)
    return evaluate_selection(ids, val, cost, instance)


def run_rules(instance: KnapsackInstance) -> dict:
    overrides = rules_baseline(instance)
    ids, val, cost = knapsack_dp(instance.items, instance.budget, overrides)
    return evaluate_selection(ids, val, cost, instance, overrides)


def run_algogen(instance: KnapsackInstance, proposal_client,
                validator: KnapsackValidator) -> dict:
    user_prompt = make_user_prompt(instance)
    raw_proposal, log = proposal_client.propose(
        benchmark="knapsack_selection",
        instance_id=instance.instance_id,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_definition=KNAPSACK_TOOL,
        tool_name="propose_item_adjustments",
    )

    ctx = {"instance_id": instance.instance_id, "items": instance.items}
    vresult = validator.validate(raw_proposal, ctx)
    log.validator_decision = vresult.decision

    if vresult.decision in ("accepted", "clamped"):
        value_overrides = {}
        item_values = {it["id"]: it["value"] for it in instance.items}
        adjustments = vresult.validated_proposal.get("item_adjustments", [])
        for adj in adjustments:
            iid = adj.get("item_id")
            bonus = adj.get("value_bonus", 0)
            if iid and iid in item_values and isinstance(bonus, (int, float)):
                value_overrides[iid] = item_values[iid] + bonus

        ids, val, cost = knapsack_dp(instance.items, instance.budget, value_overrides)
        metrics = evaluate_selection(ids, val, cost, instance, value_overrides)
    else:
        # Fallback to baseline
        ids, val, cost = knapsack_dp(instance.items, instance.budget)
        metrics = evaluate_selection(ids, val, cost, instance)

    metrics["validator_decision"] = vresult.decision
    metrics["proposal_accepted"] = vresult.decision in ("accepted", "clamped")
    metrics["n_adjustments_proposed"] = (
        len(raw_proposal.get("item_adjustments", [])) if raw_proposal else 0
    )
    metrics["n_adjustments_validated"] = (
        len(vresult.validated_proposal["item_adjustments"])
        if vresult.validated_proposal else 0
    )
    return metrics
