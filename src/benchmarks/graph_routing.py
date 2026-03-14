"""
Benchmark 1: Context-conditioned graph routing.

Dataset: Synthetic city transport graph with textual disruption/preference context.
Backbone: Dijkstra's shortest path.
Proposal: Claude suggests bounded edge-weight adjustments based on textual context.
Validator: Clamps adjustments, rejects invalid edges, preserves graph consistency.
"""
import heapq
import json
import random
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.validator import BaseValidator, ValidationResult


# ─── Dataset Generation ───

CITY_NAMES = [
    "Central Station", "Airport", "University", "Hospital", "Shopping Mall",
    "Tech Park", "Old Town", "Harbor", "Stadium", "Museum",
    "City Hall", "Library", "Market Square", "Suburb North", "Suburb South",
    "Industrial Zone", "Beach", "Park", "Convention Center", "Residential East",
]

DISRUPTION_TEMPLATES = [
    "Major construction on the route between {a} and {b}. Expected delays of {d} minutes.",
    "Traffic accident reported near {a}. Avoid routes through this area if possible.",
    "Public event at {b} causing road closures. Expect heavy congestion nearby.",
    "Road maintenance between {a} and {b} reducing capacity. Moderate delays expected.",
    "Flooding reported near {a}. Some routes may be impassable.",
    "VIP motorcade scheduled near {b}. Temporary road closures in effect.",
    "Gas leak reported between {a} and {b}. Emergency crews on scene.",
    "Weather alert: icy conditions on routes near {a}. Drive with caution.",
]

PREFERENCE_TEMPLATES = [
    "The traveler prefers scenic routes and is willing to add up to {d} minutes for a nicer path.",
    "The traveler wants to avoid {a} due to personal preference.",
    "The traveler has a meeting near {b} and prioritizes arriving quickly.",
    "The traveler prefers routes with fewer transfers, even if slightly longer.",
    "The traveler is mobility-impaired and needs accessible routes. Avoid {a} if accessibility is limited.",
]

NO_CONTEXT = "No additional context available."

NOISY_CONTEXT_TEMPLATE = (
    "Unrelated news: The weather in Paris is sunny. Stock markets are up. "
    "A new restaurant opened downtown. The local team won their game. "
    "Construction has been reported somewhere. Traffic may or may not be affected. "
    "Random data: {r1}, {r2}, {r3}."
)

CONTRADICTORY_TEMPLATE = (
    "Route between {a} and {b} is completely blocked. "
    "Also, route between {a} and {b} is perfectly clear with no delays."
)


@dataclass
class GraphInstance:
    instance_id: str
    nodes: list[str]
    edges: list[tuple[str, str, float]]  # (src, dst, weight)
    source: str
    target: str
    context: str
    context_type: str  # "normal", "empty", "noisy", "contradictory"
    optimal_path_baseline: list[str] | None = None
    optimal_cost_baseline: float | None = None
    affected_edges: list[tuple[str, str]] = field(default_factory=list)
    seed: int = 0


def generate_graph(n_nodes: int = 15, edge_density: float = 0.3,
                   seed: int = 42) -> tuple[list[str], list[tuple[str, str, float]]]:
    rng = np.random.RandomState(seed)
    nodes = CITY_NAMES[:n_nodes]
    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_density:
                w = round(rng.uniform(5, 50), 1)
                edges.append((nodes[i], nodes[j], w))
                edges.append((nodes[j], nodes[i], w))  # bidirectional
    # Ensure connectivity with a spanning path
    perm = rng.permutation(n_nodes)
    for k in range(n_nodes - 1):
        a, b = nodes[perm[k]], nodes[perm[k + 1]]
        if not any(e[0] == a and e[1] == b for e in edges):
            w = round(rng.uniform(5, 50), 1)
            edges.append((a, b, w))
            edges.append((b, a, w))
    return nodes, edges


def generate_context(nodes: list[str], edges: list[tuple[str, str, float]],
                     context_type: str, seed: int) -> tuple[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    affected = []

    if context_type == "empty":
        return NO_CONTEXT, []

    if context_type == "noisy":
        return NOISY_CONTEXT_TEMPLATE.format(
            r1=rng.randint(100, 999), r2=rng.randint(100, 999), r3=rng.randint(100, 999)
        ), []

    if context_type == "contradictory":
        e = rng.choice(edges)
        return CONTRADICTORY_TEMPLATE.format(a=e[0], b=e[1]), [(e[0], e[1])]

    # Normal context: pick 2-4 disruptions/preferences
    parts = []
    n_disruptions = rng.randint(1, 3)
    for _ in range(n_disruptions):
        e = rng.choice(edges)
        template = rng.choice(DISRUPTION_TEMPLATES)
        delay = rng.randint(5, 30)
        parts.append(template.format(a=e[0], b=e[1], d=delay))
        affected.append((e[0], e[1]))
    n_prefs = rng.randint(0, 2)
    for _ in range(n_prefs):
        n = rng.choice(nodes)
        template = rng.choice(PREFERENCE_TEMPLATES)
        parts.append(template.format(a=n, b=n, d=rng.randint(3, 15)))

    return " ".join(parts), affected


def generate_dataset(n_instances: int = 50, n_nodes: int = 15,
                     base_seed: int = 42) -> list[GraphInstance]:
    instances = []
    nodes, edges = generate_graph(n_nodes=n_nodes, seed=base_seed)

    for i in range(n_instances):
        seed = base_seed + i
        rng = random.Random(seed)

        # Pick source and target
        src = rng.choice(nodes)
        tgt = rng.choice([n for n in nodes if n != src])

        # Context type distribution: 70% normal, 10% each for stress tests
        r = rng.random()
        if r < 0.7:
            ctx_type = "normal"
        elif r < 0.8:
            ctx_type = "empty"
        elif r < 0.9:
            ctx_type = "noisy"
        else:
            ctx_type = "contradictory"

        context, affected = generate_context(nodes, edges, ctx_type, seed)
        iid = f"graph_{i:04d}"
        inst = GraphInstance(
            instance_id=iid,
            nodes=nodes,
            edges=edges,
            source=src,
            target=tgt,
            context=context,
            context_type=ctx_type,
            affected_edges=affected,
            seed=seed,
        )
        instances.append(inst)

    return instances


# ─── Dijkstra ───

def dijkstra(nodes: list[str], edges: list[tuple[str, str, float]],
             source: str, target: str,
             weight_overrides: dict[tuple[str, str], float] | None = None
             ) -> tuple[list[str], float]:
    """Dijkstra's shortest path. Returns (path, cost) or ([], inf)."""
    adj: dict[str, list[tuple[str, float]]] = {n: [] for n in nodes}
    for src, dst, w in edges:
        eff_w = w
        if weight_overrides and (src, dst) in weight_overrides:
            eff_w = weight_overrides[(src, dst)]
        adj[src].append((dst, eff_w))

    dist = {n: float('inf') for n in nodes}
    prev = {n: None for n in nodes}
    dist[source] = 0
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == target:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    if dist[target] == float('inf'):
        return [], float('inf')

    path = []
    cur = target
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, dist[target]


# ─── Evaluation ───

def evaluate_path(path: list[str], cost: float, instance: GraphInstance,
                  weight_overrides: dict[tuple[str, str], float] | None = None) -> dict:
    """Evaluate a routing solution. Returns metrics dict."""
    # Compute baseline (no overrides)
    base_path, base_cost = dijkstra(
        instance.nodes, instance.edges, instance.source, instance.target
    )

    # Check if the path avoids disrupted edges
    disrupted_avoided = 0
    disrupted_total = len(instance.affected_edges)
    if disrupted_total > 0:
        path_edges = set()
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i + 1]))
        for ae in instance.affected_edges:
            if ae not in path_edges and (ae[1], ae[0]) not in path_edges:
                disrupted_avoided += 1

    return {
        "path": path,
        "cost": cost,
        "baseline_cost": base_cost,
        "cost_ratio": cost / base_cost if base_cost > 0 else 1.0,
        "path_length": len(path),
        "baseline_path_length": len(base_path),
        "valid_path": len(path) > 0 and path[0] == instance.source and path[-1] == instance.target,
        "disrupted_edges_avoided": disrupted_avoided,
        "disrupted_edges_total": disrupted_total,
        "avoidance_rate": disrupted_avoided / disrupted_total if disrupted_total > 0 else None,
    }


# ─── Validator ───

class GraphRoutingValidator(BaseValidator):
    def __init__(self, strength: str = "full", max_adjustment_pct: float = 0.5,
                 min_weight: float = 1.0):
        super().__init__(name="graph_routing_validator", strength=strength)
        self.max_adjustment_pct = max_adjustment_pct
        self.min_weight = min_weight

    def _check_schema(self, proposal: dict) -> ValidationResult | None:
        if not isinstance(proposal, dict):
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=["proposal is not a dict"],
            )
        if "edge_adjustments" not in proposal:
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=["missing 'edge_adjustments' key"],
            )
        if not isinstance(proposal["edge_adjustments"], list):
            return ValidationResult(
                decision="rejected",
                original_proposal=proposal,
                validated_proposal=None,
                reasons=["'edge_adjustments' is not a list"],
            )
        return None

    def _full_validate(self, proposal: dict, context: dict) -> ValidationResult:
        edges = context.get("edges", [])
        edge_set = {(e[0], e[1]) for e in edges}
        edge_weights = {(e[0], e[1]): e[2] for e in edges}

        valid_adjustments = []
        reasons = []
        clamped = []

        for adj in proposal["edge_adjustments"]:
            if not isinstance(adj, dict):
                reasons.append(f"skipped non-dict adjustment: {adj}")
                continue
            src = adj.get("source")
            dst = adj.get("destination")
            factor = adj.get("weight_multiplier")

            if src is None or dst is None or factor is None:
                reasons.append(f"skipped adjustment with missing fields: {adj}")
                continue

            if (src, dst) not in edge_set:
                reasons.append(f"rejected nonexistent edge ({src}, {dst})")
                continue

            if not isinstance(factor, (int, float)):
                reasons.append(f"rejected non-numeric multiplier for ({src}, {dst})")
                continue

            # Clamp multiplier
            original_factor = factor
            lower = 1.0 - self.max_adjustment_pct
            upper = 1.0 + self.max_adjustment_pct
            factor = max(lower, min(upper, factor))

            orig_w = edge_weights[(src, dst)]
            new_w = max(self.min_weight, orig_w * factor)

            if factor != original_factor:
                clamped.append(f"({src},{dst}): {original_factor:.2f} -> {factor:.2f}")

            valid_adjustments.append({
                "source": src,
                "destination": dst,
                "weight_multiplier": factor,
                "new_weight": round(new_w, 2),
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
            validated_proposal={"edge_adjustments": valid_adjustments},
            reasons=reasons,
            clamped_fields=clamped,
        )


# ─── Tool Schema ───

GRAPH_ROUTING_TOOL = {
    "name": "propose_edge_adjustments",
    "description": (
        "Propose edge weight adjustments for a graph routing problem based on "
        "contextual information about disruptions, delays, and preferences. "
        "Each adjustment is a multiplier applied to the original edge weight. "
        "Use multipliers > 1.0 to penalize edges (make them more costly) and "
        "< 1.0 to favor edges (make them cheaper). Stay within [0.5, 1.5]."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "edge_adjustments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source node name"},
                        "destination": {"type": "string", "description": "Destination node name"},
                        "weight_multiplier": {
                            "type": "number",
                            "description": "Multiplier for edge weight. >1 penalizes, <1 favors. Range [0.5, 1.5]."
                        },
                        "reasoning": {"type": "string", "description": "Brief reason for this adjustment"},
                    },
                    "required": ["source", "destination", "weight_multiplier"],
                },
                "description": "List of edge weight adjustments based on context",
            },
        },
        "required": ["edge_adjustments"],
    },
}


# ─── Prompts ───

SYSTEM_PROMPT = (
    "You are a routing advisor. Given a transport graph and contextual information "
    "about disruptions, preferences, and conditions, propose edge weight adjustments "
    "to improve routing quality. Use the propose_edge_adjustments tool. "
    "Only adjust edges that are relevant to the context. Be conservative."
)

def make_user_prompt(instance: GraphInstance) -> str:
    edge_list = "\n".join(
        f"  {s} -> {d} (weight: {w})" for s, d, w in instance.edges
    )
    return (
        f"Route from: {instance.source}\n"
        f"Route to: {instance.target}\n\n"
        f"Graph edges:\n{edge_list}\n\n"
        f"Context: {instance.context}\n\n"
        f"Propose edge weight adjustments to improve routing based on the context."
    )


# ─── Simple Rules Baseline ───

def rules_baseline(instance: GraphInstance) -> dict[tuple[str, str], float]:
    """Simple rule-based context interpreter.
    Looks for keywords in context and penalizes mentioned edges."""
    overrides = {}
    context_lower = instance.context.lower()

    for src, dst, w in instance.edges:
        src_lower = src.lower()
        dst_lower = dst.lower()
        penalty = 1.0

        # If edge endpoints are mentioned with disruption keywords
        if src_lower in context_lower or dst_lower in context_lower:
            for keyword in ["construction", "accident", "flooding", "closure",
                           "blocked", "delay", "maintenance", "emergency", "icy"]:
                if keyword in context_lower:
                    penalty = max(penalty, 1.3)
                    break

        if penalty > 1.0:
            overrides[(src, dst)] = w * penalty

    return overrides


# ─── Run Functions ───

def run_baseline(instance: GraphInstance) -> dict:
    path, cost = dijkstra(instance.nodes, instance.edges, instance.source, instance.target)
    return evaluate_path(path, cost, instance)


def run_rules(instance: GraphInstance) -> dict:
    overrides = rules_baseline(instance)
    path, cost = dijkstra(instance.nodes, instance.edges, instance.source, instance.target, overrides)
    return evaluate_path(path, cost, instance, overrides)


def run_algogen(instance: GraphInstance, proposal_client, validator: GraphRoutingValidator) -> dict:
    """Run the full Algogen pipeline."""
    user_prompt = make_user_prompt(instance)
    raw_proposal, log = proposal_client.propose(
        benchmark="graph_routing",
        instance_id=instance.instance_id,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_definition=GRAPH_ROUTING_TOOL,
        tool_name="propose_edge_adjustments",
    )

    ctx = {
        "instance_id": instance.instance_id,
        "edges": instance.edges,
        "nodes": instance.nodes,
    }
    vresult = validator.validate(raw_proposal, ctx)
    log.validator_decision = vresult.decision

    if vresult.decision in ("accepted", "clamped"):
        # Apply validated adjustments
        edge_weights = {(e[0], e[1]): e[2] for e in instance.edges}
        overrides = {}
        adjustments = vresult.validated_proposal.get("edge_adjustments", [])
        for adj in adjustments:
            src = adj.get("source")
            dst = adj.get("destination")
            if src is None or dst is None:
                continue
            if "new_weight" in adj:
                overrides[(src, dst)] = adj["new_weight"]
            elif "weight_multiplier" in adj:
                # Raw proposal (no-validator mode): compute weight ourselves
                orig_w = edge_weights.get((src, dst))
                if orig_w is not None:
                    mult = adj["weight_multiplier"]
                    if isinstance(mult, (int, float)):
                        overrides[(src, dst)] = max(1.0, orig_w * mult)

        path, cost = dijkstra(instance.nodes, instance.edges,
                             instance.source, instance.target, overrides)
        metrics = evaluate_path(path, cost, instance, overrides)
    else:
        # Fallback to baseline
        path, cost = dijkstra(instance.nodes, instance.edges,
                             instance.source, instance.target)
        metrics = evaluate_path(path, cost, instance)

    metrics["validator_decision"] = vresult.decision
    metrics["proposal_accepted"] = vresult.decision in ("accepted", "clamped")
    metrics["n_adjustments_proposed"] = (
        len(raw_proposal.get("edge_adjustments", [])) if raw_proposal else 0
    )
    metrics["n_adjustments_validated"] = (
        len(vresult.validated_proposal["edge_adjustments"])
        if vresult.validated_proposal else 0
    )
    return metrics
