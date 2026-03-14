"""
Minimal example: Propose-Validate-Repair pipeline for NL-to-PDDL translation.

This demonstrates the core Algogens pattern on the Gripper domain from Planetarium.
Requires: ANTHROPIC_API_KEY environment variable, pyperplan installed.

Usage:
    python examples/pddl_pipeline.py
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.proposal_client import ProposalClient

# ── PDDL Domain Definition (Gripper) ──────────────────────────────────────────

GRIPPER_DOMAIN = """(define (domain gripper)
  (:predicates (room ?r) (ball ?b) (gripper ?g) (at-robby ?r)
               (at ?b ?r) (free ?g) (carry ?b ?g))
  (:action move :parameters (?from ?to)
    :precondition (and (room ?from) (room ?to) (at-robby ?from))
    :effect (and (at-robby ?to) (not (at-robby ?from))))
  (:action pick :parameters (?obj ?room ?gripper)
    :precondition (and (ball ?obj) (room ?room) (gripper ?gripper)
                       (at ?obj ?room) (at-robby ?room) (free ?gripper))
    :effect (and (carry ?obj ?gripper) (not (at ?obj ?room)) (not (free ?gripper))))
  (:action drop :parameters (?obj ?room ?gripper)
    :precondition (and (ball ?obj) (room ?room) (gripper ?gripper)
                       (carry ?obj ?gripper) (at-robby ?room))
    :effect (and (at ?obj ?room) (free ?gripper) (not (carry ?obj ?gripper)))))
"""

# ── Validator Stack ───────────────────────────────────────────────────────────

DOMAIN_PREDICATES = {"room", "ball", "gripper", "at-robby", "at", "free", "carry"}


def v1_schema(resp: dict) -> tuple[bool, str]:
    """V1: Check required fields exist."""
    if not isinstance(resp, dict):
        return False, "Response is not a dict"
    for field in ["objects", "init", "goal"]:
        if field not in resp or not resp[field]:
            return False, f"Missing or empty '{field}'"
    return True, ""


def v2_grounding(resp: dict) -> tuple[bool, str]:
    """V2: Check predicates and objects are domain-legal."""
    objects = set(resp.get("objects", []))
    for section in ["init", "goal"]:
        for pred_str in resp.get(section, []):
            parts = pred_str.strip().split()
            if not parts:
                return False, f"Empty predicate in {section}"
            if parts[0] not in DOMAIN_PREDICATES:
                return False, f"Invalid predicate '{parts[0]}' in {section}"
            for arg in parts[1:]:
                if arg not in objects:
                    return False, f"'{pred_str}' references undefined object '{arg}'"
    return True, ""


def v3_consistency(resp: dict) -> tuple[bool, str]:
    """V3: Check for duplicate predicates."""
    init = resp.get("init", [])
    if len(init) != len(set(init)):
        dupes = [p for p in init if init.count(p) > 1]
        return False, f"Duplicate predicates: {set(dupes)}"
    return True, ""


def v4_solvability(resp: dict) -> tuple[bool, str]:
    """V4: Check if pyperplan can solve the assembled PDDL."""
    import tempfile, os, signal
    from pyperplan.pddl.parser import Parser
    from pyperplan import grounding as pyp_grounding
    from pyperplan.search import breadth_first_search

    obj_str = " ".join(resp["objects"])
    init_str = " ".join(f"({p})" for p in resp["init"])
    goal_str = " ".join(f"({p})" for p in resp["goal"])
    problem_pddl = f"""(define (problem generated)
    (:domain gripper)
    (:objects {obj_str})
    (:init {init_str})
    (:goal (and {goal_str})))"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as df:
        df.write(GRIPPER_DOMAIN)
        domain_file = df.name
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pddl', delete=False) as pf:
        pf.write(problem_pddl)
        problem_file = pf.name

    try:
        def timeout_handler(signum, frame):
            raise TimeoutError()

        parser = Parser(domain_file, problem_file)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        task = pyp_grounding.ground(problem)

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        try:
            solution = breadth_first_search(task)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        if solution is None:
            return False, "No plan found (unsolvable or too complex)"
        return True, ""
    except TimeoutError:
        return False, "Planner timeout (10s)"
    except Exception as e:
        return False, f"Planner error: {e}"
    finally:
        os.unlink(domain_file)
        os.unlink(problem_file)


def validate(resp: dict) -> tuple[bool, str]:
    """Run all validators in order, short-circuit on first failure."""
    for name, fn in [("V1", v1_schema), ("V2", v2_grounding),
                     ("V3", v3_consistency), ("V4", v4_solvability)]:
        ok, msg = fn(resp)
        if not ok:
            return False, f"{name}: {msg}"
    return True, ""


# ── Tool Schema ───────────────────────────────────────────────────────────────

TOOL_DEFINITION = {
    "name": "generate_pddl_problem",
    "description": "Generate a PDDL problem specification from a natural-language description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of object names (e.g., ['room1', 'ball1', 'gripper1'])"
            },
            "init": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of init predicates (e.g., ['room room1', 'at ball1 room1'])"
            },
            "goal": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of goal predicates (e.g., ['at ball1 room2'])"
            }
        },
        "required": ["objects", "init", "goal"]
    }
}

SYSTEM_PROMPT = """You are a PDDL problem generator. Given a natural-language task description
and a PDDL domain definition, generate the correct PDDL problem specification.

IMPORTANT: Use the exact predicate names and argument orders from the domain definition.
For the Gripper domain, the carry predicate is (carry ?ball ?gripper), NOT (carry ?gripper ?ball)."""

# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(nl_description: str, repair_budget: int = 3) -> dict:
    """Run the propose-validate-repair pipeline on a single NL instance."""
    client = ProposalClient(model_id="claude-sonnet-4-6")

    user_prompt = f"""Domain definition:
{GRIPPER_DOMAIN}

Natural-language problem description:
{nl_description}

Generate the PDDL problem specification using the generate_pddl_problem tool."""

    # Initial proposal
    proposal, log = client.propose(
        benchmark="gripper",
        instance_id="example",
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_definition=TOOL_DEFINITION,
        tool_name="generate_pddl_problem",
    )

    for round_num in range(repair_budget + 1):
        if proposal is None:
            print(f"  Round {round_num}: No response from model")
            break

        ok, diagnostic = validate(proposal)
        if ok:
            print(f"  Round {round_num}: All validators passed!")
            return {"proposal": proposal, "rounds": round_num, "status": "success"}

        print(f"  Round {round_num}: {diagnostic}")

        if round_num == repair_budget:
            print(f"  Repair budget exhausted after {repair_budget} rounds")
            return {"proposal": proposal, "rounds": round_num, "status": "exhausted"}

        # Repair: feed diagnostic back to model
        repair_prompt = f"""{user_prompt}

Your previous output was:
{proposal}

The validator found this error:
{diagnostic}

Please fix the error and regenerate the PDDL problem specification."""

        proposal, log = client.propose(
            benchmark="gripper",
            instance_id="example",
            system_prompt=SYSTEM_PROMPT,
            user_prompt=repair_prompt,
            tool_definition=TOOL_DEFINITION,
            tool_name="generate_pddl_problem",
        )

    return {"proposal": proposal, "rounds": repair_budget, "status": "failed"}


if __name__ == "__main__":
    # Example: a simple Gripper instance
    nl = """You have 2 rooms (room1, room2). You have 2 balls (ball1, ball2).
You have 2 grippers (gripper1, gripper2).
The robot is in room1. ball1 is in room1. ball2 is in room2.
gripper1 is free. gripper2 is free.
Goal: ball1 should be in room2. ball2 should be in room1."""

    print("Running Algogens propose-validate-repair pipeline...")
    print(f"Input: {nl}\n")
    result = run_pipeline(nl)
    print(f"\nResult: {result['status']} after {result['rounds']} rounds")
    if result['proposal']:
        print(f"Objects: {result['proposal'].get('objects', [])}")
        print(f"Init: {result['proposal'].get('init', [])}")
        print(f"Goal: {result['proposal'].get('goal', [])}")
