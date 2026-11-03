# Algogens: A Propose-Validate-Repair Pipeline for NL-to-Formal Translation

Algogens is a framework for translating natural-language task descriptions into formal specifications using large language models (LLMs) with structured validation and bounded repair. The core idea: instead of trusting one-shot LLM generation, the system validates the output against deterministic checkers and feeds structured diagnostics back to the model for targeted correction.

## Architecture

```
NL description --> LLM (structured proposal) --> Validator stack --> Accept / Repair
                        ^                            |
                        |_____ diagnostic feedback ___|
```

**Validator stack (for PDDL):**
- **V1 (Schema):** JSON structure conformance
- **V2 (Grounding):** All predicates and objects are domain-legal
- **V3 (Consistency):** No contradictory or duplicate literals
- **V4 (Solvability):** Planner can find a valid plan

The pipeline terminates after at most K+1 model calls (default K=3), with early exit when all validators pass.

## Key Findings (from the paper)

- On clean inputs, generic self-refinement matches structured repair (both ~76-78% on Gripper)
- Under validator-targeted perturbations, structured repair outperforms self-refinement by 33-53 percentage points
- The repair advantage held for both models evaluated (Claude Sonnet 4.6 and GPT-5.4); broader model coverage is future work
- Repair dominates blind retry in cost-accuracy terms (higher accuracy at lower cost), reaching its accuracy at roughly 2x the cost of one-shot generation

## Project Structure

```
src/
  core/
    validator.py          # Abstract validator interface
    proposal_client.py    # LLM client with tool use (Anthropic API)
  benchmarks/
    graph_routing.py      # Context-conditioned shortest-path routing
    knapsack_selection.py # Knapsack item selection with constraints
    semantic_clustering.py # Text clustering on 20 Newsgroups
    clustering_v2.py      # Enhanced clustering with consistency checks
  datasets/
    newsgroups.py         # 20 Newsgroups dataset loader
examples/
  pddl_pipeline.py       # Minimal PDDL pipeline example
requirements.txt          # Python dependencies
```

## Installation

```bash
git clone https://github.com/ashachar/algogens.git
cd algogens
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, an Anthropic API key (set `ANTHROPIC_API_KEY` environment variable).

For the PDDL pipeline, you also need [pyperplan](https://github.com/aibasel/pyperplan):
```bash
pip install pyperplan
```

## Quick Start

```python
from src.core.validator import BaseValidator, ValidationResult
from src.core.proposal_client import ProposalClient

# Initialize the LLM client
client = ProposalClient(model_id="claude-sonnet-4-6")

# Define your validator (see src/benchmarks/ for examples)
# Call client.propose() with your system prompt, user prompt, and tool schema
# Validate the output, and if it fails, feed diagnostics back for repair
```

See `examples/pddl_pipeline.py` for a complete working example.

## Non-PDDL Benchmarks

The framework generalizes beyond PDDL. The `src/benchmarks/` directory includes implementations for:

- **Graph Routing:** LLM adjusts edge weights based on natural-language disruption context; validator ensures graph consistency
- **Knapsack Selection:** LLM selects items subject to capacity constraints; validator checks feasibility
- **Semantic Clustering:** LLM assigns text documents to clusters; validator checks cluster constraints and coverage

Each benchmark demonstrates the propose-validate-repair pattern on a different task type.

## Citation

```bibtex
@unpublished{shachar2026algogens_pipeline,
  title   = {A Propose-Validate-Repair Pipeline for Natural-Language-to-{PDDL}
             Translation: Controlled Experiments on Iterative Refinement,
             Robustness, and Validator Design},
  author  = {Shachar, Amir},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
