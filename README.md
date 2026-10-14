# Groupthink in Multi-Agent LLM Systems

> Does adding more AI agents make decisions worse?

An Asch-inspired controlled experiment that tests whether LLM-based agents
exhibit human-like conformity during group deliberation — abandoning a correct
individual answer to match group consensus — and whether a designated
contrarian agent can mitigate this effect.

---

## Research Question

Do LLM agents exhibit Asch-style conformity, and does an explicit dissenting
agent role reduce it?

### Hypotheses

- **H1** — Group deliberation accuracy < solo accuracy (conformity cost exists)
- **H2** — A contrarian agent recovers accuracy lost to conformity
- **H3** — Conformity is more pronounced on trick questions than factual or logic ones

---

## Key Findings

| Condition | Accuracy | Flip Rate | Rescue Rate |
|---|---|---|---|
| solo_baseline | 76.0% | 0.0% | 0.0% |
| group_deliberation | 76.0% | 10.0% | 6.0% |
| group_with_contrarian | 70.0% | 10.0% | 0.0% |

- Peer exposure causes agents to revise correct answers at a **10% rate**
- Group deliberation matches solo accuracy but adds modest rescue capability
- An unguided contrarian **reduces accuracy by 6pp** with zero rescue benefit
- Trick questions showed the highest condition sensitivity (−18.8pp under contrarian vs. group)

---

## System Design

**Framework:** LangGraph StateGraph. Each condition is an explicit state graph
with shared state passed between nodes. Conditions differ only in graph
topology, keeping the experiment methodologically clean.

**Agents:** 3 × qwen2.5:7b via Ollama, temperature 0.7. Single homogeneous
model across all agents to isolate group-dynamic effects.

**Judge:** gemma2:9b at temperature 0 for free-text equivalence scoring,
with a deterministic numeric pre-check for currency/percentage variants.

---

## Experimental Conditions

| Condition | Description |
|---|---|
| `solo_baseline` | Each agent answers independently; majority of 3 is final |
| `group_deliberation` | Round 1: all agents answer. Round 2: each sees peers and may revise. Majority vote. |
| `group_with_contrarian` | Identical to group, except Agent 3 is instructed to challenge emerging consensus before agreeing |

**Question bank:** 50 open-ended free-text questions — 17 logic/math, 17 trivia/factual,
16 trick questions with a plausible-but-wrong attractor answer.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Pull models (requires Ollama)
ollama pull qwen2.5:7b
ollama pull gemma2:9b
```

## Usage

```bash
# Run all three conditions
python src/run_experiment.py

# Analyze and generate CSVs
python src/analyze_results.py

# Launch dashboard
python -m streamlit run src/dashboard.py
```

---

## Project Structure

```
.
├── src/
│   ├── agent.py                # Agent node definitions
│   ├── config.py               # Model and experiment config
│   ├── dashboard.py            # Streamlit dashboard
│   ├── graph.py                # LangGraph StateGraph definitions
│   ├── judge.py                # LLM-as-judge scorer
│   ├── results.py              # Results parsing and aggregation
│   ├── run_experiment.py       # Entry point — runs all three conditions
│   └── state.py                # Shared state schema
├── results/
│   ├── raw/                    # Raw JSON outputs per condition
│   └── processed/              # Summary CSVs (dashboard input)
├── questions.json              # Question bank with category labels
└── requirements.txt
```

---

## Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent orchestration
- [Ollama](https://ollama.com) — local LLM inference
- [Streamlit](https://streamlit.io) — results dashboard
- qwen2.5:7b — deliberating agents
- gemma2:9b — LLM-as-judge scorer

---

## Limitations

- n=50; differences are descriptive, not statistically significant
- Single base model across all agents
- LLM judge introduces its own scoring error

---
