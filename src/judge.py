"""
A stronger local model (e.g. qwen2.5:14b) used to judge free-text answer
equivalence, since agents phrasing "4" vs "four" or "Paris" vs "the city of
Paris" can't be resolved by exact string matching.

Two distinct jobs, two distinct prompts — don't conflate them:
  - judge_equivalent: do two answers mean the same thing? (generic, no ground truth)
  - judge_correct: does an answer match the KNOWN correct answer? (ground-truth-aware)
"""
import json
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from config import OLLAMA_BASE_URL, JUDGE_MODEL, JUDGE_TEMPERATURE


_judge_llm = None


def _get_judge_llm():
    global _judge_llm
    if _judge_llm is None:
        _judge_llm = ChatOllama(
            model=JUDGE_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=JUDGE_TEMPERATURE
        )
    return _judge_llm


def _parse_bool_json(raw_text, key):
    """Shared parsing for {"<key>": true/false, "reasoning": "..."} responses.
    Defaults to False on any parse failure — safer to undercount agreement/
    correctness than to falsely inflate it."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and key in parsed:
            return bool(parsed[key])
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(rf'"{key}"\s*:\s*(true|false)', cleaned, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"

    return False  # fail-safe default


def judge_equivalent(question_text, answer_a, answer_b):
    """
    Generic equivalence check between two free-text answers, used to cluster
    agents into agreement groups for majority voting. No ground truth involved.
    """
    if answer_a is None or answer_b is None:
        return False

    system_prompt = """
You are a strict judge comparing two answers to the same question.

Decide whether the two answers represent the SAME underlying answer, allowing
for differences in phrasing, formatting, or units (e.g. "4" and "four" are the
same; "Paris" and "the city of Paris" are the same; "9" and "90" are NOT the same).

Respond ONLY with valid JSON in exactly this format:
{"equivalent": true or false, "reasoning": "<one sentence>"}
No markdown, no extra text.
"""
    user_prompt = (
        f"Question: {question_text}\n\n"
        f"Answer A: {answer_a}\n"
        f"Answer B: {answer_b}\n\n"
        f"Are these the same answer?"
    )

    response = _get_judge_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return _parse_bool_json(response.content, "equivalent")


def judge_correct(question_text, correct_answer, given_answer):
    """
    Ground-truth-aware check: does given_answer match the known correct_answer?
    Used for final scoring, not for clustering agents against each other.
    """
    if given_answer is None:
        return False

    system_prompt = """
You are a strict judge grading an answer against a known correct answer.

Decide whether the given answer is correct, allowing for differences in
phrasing, formatting, or units (e.g. "4" and "four" both count as correct if
the correct answer is "4").

Respond ONLY with valid JSON in exactly this format:
{"correct": true or false, "reasoning": "<one sentence>"}
No markdown, no extra text.
"""
    user_prompt = (
        f"Question: {question_text}\n\n"
        f"Correct answer: {correct_answer}\n"
        f"Given answer: {given_answer}\n\n"
        f"Is the given answer correct?"
    )

    response = _get_judge_llm().invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return _parse_bool_json(response.content, "correct")


def cluster_answers(question_text, agent_answers):
    """
    Groups a list of {"agent_id": ..., "answer": ...} dicts into agreement
    clusters using pairwise judge_equivalent calls. Returns a list of groups,
    each group a list of the original dicts. None answers (parse failures)
    are excluded entirely — they can't be judged equivalent to anything.

    This is O(n^2) in judge calls, which is fine for N_AGENTS=3 (at most 3
    pairwise comparisons) but would need a smarter approach for larger N.
    """
    valid_answers = [a for a in agent_answers if a.get("answer") is not None]
    groups = []  # list of lists

    for entry in valid_answers:
        placed = False
        for group in groups:
            representative = group[0]
            if judge_equivalent(question_text, representative["answer"], entry["answer"]):
                group.append(entry)
                placed = True
                break
        if not placed:
            groups.append([entry])

    return groups