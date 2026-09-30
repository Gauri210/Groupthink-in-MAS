"""
A stronger local model (e.g. gemma2:9b) used to judge free-text answer
equivalence, since agents phrasing "4" vs "four" or "Paris" vs "the city of
Paris" can't be resolved by exact string matching.

IMPORTANT: before calling the LLM at all, both judge_equivalent and
judge_correct try a deterministic numeric comparison first. This matters a
lot for a question bank with currency/count answers ("$0.05" vs "10 cents",
"$0.10" vs "0.1") — an LLM judge can and will get these wrong (e.g. merging
"$0.05" and "10 cents" into "the same answer" because both are small money
amounts), which silently corrupts majority voting and correctness scoring.
The numeric check removes that failure mode entirely for anything that
parses cleanly as a number; only genuinely non-numeric answers fall through
to the LLM.
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


def _try_parse_number(text):
    """
    Attempt to extract a single numeric value from free text, handling
    common formats: "$0.05", "0.05", "5 cents" (converted to 0.05),
    "10 dollars", "4", "four" is NOT handled here (word-numbers fall
    through to the LLM judge instead — deliberately, since spelled-out
    numbers are more ambiguous to regex-parse reliably).

    Returns a float, or None if no confident numeric parse was possible.
    """
    if text is None:
        return None
    t = str(text).strip().lower()

    # "$0.05", "$1.10", "0.05", "5"
    dollar_match = re.fullmatch(r'\$?\s*(\d+(?:\.\d+)?)', t)
    if dollar_match:
        return float(dollar_match.group(1))

    # "5 cents", "10 cents" -> convert to dollar-equivalent units
    cents_match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*cents?', t)
    if cents_match:
        return float(cents_match.group(1)) / 100.0

    # "5 dollars", "1.10 dollars"
    dollars_match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*dollars?', t)
    if dollars_match:
        return float(dollars_match.group(1))

    return None


def _numeric_equivalent(answer_a, answer_b, tolerance=1e-6):
    """
    Returns True/False if BOTH answers parse as numbers and can be compared
    directly, or None if either answer doesn't parse cleanly (caller should
    fall back to the LLM judge in that case).
    """
    num_a = _try_parse_number(answer_a)
    num_b = _try_parse_number(answer_b)
    if num_a is None or num_b is None:
        return None
    return abs(num_a - num_b) < tolerance


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

    Tries a deterministic numeric comparison FIRST — only calls the LLM if
    at least one answer doesn't parse as a clean number.
    """
    if answer_a is None or answer_b is None:
        return False

    numeric_result = _numeric_equivalent(answer_a, answer_b)
    if numeric_result is not None:
        return numeric_result

    system_prompt = """
You are a strict judge comparing two answers to the same question.

Decide whether the two answers represent the SAME underlying answer, allowing
for differences in phrasing or wording (e.g. "Paris" and "the city of Paris"
are the same). Do NOT treat different numeric values as the same just because
they are both plausible-sounding — only exact numeric equivalents count.

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

    Same numeric-first approach as judge_equivalent.
    """
    if given_answer is None:
        return False

    numeric_result = _numeric_equivalent(correct_answer, given_answer)
    if numeric_result is not None:
        return numeric_result

    system_prompt = """
You are a strict judge grading an answer against a known correct answer.

Decide whether the given answer is correct, allowing for differences in
phrasing or wording (e.g. "Paris" and "the city of Paris" both count as
correct if the correct answer is "Paris"). Do NOT treat different numeric
values as correct just because they are close or plausible-sounding — only
exact numeric equivalents count as correct.

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
    clusters using judge_equivalent (numeric-first, LLM fallback). Returns a
    list of groups, each group a list of the original dicts. None answers
    (parse failures) are excluded entirely.
    """
    valid_answers = [a for a in agent_answers if a.get("answer") is not None]
    groups = []

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