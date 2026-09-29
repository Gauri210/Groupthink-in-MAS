import json
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from config import OLLAMA_BASE_URL, TEMPERATURE


JSON_FORMAT_INSTRUCTIONS = """
You MUST respond ONLY with valid JSON in exactly this format:

{
    "answer": "<your answer — a short number, word, or phrase>",
    "reasoning": "<1 sentence explaining your reasoning>"
}

Keep the answer as short and direct as possible (e.g. a single number or
a few words). Do not include markdown, code fences, or any text outside
the JSON.
"""


class Agent:

    def __init__(self, agent_id, model, role="standard"):
        self.agent_id = agent_id
        self.role = role

        self.llm = ChatOllama(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=TEMPERATURE
        )

    def parse_response(self, raw_text):
        """
        Parse the model's response into {"answer": <str or None>, "reasoning": str}.

        No longer validates against a fixed option set (free-text answers now) —
        just requires a non-empty "answer" field. A genuine parse failure returns
        answer=None so it can be filtered out of analysis rather than silently
        counted as a real vote.
        """
        cleaned_text = raw_text.strip()
        cleaned_text = re.sub(r"```json\s*", "", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r"```\s*", "", cleaned_text)

        def _finalize(answer, reasoning):
            if answer is None:
                return {"answer": None, "reasoning": str(reasoning)[:500]}
            answer = str(answer).strip()
            if not answer:
                return {"answer": None, "reasoning": "EMPTY_ANSWER: " + str(reasoning)[:500]}
            return {"answer": answer, "reasoning": str(reasoning)[:500]}

        # 1. Try normal JSON parsing
        try:
            parsed = json.loads(cleaned_text)
            if isinstance(parsed, dict) and "answer" in parsed:
                return _finalize(parsed.get("answer"), parsed.get("reasoning", ""))
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Try to find JSON embedded in surrounding text
        json_match = re.search(r'\{.*?"answer"\s*:\s*"([^"]*)".*?\}', cleaned_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                return _finalize(parsed.get("answer"), parsed.get("reasoning", ""))
            except json.JSONDecodeError:
                pass

        # 3. Fallback: regex search for "answer": "..."
        match = re.search(r'"answer"\s*:\s*"([^"]*)"', cleaned_text, re.IGNORECASE)
        if match:
            return _finalize(match.group(1), raw_text)

        # 4. Complete parsing failure
        return {"answer": None, "reasoning": "PARSE_FAILED: " + raw_text[:500]}

    def answer_solo(self, question_text):
        """Answer a free-text question independently."""
        system_prompt = (
            "You are an agent answering a question independently.\n\n"
            "Think briefly and carefully, then commit to ONE answer.\n"
            + JSON_FORMAT_INSTRUCTIONS
        )

        user_prompt = f"Question: {question_text}"

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return self.parse_response(response.content)

    def answer_with_peers(self, question_text, peer_answers_list):
        """Answer a free-text question while considering other agents' answers."""
        peers_text_parts = []
        for peer in peer_answers_list:
            if isinstance(peer, dict):
                peer_id = peer.get("agent_id")
                peer_answer = peer.get("answer")
                peer_reasoning = peer.get("reasoning", "")
            else:
                peer_id = getattr(peer, "agent_id", None)
                peer_answer = getattr(peer, "answer", None)
                peer_reasoning = getattr(peer, "reasoning", "")

            if peer_id != self.agent_id:
                peers_text_parts.append(
                    f"- Agent {peer_id} answered \"{peer_answer}\" because {peer_reasoning}"
                )
        peers_text = "\n".join(peers_text_parts)

        if self.role == "contrarian":
            role_instructions = (
                "You are a contrarian reasoning agent.\n\n"
                "You are reviewing a question along with answers and reasoning from other agents.\n\n"
                "Critically challenge the group's emerging consensus if you believe it is wrong. "
                "Look for mistakes, unsupported assumptions, and overlooked details.\n\n"
                "However, do NOT disagree merely for the sake of being different. If the consensus "
                "is correct and well-supported, you should agree with it.\n\n"
                "Think briefly and commit to ONE answer.\n"
            )
        else:
            role_instructions = (
                "You are a reasoning agent participating in a group.\n\n"
                "Consider the answers and reasoning provided by the other agents.\n\n"
                "Decide whether to keep or change your own answer based on the strength of the "
                "arguments, not simply based on majority vote.\n\n"
                "Think briefly and commit to ONE answer.\n"
            )

        system_prompt = role_instructions + JSON_FORMAT_INSTRUCTIONS

        user_prompt = (
            f"Question: {question_text}\n\n"
            f"Other agents' current answers:\n{peers_text}\n\n"
            f"What is your answer now?"
        )

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return self.parse_response(response.content)


def normalize_answer(answer):
    """
    Normalize a free-text answer for equivalence comparison during voting
    and correctness checking: lowercase, strip whitespace, strip surrounding
    punctuation/articles. Used by graph.py's vote node — kept here so agent
    output and comparison logic use the same normalization rules.
    """
    if answer is None:
        return None
    text = str(answer).strip().lower()
    text = re.sub(r'^(the|a|an)\s+', '', text)
    text = re.sub(r'[.,!?;:\'"]+$', '', text)
    text = text.strip()
    return text if text else None