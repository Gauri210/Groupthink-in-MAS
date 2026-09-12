"""
Shared state object that flows through every node of the LangGraph.
"""
from typing import TypedDict, List, Dict, Optional


class AgentAnswer(TypedDict):
    agent_id: str
    answer: Optional[str]   # free-text answer, or None if parsing failed
    reasoning: str
    round: int               # 0 = initial/solo, 1+ = after seeing others


class DeliberationState(TypedDict):
    # question context
    question_id: int
    question_text: str
    correct_answer: str
    lure_answer: Optional[str]   # plausible wrong answer, for analysis only
    condition: str                 # "solo_baseline" | "group_deliberation" | "group_with_contrarian"

    # rolling record of every agent's answer at every round
    history: List[AgentAnswer]

    # current round tracking
    round_num: int
    max_rounds: int

    # final outcome (filled in by the vote node)
    final_answer: Optional[str]     # None if no majority reached
    is_correct: Optional[bool]       # judged against correct_answer
    flipped: Optional[bool]           # did majority change from round 0 to final?
    rescued: Optional[bool]           # did a wrong initial majority get corrected?