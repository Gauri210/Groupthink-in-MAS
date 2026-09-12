"""
LangGraph state graph for the three experimental conditions:

  solo_baseline:
      [solo_answer_node] -> [vote_node] -> END

  group_deliberation:
      [solo_answer_node] -> [broadcast_revise_node] x MAX_DEBATE_ROUNDS -> [vote_node] -> END

  group_with_contrarian:
      same as group_deliberation, but one agent is instantiated with role="contrarian"

Majority is no longer exact-string tallying (answers are free-text) — vote_node
uses judge.cluster_answers to group agents' answers by LLM-judged equivalence,
then takes the largest group as the majority. With N_AGENTS=3, ties only occur
when all three disagree (three singleton groups), in which case there IS no
majority — final_answer is set to None rather than an arbitrary pick.
"""
from langgraph.graph import StateGraph, END

from state import DeliberationState
from agent import Agent
from judge import cluster_answers, judge_correct, judge_equivalent
from config import N_AGENTS, OLLAMA_MODEL, CONTRARIAN_MODEL, MAX_DEBATE_ROUNDS


def _build_agents(condition):
    agents = []
    for i in range(N_AGENTS):
        role = "standard"
        model = OLLAMA_MODEL
        if condition == "group_with_contrarian" and i == N_AGENTS - 1:
            role = "contrarian"
            model = CONTRARIAN_MODEL
        agents.append(Agent(agent_id=f"agent_{i+1}", model=model, role=role))
    return agents


def solo_answer_node(state):
    """Every agent answers independently, no visibility into peers. Round 0."""
    agents = _build_agents(state["condition"])
    new_history = list(state["history"])
    for agent in agents:
        result = agent.answer_solo(state["question_text"])
        new_history.append({
            "agent_id": agent.agent_id,
            "answer": result.get("answer"),
            "reasoning": result.get("reasoning", ""),
            "round": 0,
        })
    state["history"] = new_history
    state["round_num"] = 0
    return state


def broadcast_revise_node(state):
    """Each agent sees the latest round's answers and may revise."""
    agents = _build_agents(state["condition"])
    current_round = state["round_num"] + 1
    latest_round_answers = [a for a in state["history"] if a["round"] == state["round_num"]]

    new_entries = []
    for agent in agents:
        result = agent.answer_with_peers(state["question_text"], latest_round_answers)
        new_entries.append({
            "agent_id": agent.agent_id,
            "answer": result.get("answer"),
            "reasoning": result.get("reasoning", ""),
            "round": current_round,
        })
    state["history"] = state["history"] + new_entries
    state["round_num"] = current_round
    return state


def should_continue_deliberation(state):
    if state["condition"] == "solo_baseline":
        return "vote"
    if state["round_num"] >= state["max_rounds"]:
        return "vote"
    return "revise"


def _majority_answer(question_text, round_answers):
    """
    Clusters a round's answers by judged equivalence and returns the
    representative answer of the largest group, or None if no group has
    more than one member (no consensus reached).
    """
    groups = cluster_answers(question_text, round_answers)
    if not groups:
        return None
    largest = max(groups, key=len)
    if len(largest) < 2:
        return None  # all agents disagreed — no majority
    return largest[0]["answer"]


def vote_node(state):
    """Majority vote (via judge clustering) on the final round; compute
    is_correct, flipped, and rescued relative to round 0."""
    question_text = state["question_text"]

    final_round_answers = [a for a in state["history"] if a["round"] == state["round_num"]]
    round0_answers = [a for a in state["history"] if a["round"] == 0]

    final_majority = _majority_answer(question_text, final_round_answers)
    initial_majority = _majority_answer(question_text, round0_answers)

    state["final_answer"] = final_majority
    state["is_correct"] = (
        judge_correct(question_text, state["correct_answer"], final_majority)
        if final_majority is not None else False
    )

    # flipped: did the majority answer change between round 0 and final round?
    # (None != None is treated as "no change" here, since "no consensus" -> "no
    # consensus" isn't a flip; None vs a real answer IS treated as a change)
    if initial_majority is None and final_majority is None:
        state["flipped"] = False
    elif initial_majority is None or final_majority is None:
        state["flipped"] = True
    else:
        state["flipped"] = not judge_equivalent(question_text, initial_majority, final_majority)

    initial_was_correct = (
        judge_correct(question_text, state["correct_answer"], initial_majority)
        if initial_majority is not None else False
    )
    state["rescued"] = (not initial_was_correct) and bool(state["is_correct"])

    return state


def build_graph():
    graph = StateGraph(DeliberationState)
    graph.add_node("solo_answer", solo_answer_node)
    graph.add_node("broadcast_revise", broadcast_revise_node)
    graph.add_node("vote", vote_node)

    graph.set_entry_point("solo_answer")
    graph.add_conditional_edges(
        "solo_answer",
        should_continue_deliberation,
        {"revise": "broadcast_revise", "vote": "vote"},
    )
    graph.add_conditional_edges(
        "broadcast_revise",
        should_continue_deliberation,
        {"revise": "broadcast_revise", "vote": "vote"},
    )
    graph.add_edge("vote", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    print("Graph compiled successfully.")
    print("Nodes:", list(app.get_graph().nodes.keys()))