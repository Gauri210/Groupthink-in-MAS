# LangGraph state graph wiring the deliberation nodes.
# TODO: implement nodes for solo answer, broadcast/revise loop, and majority vote,
# with conditional edges based on `condition` (solo_baseline / group_deliberation /
# group_with_contrarian).


def solo_answer_node(state):
    """TODO: every agent answers independently, round 0."""
    pass


def broadcast_revise_node(state):
    """TODO: each agent sees the latest round's answers and may revise."""
    pass


def should_continue_deliberation(state):
    """TODO: decide whether to loop back to broadcast_revise or move to vote."""
    pass


def vote_node(state):
    """TODO: majority vote on final round; compute flipped/rescued flags vs round 0."""
    pass


def build_graph():
    """TODO: assemble StateGraph with nodes above and conditional edges, return compiled graph."""
    pass
