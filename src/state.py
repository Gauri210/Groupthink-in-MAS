# Shared state object passed through every node of the LangGraph.
# TODO: define fields (question context, history of agent answers, round tracking,
# final outcome flags like flipped / rescued).

from typing import TypedDict


class DeliberationState(TypedDict):
    pass  # TODO
