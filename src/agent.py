# Ollama-backed agent wrapper.
# TODO: implement prompting for solo answer, revised answer (after seeing peers),
# and contrarian answer.


class Agent:
    def __init__(self, agent_id, model, role="standard"):
        # TODO
        pass

    def answer_solo(self, question_text, options):
        """TODO: prompt the model for an independent answer, return {"answer": ..., "reasoning": ...}"""
        pass

    def answer_with_peers(self, question_text, options, peer_answers):
        """TODO: prompt the model with peer answers visible, allow it to revise or hold its answer."""
        pass
