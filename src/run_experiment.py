print("SCRIPT STARTED", flush=True)

import argparse
import datetime
import json
import os

from graph import build_graph
from config import (
    CONDITIONS,
    MAX_DEBATE_ROUNDS,
    QUESTIONS_PATH,
    RESULTS_DIR,
)


def load_questions(path):
    """Load questions from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(limit=None, question_ids=None):
    """Run the debate graph for the selected questions and conditions."""

    # Load all questions
    questions = load_questions(QUESTIONS_PATH)

    # Filter questions by ID or limit
    if question_ids:
        questions = [
            q for q in questions
            if q["id"] in question_ids
        ]

    elif limit is not None:
        questions = questions[:limit]

    # Build the graph once and reuse it
    app = build_graph()

    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Generate timestamped output filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    out_path = os.path.join(
        RESULTS_DIR,
        f"run_{timestamp}.jsonl"
    )

    # Calculate total number of runs
    total = len(questions) * len(CONDITIONS)
    done = 0
    failed = 0

    print(f"Questions: {len(questions)}")
    print(f"Conditions: {len(CONDITIONS)}")
    print(f"Total runs: {total}")
    print(f"Output: {out_path}")
    print()

    # Open JSONL output file
    with open(out_path, "w", encoding="utf-8") as f:

        # Process every question
        for q in questions:

            # Run every experimental condition
            for condition in CONDITIONS:

                # Build initial state
                initial_state = {
                    "question_id": q["id"],
                    "question_text": q["question"],
                    "correct_answer": q["correct_answer"],
                    "wrong_answer": q.get("wrong_answer"),
                    "condition": condition,

                    "history": [],
                    "round_num": 0,
                    "max_rounds": MAX_DEBATE_ROUNDS,

                    "final_answer": None,
                    "is_correct": None,
                    "flipped": None,
                    "rescued": None,
                }

                # Run the graph, but don't let one bad question kill the
                # entire multi-hour run — log the failure and move on.
                try:
                    final_state = app.invoke(initial_state)

                    record = {
                        "question_id": q["id"],
                        "category": q["category"],
                        "difficulty": q["difficulty"],

                        "condition": condition,

                        "correct_answer": q["correct_answer"],
                        "wrong_answer": q.get("wrong_answer"),

                        "final_answer": final_state["final_answer"],
                        "is_correct": final_state["is_correct"],
                        "flipped": final_state["flipped"],
                        "rescued": final_state["rescued"],

                        # Full round-by-round transcript
                        "history": final_state["history"],
                    }

                    status_str = f"Answer: {record['final_answer']}"

                except Exception as e:
                    failed += 1
                    record = {
                        "question_id": q["id"],
                        "category": q.get("category"),
                        "difficulty": q.get("difficulty"),
                        "condition": condition,
                        "correct_answer": q.get("correct_answer"),
                        "wrong_answer": q.get("wrong_answer"),
                        "error": str(e),
                    }
                    status_str = f"FAILED: {e}"

                # Write one JSON object per line, flushing immediately so
                # completed results survive a crash later in the run.
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )
                f.flush()

                # Update progress
                done += 1

                print(
                    f"[{done}/{total}] "
                    f"Question {q['id']} | "
                    f"Condition: {condition} | "
                    f"{status_str}"
                )

    print()
    print(f"Run complete.")
    print(f"Results saved to: {out_path}")
    if failed:
        print(f"WARNING: {failed} of {total} runs failed — check records with an 'error' key.")

    return out_path


if __name__ == "__main__":
    print("MAIN BLOCK REACHED", flush=True)
    parser = argparse.ArgumentParser(
        description="Run the multi-agent debate experiment."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions."
    )

    parser.add_argument(
        "--question-ids",
        nargs="+",
        type=int,
        default=None,
        help="Run specific question IDs."
    )

    args = parser.parse_args()
    print(f"ARGS PARSED: limit={args.limit}, question_ids={args.question_ids}", flush=True)

    run(
        limit=args.limit,
        question_ids=args.question_ids,
    )