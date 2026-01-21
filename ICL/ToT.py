# -*- coding: utf-8 -*-
"""
Task Multi-turn Tree-of-Thought reasoning using Ollama HTTP API.
- Generates branches as different thoughts
- Shows summarized branches and how they evolve.

Created on Fri November 28 12:02:36 2025

@author: agha
"""



import os, json, requests, time
from dotenv import load_dotenv


load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
MODEL = os.getenv("OLLAMA_MODEL")

HEADERS = {"Content-Type": "application/json"}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_branches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "supporting_points": {"type": "array", "items": {"type": "string"}},
                    "evaluation": {"type": "string"},
                    "score": {"type": "number"}
                },
                "required": ["hypothesis", "evaluation"]
            }
        }
    },
    "required": ["thought_branches"]
}

FINAL_SCHEMA = {
    "type": "object",
    "properties": {
        "final_conclusion": {"type": "string"},
        "summary_reasoning": {"type": "string"}
    },
    "required": ["final_conclusion"]
}


def ollama_chat(messages, format_schema=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False
    }
    if format_schema:
        payload["format"] = format_schema
    resp = requests.post(OLLAMA_HOST, headers=HEADERS, json=payload, timeout=180)
    resp.raise_for_status()
    return resp.json()


def parse_structured_content(resp_json):
    content = None
    if "message" in resp_json and isinstance(resp_json["message"], dict):
        content = resp_json["message"].get("content")
    elif "response" in resp_json:
        content = resp_json["response"]
    try:
        return json.loads(content)
    except Exception:
        return {"thought_branches": []}


def generate_branches(question, round_num=1):
    system_prompt = (
        "You are a reasoning assistant using a Tree-of-Thought method.\n"
        "Generate 3 concise branches (hypotheses) for answering the question below.\n"
        "Each branch must include: hypothesis, 1–3 supporting points, an evaluation, and a numeric score (1–10).\n"
        "Use clear, summarized reasoning only.\n"
        f"This is Round {round_num} of exploration."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    resp = ollama_chat(messages, format_schema=RESPONSE_SCHEMA)
    return parse_structured_content(resp)


def prune_branches(branches, keep_top_n=2):
    sorted_b = sorted(branches, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_b[:keep_top_n]


def refine_branches(question, top_branches, round_num=2):
    summaries = "\n".join(
        [f"- {b['hypothesis']} (Eval: {b.get('evaluation')}, Score: {b.get('score', 0)})"
         for b in top_branches]
    )
    prompt = (
        f"We are continuing Tree-of-Thought reasoning, Round {round_num}.\n"
        f"Here are the current top hypotheses:\n{summaries}\n\n"
        "Propose 2 refined or combined hypotheses that improve on them.\n"
        "Return in the same JSON format."
    )
    messages = [
        {"role": "system", "content": "You are refining reasoning safely."},
        {"role": "user", "content": prompt}
    ]
    resp = ollama_chat(messages, format_schema=RESPONSE_SCHEMA)
    return parse_structured_content(resp)


def conclude_answer(question, best_branches):
    summary_text = "\n".join(
        [f"- {b['hypothesis']} (Eval: {b['evaluation']})" for b in best_branches]
    )
    prompt = (
        f"Based on the refined hypotheses below, write a short FINAL CONCLUSION.\n"
        f"Include a 2–3 sentence summary of reasoning.\n"
        f"{summary_text}"
    )
    messages = [
        {"role": "system", "content": "You are forming the final conclusion safely."},
        {"role": "user", "content": prompt}
    ]
    resp = ollama_chat(messages, format_schema=FINAL_SCHEMA)
    content = resp.get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except Exception:
        return {"final_conclusion": content, "summary_reasoning": ""}


def tree_of_thought(question):
    print("Generating initial thought branches...")
    data1 = generate_branches(question, round_num=1)
    branches1 = data1.get("thought_branches", [])
    for b in branches1:
        print(f"  - {b['hypothesis']} (Score {b.get('score', '?')})")

    top1 = prune_branches(branches1)
    print("\nRefining top branches...")
    data2 = refine_branches(question, top1, round_num=2)
    branches2 = data2.get("thought_branches", [])
    for b in branches2:
        print(f"  - {b['hypothesis']} (Score {b.get('score', '?')})")

    combined = prune_branches(branches1 + branches2)
    print("\nConcluding final answer...")
    conclusion = conclude_answer(question, combined)

    return {
        "round1_branches": branches1,
        "round2_branches": branches2,
        "final": conclusion
    }


def pretty_print(result):
    print("\n=== TREE OF THOUGHT SUMMARY ===\n")
    for i, b in enumerate(result["round1_branches"], 1):
        print(f"R1.{i}) {b['hypothesis']} (Score {b.get('score', '?')}) - {b['evaluation']}")
    print("\n--- Refinements ---\n")
    for i, b in enumerate(result["round2_branches"], 1):
        print(f"R2.{i}) {b['hypothesis']} (Score {b.get('score', '?')}) - {b['evaluation']}")
    print("\n=== FINAL CONCLUSION ===\n")
    print(result["final"].get("final_conclusion", ""))
    print("\nSummary:", result["final"].get("summary_reasoning", ""))


if __name__ == "__main__":
    q = input("Enter a complex question: ").strip()
    result = tree_of_thought(q)
    pretty_print(result)

"How could a mid-sized coastal city adapt to rising sea levels sustainably with minimal budget?"