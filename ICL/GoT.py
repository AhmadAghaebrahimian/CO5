# -*- coding: utf-8 -*-
"""
Task Graph-of-Thought (GoT) reasoning using Ollama HTTP API.
- Explores reasoning as a network of connected ideas.
- Each node = idea; each edge = relationship between ideas.

Created on Fri November 28 10:14:29 2025

@author: agha
"""



import os, json, requests
from dotenv import load_dotenv


load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
MODEL = os.getenv("OLLAMA_MODEL")
HEADERS = {"Content-Type": "application/json"}

# Step 1: Generate key nodes (ideas)
NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "nodes": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4
        }
    },
    "required": ["nodes"]
}

# Step 2: Generate relationships (edges)
EDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                    "relationship": {"type": "string"}
                },
                "required": ["source", "target", "relationship"]
            }
        }
    },
    "required": ["edges"]
}

# Step 3: Generate final synthesis
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "insight_summary": {"type": "string"},
        "key_clusters": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["insight_summary"]
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


def extract_json(resp):
    """Try to safely parse the structured JSON output."""
    content = resp.get("message", {}).get("content", "")
    try:
        return json.loads(content)
    except Exception:
        return {}


def generate_nodes(question):
    system_prompt = (
        "You are a reasoning assistant using the Graph-of-Thought method.\n"
        "Generate 4–6 concise key ideas (nodes) relevant to the user's question.\n"
        "Each node should be a short noun phrase or key concept."
    )
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    r = ollama_chat(msgs, NODE_SCHEMA)
    return extract_json(r).get("nodes", [])


def generate_edges(nodes):
    node_list = ", ".join(nodes)
    prompt = (
        f"Here are reasoning nodes: {node_list}.\n"
        "Describe how they are related. Create directed edges with a short relationship label "
        "(e.g. 'influences', 'supports', 'contradicts')."
    )
    msgs = [
        {"role": "system", "content": "You are linking reasoning nodes safely."},
        {"role": "user", "content": prompt}
    ]
    r = ollama_chat(msgs, EDGE_SCHEMA)
    return extract_json(r).get("edges", [])


def summarize_graph(nodes, edges, question):
    node_text = ", ".join(nodes)
    edge_text = "\n".join(
        [f"{e['source']} → {e['relationship']} → {e['target']}" for e in edges]
    )
    prompt = (
        f"Based on this reasoning graph of ideas and their relationships, write a short final insight.\n"
        f"Question: {question}\n\n"
        f"Nodes: {node_text}\nEdges:\n{edge_text}\n\n"
        "Provide a 3–5 sentence 'insight_summary' and 2–3 'key_clusters' (themes)."
    )
    msgs = [
        {"role": "system", "content": "You are synthesizing graph-based reasoning."},
        {"role": "user", "content": prompt}
    ]
    r = ollama_chat(msgs, SUMMARY_SCHEMA)
    return extract_json(r)


def graph_of_thought(question):
    print("Generating key reasoning nodes...")
    nodes = generate_nodes(question)
    print("Nodes:", nodes)

    print("\nCreating relationships between ideas...")
    edges = generate_edges(nodes)
    for e in edges:
        print(f"  - {e['source']} → {e['relationship']} → {e['target']}")

    print("\nSummarizing reasoning graph...")
    summary = summarize_graph(nodes, edges, question)
    return {"nodes": nodes, "edges": edges, "summary": summary}


def pretty_print(result):
    print("\n=== GRAPH OF THOUGHT ===\n")
    print("Nodes:")
    for n in result["nodes"]:
        print(f" - {n}")
    print("\nEdges:")
    for e in result["edges"]:
        print(f" - {e['source']} → {e['relationship']} → {e['target']}")
    print("\n=== INSIGHT SUMMARY ===\n")
    print(result["summary"].get("insight_summary", ""))
    clusters = result["summary"].get("key_clusters", [])
    if clusters:
        print("\nKey clusters:")
        for c in clusters:
            print(" -", c)


if __name__ == "__main__":
    import sys
    q = input("Enter a complex question: ")
    result = graph_of_thought(q)
    pretty_print(result)
"How could a mid-sized coastal city adapt to rising sea levels sustainably with minimal budget?"