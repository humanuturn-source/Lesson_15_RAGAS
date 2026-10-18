import os
import json
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
import networkx as nx
import chromadb
import ollama

# Configuration matching ingest_graph.py
GRAPH_STORE_PATH = "./graphstore.json"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "graphrag_vectors"
EMBEDDING_MODEL = "nomic-embed-text"

mcp = FastMCP("GraphRAG Tools Server")

# 1. Load Persisted NetworkX Graph from JSON
G = nx.DiGraph()
if os.path.exists(GRAPH_STORE_PATH):
    with open(GRAPH_STORE_PATH, "r", encoding="utf-8") as f:
        graph_data = json.load(f)
        G = nx.node_link_graph(graph_data)
    print(f"[*] Loaded Graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
else:
    print(f"[!] Warning: '{GRAPH_STORE_PATH}' not found. Run ingest_graph.py first.")

# 2. Connect to Persisted ChromaDB Vector Store
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    print(f"[*] Connected to ChromaDB collection '{COLLECTION_NAME}' ({collection.count()} chunks).")
except Exception as e:
    collection = None
    print(f"[!] Warning: Could not access ChromaDB collection: {e}")


@mcp.tool()
def vector_hybrid_search(query_embedding: List[float], top_k: int = 3) -> str:
    """Performs vector similarity search against persisted document chunks in ChromaDB using query embeddings."""
    if not collection or collection.count() == 0:
        return "Vector database is empty or not initialized."

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count())
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        if not docs:
            return "No relevant passages found in vector store."

        formatted = []
        for doc, meta in zip(docs, metas):
            source = meta.get("source", "unknown") if meta else "unknown"
            formatted.append(f"-[Source: {source}]\n{doc}")

        return f"Retrieved Context Passages:\n\n" + "\n\n".join(formatted)
    except Exception as e:
        return f"Error executing vector search: {str(e)}"


@mcp.tool()
def graph_entity_search(entity_name: str, max_depth: int = 2) -> str:
    """Retrieves entity connections and relationships from the ingested knowledge graph."""
    if G.number_of_nodes() == 0:
        return "Knowledge graph is empty or not initialized."

    matched_node = None
    for node in G.nodes():
        if entity_name.lower() in str(node).lower():
            matched_node = node
            break

    if not matched_node:
        return f"Entity '{entity_name}' not found in the Knowledge Graph."

    connections = []
    # Outgoing relationships
    for u, v, data in G.out_edges(matched_node, data=True):
        relation = data.get("relation", "CONNECTED_TO")
        connections.append(f"({u}) --[{relation}]--> ({v})")

    # Incoming relationships
    for u, v, data in G.in_edges(matched_node, data=True):
        relation = data.get("relation", "CONNECTED_TO")
        connections.append(f"({u}) --[{relation}]--> ({v})")

    summary = {
        "entity": matched_node,
        "connections": connections
    }
    return json.dumps(summary, indent=2)


@mcp.tool()
def graph_synthesis_query(entity_a: str, entity_b: str) -> str:
    """Finds direct or multi-hop path relationships between two entities in the knowledge graph."""
    if G.number_of_nodes() == 0:
        return "Knowledge graph is empty or not initialized."

    node_a = next((n for n in G.nodes() if entity_a.lower() in str(n).lower()), None)
    node_b = next((n for n in G.nodes() if entity_b.lower() in str(n).lower()), None)

    if not node_a or not node_b:
        return f"Could not locate one or both entities in graph: '{entity_a}', '{entity_b}'."

    try:
        undirected_G = G.to_undirected()
        path = nx.shortest_path(undirected_G, source=node_a, target=node_b)
        path_str = " -> ".join(path)
        return f"Graph Path Found between '{node_a}' and '{node_b}':\n{path_str}"
    except nx.NetworkXNoPath:
        return f"No path relationship exists in graph between '{node_a}' and '{node_b}'."
    except Exception as e:
        return f"Error executing path synthesis: {str(e)}"


if __name__ == "__main__":
    mcp.run()
