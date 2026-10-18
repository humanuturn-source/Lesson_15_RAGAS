import os
import json
import re
from typing import List, Dict, Any
import networkx as nx
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

# --- Configuration Constants ---
DOCS_DIRECTORY = "./documents"
GRAPH_STORE_PATH = "./graphstore.json"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "graphrag_vectors"
LLM_MODEL = "gemma4"
EMBEDDING_MODEL = "nomic-embed-text"  # embedding model


# Initialize ChromaDB persistent storage
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
vector_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

def load_documents(directory: str) -> List[Dict[str, str]]:
    """Reads all markdown and text files from the target directory."""
    documents = []
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[!] Created missing documents directory at '{directory}'. Add your files and re-run.")
        return documents
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".txt", ".md")):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    documents.append({"filename": file, "content": f.read()})
    return documents

def extract_entities_and_relations(text_chunk: str) -> List[Dict[str, str]]:
    """Uses Gemma 4 via Ollama to extract explicit entity-relation-entity triples."""
    prompt = f"""You are a Knowledge Graph Extraction Engine. 
Extract all distinct relationships from the following text fragment.
Return ONLY a valid JSON list of objects containing keys: "source", "target", and "relation".
Do not include commentary, markdown backticks, or intro/outro text.

Example format:
[
  {{"source": "ServiceA", "target": "DatabaseB", "relation": "WRITES_TO"}},
  {{"source": "ComponentC", "target": "ServiceA", "relation": "DEPENDS_ON"}}
]

Text Fragment:
{text_chunk}
"""
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        content = response["message"]["content"].strip()
        # Clean JSON markdown formatting if returned
        content = re.sub(r"^```json\s*", "", content)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        
        extracted = json.loads(content)
        if isinstance(extracted, list):
            return extracted
    except Exception as e:
        print(f"[Warning] Failed graph extraction on chunk: {e}")
    return []

def generate_embedding(text: str) -> List[float]:
    """Generates embedding vector using local Ollama model."""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]

def build_and_persist_index():
    print("[*] Starting document ingest & indexing pipeline...")
    docs = load_documents(DOCS_DIRECTORY)
    if not docs:
        print("[!] No documents found. Ingestion aborted.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    graph = nx.DiGraph()
    
    total_chunks = 0
    for doc_idx, doc in enumerate(docs):
        chunks = text_splitter.split_text(doc["content"])
        print(f"[*] Processing '{doc['filename']}' ({len(chunks)} chunks)...")
        
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
            
            # 1. Vector Store Persistence
            embedding = generate_embedding(chunk)
            vector_collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": doc["filename"], "chunk_index": chunk_idx}]
            )
            
            # 2. Graph Construction
            triples = extract_entities_and_relations(chunk)
            for triple in triples:
                src = triple.get("source", "").strip()
                tgt = triple.get("target", "").strip()
                rel = triple.get("relation", "").strip()
                
                if src and tgt and rel:
                    graph.add_node(src, type="entity")
                    graph.add_node(tgt, type="entity")
                    graph.add_edge(src, tgt, relation=rel, source_chunk=chunk_id)
            
            total_chunks += 1

    # Persist Graph to Disk
    graph_data = nx.node_link_data(graph)
    with open(GRAPH_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)

    print(f"[✓] Pipeline complete! Indexed {total_chunks} chunks across {len(docs)} docs.")
    print(f"[✓] Graph saved to: {GRAPH_STORE_PATH}")
    print(f"[✓] Vector DB saved to: {CHROMA_DB_PATH}")

if __name__ == "__main__":
    build_and_persist_index()
