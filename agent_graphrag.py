import os
import sys
import logging
import json
import re
import time
from typing import Dict, Any, List, Optional, Tuple
import ollama

from mcp_graphrag import (
    vector_hybrid_search,
    graph_entity_search,
    graph_synthesis_query
)

root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(message)s"))
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)

logger = logging.getLogger("GraphRAG_Agent")


class MCPClientRegistry:
    def __init__(self):
        self.tools = {
            "vector_hybrid_search": vector_hybrid_search,
            "graph_entity_search": graph_entity_search,
            "graph_synthesis_query": graph_synthesis_query
        }

    def execute_mcp_tool(self, tool_name: str, **kwargs) -> Any:
        log_args = kwargs.copy()
        if "query_embedding" in log_args:
            emb = log_args["query_embedding"]
            log_args["query_embedding"] = f"[Vector embedding omitted: {len(emb)} dimensions]"
        
        logger.info(f"[MCP CLIENT] Executing tool '{tool_name}' with args: {log_args}")
        if tool_name in self.tools:
            try:
                return self.tools[tool_name](**kwargs)
            except Exception as e:
                logger.error(f"[MCP TOOL ERROR] {tool_name} failed: {str(e)}")
                return f"MCP Tool execution error: {str(e)}"
        return f"Tool '{tool_name}' not found."


class GraphRAGHelpDesk:
    def __init__(
        self, 
        reasoning_model: str = "gemma4", 
        fast_model: str = "gemma:2b"
    ):
        self.reasoning_model = reasoning_model
        self.fast_model = fast_model
        self.mcp_client = MCPClientRegistry()

    def _call_llm(
        self, 
        system_prompt: str, 
        user_prompt: str, 
        image_path: Optional[str] = None, 
        tier: str = "reasoning"
    ) -> str:
        target_model = self.reasoning_model if (image_path or tier == "reasoning") else self.fast_model
        start_time = time.time()

        try:
            message_payload = {"role": "user", "content": f"System Task: {system_prompt}\nUser Query: {user_prompt}"}
            
            if image_path and os.path.exists(image_path):
                message_payload["images"] = [image_path]
    
            response = ollama.chat(                
                model=target_model,
                messages=[message_payload],
                options={"temperature": 0.0}
            )
            elapsed = time.time() - start_time
            content = response["message"]["content"].strip()
            logger.info(f"[ROUTER METRICS] Tier: '{tier}' | Model: '{target_model}' | Latency: {elapsed:.2f}s")
            return content

        except Exception as e:
            logger.error(f"Ollama execution error on model '{target_model}': {str(e)}")
            if tier == "reasoning":
                logger.warning(f"[FAILOVER TRIGGERED] Falling back to fast model '{self.fast_model}'...")
                return self._call_llm(system_prompt, user_prompt, image_path=None, tier="fast")
            return f"Error executing local model ({target_model}): {str(e)}"

    def agent_intent_router(self, query: str, image_path: Optional[str] = None) -> str:
        logger.info("-> AGENT 1 [Dynamic Router]: Evaluating GraphRAG intent...")

        system_prompt = (
            "You are an Intent Classification Agent for a Knowledge Graph System.\n"
            "Categorize the user query into EXACTLY ONE keyword:\n\n"
            "Categories: VECTOR_SEARCH, ENTITY_LOOKUP, PATH_SYNTHESIS, GENERAL\n"
            "Return ONLY the single category keyword."
        )

        intent = self._call_llm(system_prompt, query, tier="fast").upper()
        
        for valid in ["VECTOR_SEARCH", "ENTITY_LOOKUP", "PATH_SYNTHESIS", "GENERAL"]:
            if valid in intent:
                return valid
        return "GENERAL"

    def process_query_with_context(self, user_query: str, image_path: Optional[str] = None) -> Tuple[str, List[str]]:
        logger.info("=" * 70)
        logger.info(f"USER QUERY: \"{user_query}\"")
        logger.info("=" * 70)

        intent = self.agent_intent_router(user_query, image_path=image_path)

        if intent == "VECTOR_SEARCH":
            try:
                emb_res = ollama.embeddings(model="nomic-embed-text", prompt=user_query)
                query_embedding = emb_res["embedding"]
            except Exception as e:
                return f"Embedding error: {e}", ["No context"]

            collection = self.mcp_client.tools["vector_hybrid_search"].__globals__.get("collection")
            if collection:
                res = collection.query(query_embeddings=[query_embedding], n_results=3)
                raw_chunks = res.get("documents", [[]])[0]
            else:
                raw_chunks = [str(self.mcp_client.execute_mcp_tool("vector_hybrid_search", query_embedding=query_embedding, top_k=3))]

            context_block = "\n\n".join(raw_chunks)
            
            # REFINED SYSTEM PROMPT FOR MAXIMUM FAITHFULNESS & RELEVANCY
            system_prompt = (
                "You are a precise scientific research assistant answering questions using ONLY the provided context passages.\n"
                "STRICT COMPLIANCE RULES:\n"
                "1. Answer the query directly, completely, and comprehensively using facts explicitly contained in the context.\n"
                "2. DO NOT include introductory conversational filler (e.g., 'Based on the provided context...').\n"
                "3. DO NOT introduce external knowledge, assumptions, or extrapolation.\n"
                "4. If the exact answer cannot be found in the text, output: 'I cannot find the answer in the provided documents.'\n\n"
                f"Context Passages:\n{context_block}"
            )
            ans = self._call_llm(system_prompt, user_query, tier="reasoning")
            return ans, raw_chunks

        elif intent == "ENTITY_LOOKUP":
            extract_prompt = "Extract the main entity name as JSON: {\"entity_name\": \"Entity\"}"
            raw_output = self._call_llm(extract_prompt, user_query, tier="fast")
            try:
                entity = json.loads(raw_output.replace("```json", "").replace("```", "").strip()).get("entity_name", user_query)
            except Exception:
                entity = user_query

            mcp_data = str(self.mcp_client.execute_mcp_tool("graph_entity_search", entity_name=entity, max_depth=2))
            system_prompt = (
                "You are a Knowledge Graph Entity Specialist.\n"
                "Synthesize a direct, thorough response using strictly the provided graph data network. Avoid introductory meta-text.\n\n"
                f"Graph Data:\n{mcp_data}"
            )
            ans = self._call_llm(system_prompt, user_query, tier="reasoning")
            return ans, [mcp_data]

        elif intent == "PATH_SYNTHESIS":
            extract_prompt = "Extract entity pair as JSON: {\"entity_a\": \"First\", \"entity_b\": \"Second\"}"
            raw_output = self._call_llm(extract_prompt, user_query, tier="fast")
            try:
                data = json.loads(raw_output.replace("```json", "").replace("```", "").strip())
                entity_a, entity_b = data.get("entity_a", ""), data.get("entity_b", "")
            except Exception:
                entity_a, entity_b = "", ""

            mcp_data = str(self.mcp_client.execute_mcp_tool("graph_synthesis_query", entity_a=entity_a, entity_b=entity_b))
            system_prompt = (
                "You are a Knowledge Graph Path Specialist.\n"
                "Explain the relationship network linking these entities using strictly the provided graph path. Direct answer only.\n\n"
                f"Graph Path:\n{mcp_data}"
            )
            ans = self._call_llm(system_prompt, user_query, tier="reasoning")
            return ans, [mcp_data]

        else:
            ans = self._call_llm("You are a helpful assistant. Give a comprehensive, direct answer.", user_query, tier="fast")
            return ans, [user_query]

    def process_query(self, user_query: str, image_path: Optional[str] = None) -> str:
        ans, _ = self.process_query_with_context(user_query, image_path=image_path)
        return ans
