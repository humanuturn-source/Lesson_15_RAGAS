import base64
import os
import tempfile
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel

from agent_graphrag import GraphRAGHelpDesk

app = FastAPI(title="Local GraphRAG Agent Server")
graphrag_agent = GraphRAGHelpDesk(reasoning_model="gemma4", fast_model="gemma:2b")


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]


def extract_prompt_and_image(messages: List[ChatMessage]):
    user_query = ""
    temp_image_path = None

    last_user_msg = next((m for m in reversed(messages) if m.role == "user"), None)
    if not last_user_msg:
        return "", None

    if isinstance(last_user_msg.content, str):
        user_query = last_user_msg.content
    elif isinstance(last_user_msg.content, list):
        for part in last_user_msg.content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    user_query += part.get("text", "") + " "
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:image"):
                        header, encoded = image_url.split(",", 1)
                        ext = header.split(";")[0].split("/")[-1]
                        image_data = base64.b64decode(encoded)

                        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
                        temp_file.write(image_data)
                        temp_file.close()
                        temp_image_path = temp_file.name

    return user_query.strip(), temp_image_path


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "graphrag-router-agent", "object": "model", "owned_by": "local"}
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    user_query, image_path = extract_prompt_and_image(request.messages)
    response_text = graphrag_agent.process_query(user_query, image_path=image_path)

    if image_path and os.path.exists(image_path):
        os.remove(image_path)

    return {
        "id": "chatcmpl-agent",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
