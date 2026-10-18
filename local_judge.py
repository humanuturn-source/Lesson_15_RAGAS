# local_judge.py
import json
import re
import ollama
from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

class LocalOllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "gemma4"):
        self.model_name = model_name

    def load_model(self):
        return self.model_name

    def _extract_json(self, text: str) -> str:
        """Strips markdown code fences and extracts raw JSON string."""
        clean_text = re.sub(r"```(?:json)?", "", text).strip("` \n\r\t")
        start = clean_text.find("{")
        end = clean_text.rfind("}")
        if start != -1 and end != -1:
            return clean_text[start:end+1]
        return clean_text

    def generate(self, prompt: str) -> str:
        system_instruction = (
            "You are a strict, precise evaluation judge. "
            "You MUST respond ONLY with a raw, valid JSON object matching the requested schema. "
            "Do NOT include any markdown formatting, code fences, explanations, or text outside the JSON."
        )
        
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            format="json",
            options={
                "temperature": 0.0,
                "num_ctx": 8192
            }
        )
        
        raw_output = response["message"]["content"]
        return self._extract_json(raw_output)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Ollama-{self.model_name}"

def get_ragas_local_judge(model_name: str = "gemma4"):
    # Configure ChatOllama with explicit JSON formatting where supported
    llm = ChatOllama(
        model=model_name, 
        temperature=0.0, 
        format="json",
        num_ctx=8192
    )
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    return evaluator_llm, evaluator_embeddings
