# -*- coding: utf-8 -*-
# @Time       : 2025/7/16
# @Author     : Marverlises
# @File       : llm.py
# @Description: Manages the global LLM instance.

from openai import OpenAI
from llama_cpp import Llama
from typing import Optional
import json
import logging
import tiktoken

logger = logging.getLogger(__name__)


class LLM:
    """
    A wrapper class for Large Language Models, supporting both API-based and local models.
    """

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 lang: str = "English"):
        self.lang = lang
        if model is not None and api_key is not None:
            # Use API-based model (e.g., OpenAI)
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model
            self.is_api = True
        else:
            # Use local model with llama.cpp
            # Note: This requires a pre-downloaded model file.
            self.client = Llama.from_pretrained(
                repo_id="Qwen/Qwen1.5-1.8B-Chat-GGUF",
                filename="*q4_k_m.gguf",
                verbose=False
            )
            self.is_api = False

    def get_tldr(self, title: str, abstract: str, full_text: str) -> str:
        """
        Generates a detailed, structured summary (TLDR) of a paper using its full text.
        """
        prompt = f"""
        As a professional research assistant, your task is to read and understand the following research paper in its entirety. Based on the full text, provide a clear and in-depth summary in {self.lang}.

        **Your summary must be structured into the following core sections:**
        1.  **Core Idea:** What is the central problem the paper addresses? What is its main contribution or innovation?
        2.  **Methodology:** How did the authors approach the problem? Briefly describe the key techniques, architecture, and steps involved in their method.
        3.  **Conclusions:** What are the main findings of the paper? Do the experimental results and evidence presented strongly support the paper's core ideas? What are the key takeaways?

        Please ensure your summary is a comprehensive analysis derived from the full paper text, not just a rephrasing of the abstract.

        ---
        **Paper Title:** {title}

        **Abstract:** {abstract}

        **Full Paper Text:**
        {full_text}
        ---
        """

        return self.generate(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a world-class research assistant, renowned for your ability to distill complex academic papers into structured, insightful summaries in {self.lang}.",
                },
                {"role": "user", "content": prompt},
            ],
        )

    def generate(self, messages: list[dict]) -> str:
        """
        Generates a response from the LLM given a list of messages.
        """
        if self.is_api:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            return response.choices[0].message.content
        else:
            response = self.client.create_chat_completion(messages=messages)
            return response['choices'][0]['message']['content']

    def get_pdf_metadata(self, pdf_text: str) -> dict:
        """Extracts title and abstract from PDF text."""

        prompt = f"""
        As a research assistant, your task is to extract the title and abstract from the following text, which was extracted from the first few pages of a research paper PDF.

        Please return the result in a JSON format with two keys: "title" and "abstract".

        If you cannot find an abstract, return the first few coherent paragraphs of the text as the abstract.
        
        ---
        **PDF Text:**
        {pdf_text[:4000]}
        ---
        """

        metadata_str = self.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at parsing academic paper content and extracting structured metadata. You will only return JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        )

        try:
            return json.loads(metadata_str)
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response from LLM for PDF metadata.")
            return {"title": "Unknown Title", "abstract": "Failed to parse LLM response."}


# Global LLM instance
_llm: Optional[LLM] = None


def set_global_llm(model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None,
                   lang: str = "English"):
    """
    Initializes the global LLM instance.
    """
    global _llm
    _llm = LLM(model, api_key, base_url, lang)


def get_llm() -> LLM:
    """
    Retrieves the global LLM instance.
    """
    if _llm is None:
        raise Exception("LLM has not been initialized. Call set_global_llm() first.")
    return _llm
