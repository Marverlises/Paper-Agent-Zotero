# -*- coding: utf-8 -*-
# @Time       : 2025/7/16
# @Author     : Marverlises
# @File       : recommender.py
# @Description: Reranks papers based on Zotero corpus similarity.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from loguru import logger
from src.paper import ArxivPaper

def rerank_paper(papers:list[ArxivPaper], corpus:list[dict], pref_source:str) -> list[ArxivPaper]:
    """
    Reranks a list of papers based on their similarity to a corpus of Zotero papers.
    Papers added more recently to Zotero are given higher weight.
    """
    if len(corpus) == 0:
        logger.warning("Zotero corpus is empty. Cannot rerank papers. Returning original list.")
        return papers
        
    logger.info("Initializing Sentence Transformer model for embeddings...")
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1', device='cpu')

    logger.info("Embedding Zotero corpus...")
    corpus_abstracts = [item['data']['abstractNote'] for item in corpus]
    corpus_embeddings = model.encode(corpus_abstracts, show_progress_bar=True, normalize_embeddings=True)
    
    # Calculate weights for corpus papers based on recency
    # The more recent the paper, the higher the weight.
    if pref_source == 'zotero':
        corpus_dates = [item['data']['dateAdded'] for item in corpus]
        corpus_dates = np.array([np.datetime64(d) for d in corpus_dates])
        # Normalize dates to a [0, 1] range for weighting
        time_diffs = (corpus_dates - corpus_dates.min())
        if (corpus_dates.max() - corpus_dates.min()).astype(int) == 0:
            weights = np.ones_like(time_diffs, dtype=float) # All papers added at same time
        else:
            weights = time_diffs / (corpus_dates.max() - corpus_dates.min())
        weights = np.exp(weights - weights.max()) # Exponential weighting
    else:
        # For local corpus, assign equal weights
        weights = np.ones(len(corpus_embeddings), dtype=float)
    
    logger.info("Embedding new arXiv papers...")
    paper_abstracts = [paper.summary for paper in papers]
    paper_embeddings = model.encode(paper_abstracts, show_progress_bar=True, normalize_embeddings=True)
    
    logger.info("Calculating similarity scores...")
    similarity_matrix = cosine_similarity(paper_embeddings, corpus_embeddings)
    
    # Apply weights to similarity scores
    weighted_similarity = similarity_matrix * weights
    
    # Calculate final score for each paper
    scores = np.mean(weighted_similarity, axis=1)
    
    for paper, score in zip(papers, scores):
        paper.score = score
        
    # Sort papers by score in descending order
    papers.sort(key=lambda p: p.score, reverse=True)
    
    return papers