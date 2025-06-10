import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import re
from typing import Dict, Any



UNCERTAIN_WORDS = {'maybe', 'possibly', 'might', 'could', 'not sure', 'uncertain', 'unknown', 'unsure', 'probably'}
STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'do', 'does', 'to', 'of', 'and', 'or', 'with', 'for', 'in', 'on', 'by', 'at', 'from'}

def extract_keywords(text: str) -> set:
    #this function extracts keywords from the answer and removes the unneeded words, keeping the keywords
    return set(re.findall(r'\w+', text.lower())) - STOPWORDS

def is_low_quality_answer(answer: str, question: str, min_tokens: int = 15) -> bool:
    answer_words = answer.strip().split()
    #well if it's empty
    if not answer_words or len(answer_words) <= 1:
        return True
    #if the answer is same as the question
    if answer.strip().lower() == question.strip().lower():
        return True
    #if less than 15 tokens
    if len(answer_words) < min_tokens:
        return True
    #if the answer provided by model does not have keywordss from the question
    if not extract_keywords(question).intersection(extract_keywords(answer)):
        return True
    return False

def score_answer(answer: str, question: str, confidence: float = None) -> Dict[str, Any]:
    answer_clean = answer.strip().lower()
    answer_words = extract_keywords(answer_clean)
    question_keywords = extract_keywords(question)

    keyword_matches = len(question_keywords & answer_words)
    length = len(answer_clean.split())
    uncertain_count = sum(w in answer_clean for w in UNCERTAIN_WORDS)
    #this is calculated +2 for kewords matching || +1 for length (capped at 30) || -2 for uncertain words
    score = keyword_matches * 2 + min(length, 30) - uncertain_count * 2

    if confidence is not None:
        score += int(confidence * 10)

    return {
        'score': score,
        'keyword_matches': keyword_matches,
        'length': length,
        'uncertain_count': uncertain_count,
        'confidence': confidence,
        'is_low_quality': is_low_quality_answer(answer, question)
    }

class RAGModel:
    def __init__(self, EMBEDDING_MODEL='models/embedding-001', GENERATION_MODEL='gemini-1.5-flash-8b'):
        self.authorize()
        self.EMBEDDING_MODEL = EMBEDDING_MODEL
        self.GENERATION_MODEL = GENERATION_MODEL
        self.READ_FROM_FILE = False
        self.df = pd.read_csv("data/parsed_sections.csv")

        if not self.READ_FROM_FILE:
            self.df['Embedding'] = [
                self.embed_document(row.Title, row.Content) for row in self.df.itertuples(index=False)
            ]
            self.df.to_pickle("embeddings.pkl")
        else:
            self.df = pd.read_pickle("embeddings.pkl")

    def authorize(self):
        """Load environment variables from .env file."""
        load_dotenv()
        api_key = os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)

    def embed_document(self, title, content):
        response = genai.embed_content(
            model=self.EMBEDDING_MODEL,
            content=content,
            task_type="retrieval_document",
            title=title
        )
        return response["embedding"]

    def ask(self, query):
        """Generate a response to the query using the Generative AI model, with proper citation support."""

        # 1. Get embedding for the query
        request = genai.embed_content(
            model=self.EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        question_embedding = request["embedding"]

        # 2. Compute similarity with document embeddings
        dot_products = np.dot(np.stack(self.df['Embedding']), question_embedding)
        top_indices = np.argsort(dot_products)[-3:][::-1]
        top_passages_info = self.df.iloc[top_indices][['Title', 'Content']]

        # 3. Format passages with metadata (section title)
        formatted_passages = [
            f"({row.Title}) {row.Content}" for _, row in top_passages_info.iterrows()
        ]

        # 4. Prepare prompt with instruction to cite section titles
        PROMPT = f"""You are a helpful and informative AGH bot that answers questions using the reference passages below.
    Use only the relevant information, and always cite the section title when answering (e.g., "According to §3(2) - Enrollment Rules...").

    QUESTION: {query}

    PASSAGES:
    {chr(10).join(f'- {p}' for p in formatted_passages)}

    ANSWER:"""

        PROMPT_SIMPLIFIED = f"""You are a helpful and informative AGH bot that answers questions using the reference passages below.
    Cite section titles in your answer for transparency.

    QUESTION: {query}

    PASSAGES:
    {chr(10).join(f'- {row.Title}' for _, row in top_passages_info.iterrows())}

    ANSWER:"""

        # 5. Generate answer from the model
        model = genai.GenerativeModel(self.GENERATION_MODEL)
        response = model.generate_content(PROMPT)
        answer_text = response.text
        confidence = getattr(response, 'safety_ratings', None)

        tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else "N/A"

        # 6. Quality scoring
        quality = score_answer(answer_text, query, confidence=None)

        # 7. Prepare metadata for UI highlighting (5.4)
        used_chunks = [
            {'section': row.Title, 'content': row.Content}
            for _, row in top_passages_info.iterrows()
        ]

        return answer_text, PROMPT_SIMPLIFIED, used_chunks, tokens_used, quality


