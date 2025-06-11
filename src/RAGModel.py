import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
import re
from typing import Dict, Any



UNCERTAIN_WORDS = {'maybe', 'possibly', 'might', 'could', 'not sure', 'uncertain', 'unknown', 'unsure', 'probably'}
STOPWORDS = {'the', 'a', 'an', 'is', 'are', 'do', 'does', 'to', 'of', 'and', 'or', 'with', 'for', 'in', 'on', 'by',
             'at', 'from'}


def extract_keywords(text: str) -> set:
    # this function extracts keywords from the answer and removes the unneeded words, keeping the keywords
    return set(re.findall(r'\w+', text.lower())) - STOPWORDS


def has_uncertain_phrases_regex(text: str, threshold: int = 2) -> bool:
    text_lower = text.lower()
    count = 0
    for word in UNCERTAIN_WORDS:
        matches = re.findall(rf'\b{word}\b', text_lower)
        count += len(matches)
    return count >= threshold


def extract_section_numbers(text: str) -> set:
    matches = re.findall(r'§\s*(\d+)', text)
    return set(matches)


def get_valid_section_numbers(valid_titles: list[str]) -> set:
    section_numbers = set()
    for title in valid_titles:
        m = re.search(r'§\s*(\d+)', title)
        if m:
            section_numbers.add(m.group(1))
    return section_numbers


def has_invalid_sections(text: str, valid_titles: list[str]) -> bool:
    cited_numbers = extract_section_numbers(text)
    valid_numbers = get_valid_section_numbers(valid_titles)
    # Return True if any cited number is NOT in valid numbers
    return bool(cited_numbers - valid_numbers)


def is_low_quality_answer(answer: str, question: str, valid_titles: [], min_tokens: int = 15) -> (bool, dict):
    low_quality_answer_stats = {}
    answer_words = answer.strip().split()
    # well if it's empty
    if not answer_words or len(answer_words) <= 1:
        low_quality_answer_stats['empty_answers'] = 1
        return True, low_quality_answer_stats
    # if the answer is same as the question
    if answer.strip().lower() == question.strip().lower():
        low_quality_answer_stats['same_as_question'] = 1
        return True, low_quality_answer_stats
    # if less than 15 tokens
    if len(answer_words) < min_tokens:
        low_quality_answer_stats['too_short'] = 1
        return True, low_quality_answer_stats
    # if the answer provided by model does not have keywordss from the question
    if not extract_keywords(question).intersection(extract_keywords(answer)):
        low_quality_answer_stats['no_keywords'] = 1
        return True, low_quality_answer_stats
    if has_invalid_sections(answer, valid_titles):
        low_quality_answer_stats['invalid_sections'] = 1
        return True, low_quality_answer_stats
    if has_uncertain_phrases_regex(answer):
        low_quality_answer_stats['uncertain_phrases'] = 1
        return True, low_quality_answer_stats
    return False, {}


def score_answer(answer: str, question: str, valid_titles: [], confidence: float = None) -> Dict[str, Any]:
    answer_clean = answer.strip().lower()
    answer_words = extract_keywords(answer_clean)
    question_keywords = extract_keywords(question)

    keyword_matches = len(question_keywords & answer_words)
    length = len(answer_clean.split())
    uncertain_count = sum(w in answer_clean for w in UNCERTAIN_WORDS)
    # this is calculated +2 for kewords matching || +1 for length (capped at 30) || -2 for uncertain words
    score = keyword_matches * 2 + min(length, 30) - uncertain_count * 2

    if confidence is not None:
        score += int(confidence * 10)

    return {
        'score': score,
        'keyword_matches': keyword_matches,
        'length': length,
        'uncertain_count': uncertain_count,
        'confidence': confidence,
        'is_low_quality': is_low_quality_answer(answer, question, valid_titles)[0],
        'low_quality_answer_stats': is_low_quality_answer(answer, question, valid_titles)[1],
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

    def get_tokens_used(self, response):
        words = response.split()
        word_count = len(words)
        char_count = len(response)
        keywords = len(extract_keywords(response))

        return {"words": word_count * 1.5, "characters": char_count // 3, "keywords": keywords * 3}

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
        sorted_indices = np.argsort(dot_products)[::-1]

        # 2a. Apply similarity drop detection (Delta Cutoff)
        delta_cutoff_ratio = 0.90  # Allow 10% drop from top score
        top_score = dot_products[sorted_indices[0]]

        top_indices = []
        for idx in sorted_indices:
            score = dot_products[idx]
            if score < top_score * delta_cutoff_ratio:
                break
            top_indices.append(idx)

        top_indices = top_indices[:5]
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

        custom_tokens_used = self.get_tokens_used(PROMPT + answer_text)
        tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else "N/A"

        # 6. Quality scoring
        quality = score_answer(answer_text, query, self.df['Title'].tolist(), confidence=None)

        # 7. Prepare metadata for UI highlighting (5.4)
        used_chunks = [
            {'section': row.Title, 'content': row.Content}
            for _, row in top_passages_info.iterrows()
        ]

        return answer_text, PROMPT_SIMPLIFIED, used_chunks, tokens_used, custom_tokens_used, quality
