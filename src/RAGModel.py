import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai


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
        """Generate a response to the query using the Generative AI model."""

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
        top_passages = self.df.iloc[top_indices]['Content'].tolist()
        top_passages_titles = self.df.iloc[top_indices]['Title'].tolist()

        # 3. Prepare the prompt
        PROMPT = f"""You are a helpful and informative AGH bot that answers questions using the reference passages below.
        If the passages are not relevant to the question, you may ignore them.
    
        QUESTION: {query}
        PASSAGES:
        {chr(10).join(f'- {p}' for p in top_passages)}
    
        ANSWER:"""

        PROMPT_SIMPLIFIED = f"""You are a helpful and informative AGH bot that answers questions using the reference passages below.
        If the passages are not relevant to the question, you may ignore them.
    
        QUESTION: {query}
        PASSAGES:
        {chr(10).join(f'- {p}' for p in top_passages_titles)}
    
        ANSWER:"""

        # 4. Generate answer from the model
        model = genai.GenerativeModel(self.GENERATION_MODEL)
        response = model.generate_content(PROMPT)

        # 5. Extract token usage (requires safety + usage config)
        tokens_used = response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else "N/A"

        return response.text, PROMPT_SIMPLIFIED, top_passages_titles, tokens_used

