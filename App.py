from src.RAGModel import RAGModel


def main():
    """Main function to run the GenAI application."""
    model = RAGModel()
    queries = [
        "What are the requirements for passing a semester?",
        "How many ECTS credits do I need to graduate?",
        "What happens if I fail an exam?",
        "Can I retake a course if I fail?",
        "What is the maximum study period allowed?",
        "How is the final grade calculated?",
        "Are there any mandatory internships?",
        "How do I apply for a leave of absence?",
        "What documents are needed for graduation?",
        "Are there penalties for late submissions?",
        "What is the process for appealing a grade?"
    ]
    for i, q in enumerate(queries, 1):
        response, PROMPT_SIMPLIFIED, used_chunks, tokens_used, custom_tokens_used, quality = model.ask(q)
        print(f"\nQuery {i}: {q}")
        print("Response:", response)
        print("Simplified Prompt:", PROMPT_SIMPLIFIED)
        print("Used Chunks (For UI):")
        for chunk in used_chunks:
            print(f"  - Section: {chunk['section']}\n    Content: {chunk['content'][:150]}...")
        print("Quality Info:", quality)
        if quality['is_low_quality']:
            print("[LOW QUALITY ANSWER FLAGGED]")
        print("Tokens Used:", tokens_used)
        print("Custom Tokens Used:", custom_tokens_used)


if __name__ == "__main__":
    main()
