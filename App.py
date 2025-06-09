from src.RAGModel import RAGModel


def main():
    """Main function to run the GenAI application."""
    model = RAGModel()
    response, PROMPT_SIMPLIFIED, top_passages_titles, tokens_used = model.ask("What do I have to do to be awarded with a distinction while graduating?")

    print("Response:", response)
    print("Simplified Prompt:", PROMPT_SIMPLIFIED)
    print("Top Passages Titles:", top_passages_titles)
    print("Tokens Used:", tokens_used)


if __name__ == "__main__":
    main()
