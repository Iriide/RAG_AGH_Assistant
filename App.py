from src.RAGModel import RAGModel


def main():
    """Main function to run the GenAI application."""
    model = RAGModel()
    model.ask("How many times can I repeat a semester?")


if __name__ == "__main__":
    main()
