import streamlit as st
from src.RAGModel import RAGModel

def run_queries():
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

    for i, query in enumerate(queries, 1):
        with st.expander(f"Query {i}: {query}", expanded=False):
            response, simplified_prompt, used_chunks, tokens_used, custom_tokens_used ,quality,trust = model.ask(query)
            print(f"Quality: {quality}")
            st.markdown(f"**Response:**\n\n{response}")
            st.markdown(f"**Simplified Prompt:**\n\n```{simplified_prompt}```")

            st.markdown("**Retrieved Chunks:**")
            for chunk in used_chunks:
                st.markdown(f"- **Section:** {chunk['section']}")
                st.markdown(f"  ```{chunk['content'][:300]}...```")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Token Count", tokens_used)
            col2.metric("Custom Token Count", str(custom_tokens_used))
            col3.metric("Trust", str(trust))
            col4.metric("Quality", "Low ❌" if quality["low_quality_answer_stats"].keys() else "Good ✅")


def main():
    st.set_page_config(page_title="RAGModel UI", layout="wide")
    st.title("🎓 GenAI University Assistant")

    if st.button("Run RAGModel on All Queries"):
        run_queries()
    else:
        st.info("Click the button above to run the model on 11 common academic queries.")

if __name__ == "__main__":
    main()
