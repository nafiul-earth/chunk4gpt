import streamlit as st
import subprocess, sys
import yaml
from rag_backend import retrieve, build_prompt, generate_answer

st.set_page_config(page_title="Local RAG Chat", layout="wide")
st.title("💬 Local RAG Chat — llama3.2 + bge-m3")

# Sidebar: build or rebuild the index
with st.sidebar:
    st.header("Index")
    if st.button("Build or Rebuild"):
        with st.spinner("Ingesting and indexing…"):
            out = subprocess.run([sys.executable, "ingest.py"], capture_output=True, text=True)
            st.code(out.stdout or out.stderr)

    CFG = yaml.safe_load(open("config.yaml"))
    st.caption(f"LLM: {CFG['llm_model']}  |  Embeddings: {CFG['embed_model']}")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "citations" in msg and msg["citations"]:
            with st.expander("Citations"):
                for c in msg["citations"]:
                    st.markdown(f"- **{c['source']}** · score {c['score']:.3f}")

# Input
user_input = st.chat_input("Ask a question about your documents and CSVs")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve
    with st.spinner("Searching your knowledge…"):
        hits = retrieve(user_input)

    # Build prompt and answer
    prompt = build_prompt(user_input, hits)
    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = generate_answer(prompt, stream=True)
        placeholder.markdown(answer)
        # Store with citations
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": [{"source": h["source"], "score": h["score"]} for h in hits]
        })
