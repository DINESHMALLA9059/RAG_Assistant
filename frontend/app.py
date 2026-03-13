import streamlit as st
import requests

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Custom styling
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
}

.title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    font-size:18px;
    margin-bottom:30px;
}

.user-msg {
    background:#1f77b4;
    padding:12px;
    border-radius:10px;
    margin-bottom:10px;
}

.bot-msg {
    background:#222;
    padding:15px;
    border-radius:10px;
    margin-bottom:15px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">📚 Research Paper AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask questions based on research papers using RAG</div>', unsafe_allow_html=True)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.text_input("Ask your question")

if st.button("Ask"):

    if query.strip() != "":

        st.session_state.messages.append(("user", query))

        with st.spinner("Searching research papers..."):

            response = requests.get(
                "http://127.0.0.1:8000/ask",
                params={"query": query}
            )

        if response.status_code == 200:
            answer = response.json()["response"]
        else:
            answer = "Backend error."

        st.session_state.messages.append(("bot", answer))


# Display chat
for role, msg in st.session_state.messages:

    if role == "user":
        st.markdown(f'<div class="user-msg">🧑 {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {msg}</div>', unsafe_allow_html=True)


# Sidebar
st.sidebar.title("Project Info")
st.sidebar.write("""
**Research Paper RAG Assistant**

Technologies used:

- Sentence Transformers (Embeddings)
- FAISS (Vector Database)
- Llama3 via Tinyllama
- LangChain
- FastAPI Backend
- Streamlit Frontend
""")