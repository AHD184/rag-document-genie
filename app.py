# streamlit is the library to create web apps in python
import streamlit as st

# We import two defined functions from rag_utility.py
from rag_utility import extract_text_from_pdf, chunk_text

# requests is the library to make http requests in python
import requests

# Allows us access to cryptographic hashing
import hashlib

# We import embedding + retrieval tools
from embeddings_utility import ollama_embedding, top_k_cosine

# Set browser tab title and icon, cosmetic only
st.set_page_config(page_title="AHD's Document Genie", page_icon="🤖")

# Title and subheader of the web app
st.title("🤖 AHD's Document Genie")
st.subheader("📄 Upload a PDF document (Research papers, notes etc.)")

# Name of the local model running is Ollama
MODEL_NAME = "llama3.2:3b"

# This function hashes a file so that each file we upload has a unique id 
# And don't end up re-extracting the PDF text, re-chunking, re-embedding all the chunks every time we send a message while Streamlit reruns the script
def pdf_hash(uploaded_file) -> str:
    uploaded_file.seek(0)   # Sets the pointer to the start of the file
    file = uploaded_file.read() 
    uploaded_file.seek(0)   # Set the pointer back to the start so it doesn't seem empty the next time we read the file
    return hashlib.sha256(file).hexdigest() # Return a string of hexadecimal digits after hashing

# Returns a list of (chunk_index, similarity_score, chunk_text)
def retrieve_relevant_chunks_embeddings(query: str, k: int = 4):
    if not query or "chunk_embeddings" not in st.session_state:
        return []

    # Embed the user question
    query_vec = ollama_embedding(query, model="nomic-embed-text")

    # Compute cosine similarity against all chunks
    indices, scores = top_k_cosine(query_vec, st.session_state.chunk_embeddings, k=k)

    results = []
    for i, s in zip(indices, scores):  # zip is built-in function that pairs elements from two different data types together
        chunk_text = st.session_state.chunks[int(i)]
        results.append((int(i), float(s), chunk_text))

    return results

def chat_with_ollama(messages):
    # Send convo to Ollama and get one response (not in tokens)
    # messages -> {Roles: system, user, assistant; content: text}, system defines behavior (hidden)
    # Ollama runs a local server at this address
    # We use 'api/chat' endpoint to have a conversation; other types are '/generate' and '/embeddings'

    url = "http://localhost:11434/api/chat"

    # Data we send to Ollama as a dictionary, Ollama needs data in JSON format
    payload = {
        "model": MODEL_NAME,
        "messages": messages,  # Sends the whole convo
        "stream": False  # No streaming (tokens), get one full response
    }

    # Send a query to Ollama local server (in json format) and save the response
    response = requests.post(url, json=payload)
    # Raise an error if something goes wrong
    response.raise_for_status()
    # Convert response from json format to python dictionary
    data = response.json()
    # Extract assistant's content only
    return data["message"]["content"]

# ---------- PDF Upload + Chunking + Embedding Build ----------

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    text = extract_text_from_pdf(uploaded_pdf)
    st.success("PDF loaded ✅")

    doc_id = pdf_hash(uploaded_pdf)

    # Only rebuild if the PDF has changed
    if st.session_state.get("doc_id") != doc_id:
        st.session_state.doc_id = doc_id

        # Create chunks and store them so we don't redo extraction every rerun
        st.session_state.chunks = chunk_text(text, chunk_size=1200, overlap=200)

        # Create embeddings for all chunks at once (batch)
        with st.spinner("Creating embeddings for document..."):
            st.session_state.chunk_embeddings = ollama_embedding(
                st.session_state.chunks,
                model="nomic-embed-text"
            )

        st.success("Chunk embeddings created ✅")

    st.write("Number of chunks: ", len(st.session_state.chunks))

    # An expander which displays the first chunk of the text
    with st.expander("Preview first chunk"):
        st.write(st.session_state.chunks[0][:1200])

else:
    st.info("Upload a PDF to start.")

# ---------- Chat UI (Messages) ----------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant, be clear and concise."
        }
    ]

for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue

    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message to messages list
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show the user message in chat
    with st.chat_message("user"):
        st.write(user_input)

    # Build retrieval context (if PDF exists)
    context_text = ""
    retrieved_chunks = []

    if uploaded_pdf and "chunks" in st.session_state and "chunk_embeddings" in st.session_state:
        retrieved_chunks = retrieve_relevant_chunks_embeddings(user_input, k=4)

        if retrieved_chunks:
            # Join chunk texts into one context string
            context_parts = [text for _, _, text in retrieved_chunks]
            context_text = "\n\n---\n\n".join(context_parts)

    # Get assistant response from Ollama
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):  # Cosmetic spinner while waiting for response
            messages_to_send = [m.copy() for m in st.session_state.messages]

            # IMPORTANT:
            # We replace the last user message content so we don't duplicate user messages
            if context_text:
                messages_to_send[-1]["content"] = (
                    "Use the following document context to answer the question.\n"
                    "If the answer is not in the context, say: \"I don't know based on the document.\" \n\n"
                    f"CONTEXT:\n{context_text}\n\n"
                    f"QUESTION:\n{user_input}"
                )

            reply = chat_with_ollama(messages_to_send)

        st.write(reply)

        # Optional: show sources (looks professional in demo)
        if retrieved_chunks:
            with st.expander("Sources used (top chunks)"):
                for idx, score, chunk in retrieved_chunks:
                    st.write(f"Chunk {idx} | cosine = {score:.3f}")
                    st.write(" ".join(chunk.split()[:60]) + " ...")
                    st.markdown("---")

    # Save assistant response to messages list
    st.session_state.messages.append({"role": "assistant", "content": reply})

# Button to reset the convo
if st.button("Clear Chat"):
    st.session_state.messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant, be clear and concise."
        }
    ]
    st.rerun()


# -------------------------------------------------------------------
# Legacy keyword-based retrieval functions (NOT USED)
# -------------------------------------------------------------------
# These functions were used before implementing embedding retrieval.
# Keeping them here for reference.
# -------------------------------------------------------------------

# import re
# from collections import Counter

# def tokenize(text: str) -> list[str]:
#     return re.findall(r"[a-zA-Z0-9]+", text.lower())

# def keyword_overlap_score(query: str, chunk: str) -> int:
#     q_tokens = tokenize(query)
#     if not q_tokens:
#         return 0
#     q_count = Counter(q_tokens)
#     ch_count = Counter(tokenize(chunk))
#     return sum(min(q_count[t], ch_count[t]) for t in q_count)

# def retrieve_relevant_chunks(query: str, chunks: list[str], k: int = 4):
#     if not query or not chunks:
#         return []
#     scored = []
#     for i, ch in enumerate(chunks):
#         score = keyword_overlap_score(query, ch)
#         if score > 0:
#             scored.append((i, ch, score))
#     scored.sort(key=lambda x: x[2], reverse=True)
#     return scored[:k]