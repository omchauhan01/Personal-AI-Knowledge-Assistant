import re
import streamlit as st

from utils import load_documents, split_documents
from RAG_pipeline import create_vector_store, retrieve_context

st.set_page_config(page_title="Personal AI Knowledge Assistant", layout="wide")

st.title("Personal AI Knowledge Assistant")
st.caption("Free semantic document assistant using embeddings and vector search")

st.markdown("""
### What this assistant does
- Upload and process PDF documents
- Split text into searchable chunks
- Convert chunks into semantic embeddings
- Store them in a FAISS vector database
- Retrieve the most relevant passages for each question
- Return cleaner answers from document content
- Show source citations for transparency
""")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "file_count" not in st.session_state:
    st.session_state.file_count = 0


def extract_general_answer(question, docs):
    question_lower = question.lower().strip()
    combined_text = " ".join([doc.page_content for doc in docs])

    # email
    if "email" in question_lower:
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', combined_text)
        if emails:
            return f"The email mentioned is: **{emails[0]}**"

    # phone
    if "phone" in question_lower or "contact" in question_lower:
        phones = re.findall(r'(\+?\d[\d\s\-]{8,}\d)', combined_text)
        if phones:
            return f"The phone number mentioned is: **{phones[0]}**"

    # general extractive answer
    question_words = set(re.findall(r"\w+", question_lower))
    candidate_sentences = []

    for doc in docs:
        sentences = re.split(r'(?<=[.!?])\s+', doc.page_content)

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 25:
                continue

            sentence_words = set(re.findall(r"\w+", sentence.lower()))
            overlap = len(question_words.intersection(sentence_words))

            if overlap > 0:
                candidate_sentences.append((overlap, sentence))

    candidate_sentences.sort(key=lambda x: x[0], reverse=True)

    if not candidate_sentences:
        return "I could not find a clear answer in the uploaded documents."

    selected = []
    seen = set()

    for _, sentence in candidate_sentences:
        clean_sentence = sentence.strip()
        if clean_sentence.lower() not in seen:
            selected.append(clean_sentence)
            seen.add(clean_sentence.lower())

        if len(selected) == 3:
            break

    answer = " ".join(selected)

    if len(answer) > 500:
        answer = answer[:500].rsplit(" ", 1)[0] + "..."

    return answer


with st.sidebar:
    st.header("Settings")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_docs = st.button("Process Documents")
    clear_chat = st.button("Clear Chat")

    st.markdown("### System Status")

    if uploaded_files:
        st.success("Documents uploaded")
    else:
        st.warning("No documents uploaded")

    if st.session_state.vector_store is not None:
        st.success("Knowledge base ready")
    else:
        st.info("Knowledge base not built")

    st.markdown("### Uploaded Documents")

    if uploaded_files:
        st.write(f"Total Files: {len(uploaded_files)}")
        for file in uploaded_files:
            st.write(f"📄 {file.name}")
    else:
        st.write("No documents added yet.")

    st.markdown("### Knowledge Base Stats")
    st.write(f"Files Indexed: {st.session_state.file_count}")
    st.write(f"Chunks Created: {st.session_state.chunk_count}")

if clear_chat:
    st.session_state.chat_history = []
    st.rerun()

if uploaded_files and process_docs:
    with st.spinner("Processing and indexing documents..."):
        documents = load_documents(uploaded_files)
        chunks = split_documents(documents)
        vector_store = create_vector_store(chunks)

        st.session_state.vector_store = vector_store
        st.session_state.chunk_count = len(chunks)
        st.session_state.file_count = len(uploaded_files)

    st.success("Documents processed successfully.")

# show old chat first
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

user_question = st.chat_input("Ask a question about your documents")

if user_question and user_question.strip() and st.session_state.vector_store is not None:
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.spinner("Searching relevant knowledge..."):
        docs, context = retrieve_context(st.session_state.vector_store, user_question, k=4)

        retrieved_chunks = len(docs)
        context_length = len(context)
        avg_chunk_size = context_length // retrieved_chunks if retrieved_chunks else 0

        answer = extract_general_answer(user_question, docs)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

    st.markdown("### Retrieval Stats")
    st.write(f"Chunks Retrieved: {retrieved_chunks}")
    st.write(f"Context Length: {context_length} characters")
    st.write(f"Average Chunk Size: {avg_chunk_size} characters")

    st.markdown("### Supporting Sources")

    for i, doc in enumerate(docs, 1):
        source_file = doc.metadata.get("source_file", "Unknown file")
        page_num = doc.metadata.get("page", "Unknown")

        with st.expander(f"Source {i} | File: {source_file} | Page: {page_num}"):
            st.write(doc.page_content)

elif user_question and st.session_state.vector_store is None:
    st.warning("Please upload and process documents first.")