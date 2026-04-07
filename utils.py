import os
import re
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


def clean_text(text):
    if not text:
        return ""

    # Fix broken spaced letters like: B a c h e l o r
    text = re.sub(
        r'\b(?:[A-Za-z]\s){2,}[A-Za-z]\b',
        lambda m: m.group(0).replace(" ", ""),
        text
    )

    # Normalize multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_documents(uploaded_files):
    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["source_file"] = uploaded_file.name

        documents.extend(docs)
        os.remove(temp_path)

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)