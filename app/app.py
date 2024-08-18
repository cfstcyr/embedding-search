from io import BytesIO

import pandas as pd
import streamlit as st

from src.embed import embedder_factory_env
from src.extract import extract_file
from src.llm import llm_model_factory_env, llm_search
from src.log import init_logging
from src.search import search

init_logging()

st.title("Document search")

file = st.file_uploader("Upload a document", type="pdf")


def get_file_embedding(
    file, embedder=embedder_factory_env()
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = f"{file.name}_{hash(embedder)}_embedded"

    if file is None:
        return None
    if key in st.session_state:
        return st.session_state[key]

    extracted = extract_file(BytesIO(file.getvalue()), file.name, "pdf")

    embedded = embedder.embed(extracted["combined_text"])

    st.session_state[key] = (extracted, embedded)

    return extracted, embedded


if file:
    embedder = embedder_factory_env()
    data, data_embeddings = get_file_embedding(file, embedder)

    chat = st.chat_input("Ask a question")

    if chat:
        query_embedding = embedder.embed(chat)
        search_results = search(
            embeddings=data_embeddings,
            data=data,
            query_embedding=query_embedding,
            top_k=3,
        )

        llm_model = llm_model_factory_env()
        llm_search_results = llm_search(llm_model, chat, search_results)

        st.write(llm_search_results)
