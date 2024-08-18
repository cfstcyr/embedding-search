import logging

import numpy as np
import pandas as pd
from codetiming import Timer
from sentence_transformers.util import semantic_search

logger = logging.getLogger(__name__)


@Timer(
    text="Search time: {name} - {seconds:0.4f} s",
    initial_text="Starting search...",
    logger=logger.info,
)
def search(
    embeddings: np.ndarray,
    data: pd.DataFrame,
    query_embedding: np.ndarray,
    top_k: int = 5,
) -> pd.DataFrame:
    search_results = pd.DataFrame(
        semantic_search(query_embedding, embeddings, top_k=top_k)[0]
    )
    search_results = search_results.set_index("corpus_id")

    search_results_data = data.iloc[search_results.index].copy()
    search_results_data["score"] = search_results["score"]

    return search_results_data
