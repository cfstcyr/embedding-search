import pandas as pd
from openai.types.chat import ChatCompletionSystemMessageParam

from .llm_models import LLMModel
from .queries import LLM_SEARCH_QUERY, LLM_SEARCH_QUERY_SYSTEM


def llm_search(
    llm_model: LLMModel, query: str, search_results: pd.DataFrame
) -> pd.DataFrame:
    search_results["display"] = "# " + search_results["file"]
    for col in search_results.columns[search_results.columns.str.match("h[1-6]")]:
        n = int(col[1:]) + 1
        search_results["display"] += "\n" + ("#" * n) + " " + search_results[col]

    search_results["display"] += "\n\n" + search_results["text"]

    return llm_model.ask(
        LLM_SEARCH_QUERY.format(
            search_query=query,
            search_results="\n\n".join(search_results["display"].tolist()),
        ),
        previous_messages=[
            ChatCompletionSystemMessageParam(
                content=LLM_SEARCH_QUERY_SYSTEM, role="system", name="system"
            )
        ],
    )
