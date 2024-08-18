import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.embed.embed import embedder_factory_env
from src.log import init_logging
from src.search import search

from .llm import llm_model_factory_env
from .llm_search import llm_search
from .parser import llm_parser

load_dotenv()
init_logging()

args = llm_parser.parse_args()

data = pd.read_csv(args.data)
embeddings = np.load(args.embeddings)

embedder = embedder_factory_env()
query_embedding = embedder.embed_one(args.query)

search_results = search(embeddings, data, query_embedding)

llm_model = llm_model_factory_env()
llm_search_results = llm_search(llm_model, args.query, search_results)

print(llm_search_results)
