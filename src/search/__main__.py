import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.embed import embedder_factory_env
from src.log import init_logging

from .parser import search_parser
from .search import search

load_dotenv()
init_logging()

args = search_parser.parse_args()

data = pd.read_csv(args.data)
embeddings = np.load(args.embeddings)

embedder = embedder_factory_env()
query_embedding = embedder.embed_one(args.query)

search_results = search(embeddings, data, query_embedding)
search_results["display"] = "# " + search_results["file"]

for col in search_results.columns[search_results.columns.str.match("h[1-6]")]:
    n = int(col[1:]) + 1
    search_results["display"] += "\n" + ("#" * n) + " " + search_results[col]

search_results["display"] += "\n\n" + search_results["text"]

print("\n\n=============\n\n".join(search_results["display"].tolist()))
