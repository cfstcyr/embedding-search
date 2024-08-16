import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.log import init_logging

from .embed import embedder_factory_env
from .parser import embed_parser

load_dotenv()
init_logging()

logger = logging.getLogger(__name__)

args = embed_parser.parse_args()
input = Path(args.input)

data = pd.read_csv(input)

embedder = embedder_factory_env()
embeddings = embedder.embed(data[args.column_data])

output = args.output or input.with_suffix(".npy")
logger.info(f"Saving embeddings to {output}")
np.save(output, embeddings)
