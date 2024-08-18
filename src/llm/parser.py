import argparse

llm_parser = argparse.ArgumentParser(
    prog="LLM",
    description="Query a language model",
)

llm_parser.add_argument(
    "-q",
    "--query",
    type=str,
    help="Query",
    required=True,
)

llm_parser.add_argument(
    "-e",
    "--embeddings",
    type=str,
    help="File with embeddings",
    required=True,
)

llm_parser.add_argument(
    "-d",
    "--data",
    type=str,
    help="File with data",
    required=True,
)
