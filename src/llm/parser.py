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
