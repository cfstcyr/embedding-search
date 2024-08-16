import argparse

search_parser = argparse.ArgumentParser(
    prog="Search",
    description="Search query from embedded text",
)

search_parser.add_argument(
    "-q",
    "--query",
    type=str,
    help="Query to search",
    required=True,
)

search_parser.add_argument(
    "-e",
    "--embeddings",
    type=str,
    help="File with embeddings",
    required=True,
)

search_parser.add_argument(
    "-d",
    "--data",
    type=str,
    help="File with data",
    required=True,
)
