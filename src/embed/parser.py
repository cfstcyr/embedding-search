import argparse

embed_parser = argparse.ArgumentParser(
    prog="Embed",
    description="Embed text",
)

embed_parser.add_argument(
    "-i",
    "--input",
    type=str,
    help="Input file with text to embed",
    required=True,
)

embed_parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output file (default: input file with .npy extension)",
)

embed_parser.add_argument(
    "-c",
    "--column-data",
    type=str,
    help="Column name with text to embed (default: combined_text)",
    default="combined_text",
)
