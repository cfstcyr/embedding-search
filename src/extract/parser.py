import argparse

extract_parser = argparse.ArgumentParser(
    prog="Extract",
    description="Extract text from files",
)

extract_parser.add_argument(
    "-s",
    "--source",
    type=str,
    help="Path to file or directory to extract text from",
    required=True,
)

extract_parser.add_argument(
    "-o",
    "--output",
    type=str,
    help="Output file (default: extracted.csv in the parent directory of the input directory)",
)
