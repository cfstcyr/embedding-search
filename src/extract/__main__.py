import logging
import os

from dotenv import load_dotenv

from src.log import init_logging

from .extract import extract
from .parser import extract_parser

load_dotenv()
init_logging()

logger = logging.getLogger(__name__)

args = extract_parser.parse_args()

output = args.output or os.path.join(os.path.dirname(args.source), "extracted.csv")

extracted = extract(args.source)

logger.info(f"Saving extracted data to {output}")
extracted.to_csv(output, index=False)
