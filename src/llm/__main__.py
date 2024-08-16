from dotenv import load_dotenv

from src.log import init_logging

from .llm import llm_model_factory_env
from .parser import llm_parser

load_dotenv()
init_logging()

args = llm_parser.parse_args()

llm_model = llm_model_factory_env()
result = llm_model.ask(args.query)

print(result)
