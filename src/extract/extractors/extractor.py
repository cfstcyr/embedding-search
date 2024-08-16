import logging
from io import BytesIO
from pathlib import Path

import pandas as pd
from codetiming import Timer

StrOrPath = str | Path
File = StrOrPath | BytesIO

logger = logging.getLogger(__name__)


class Extractor:
    _file: File

    def __init__(self, file: File) -> None:
        self._file = file

    @Timer(
        text="Extracting text time: {seconds:.2f} seconds",
        initial_text="Starting extracting text...",
        logger=logger.info,
    )
    def extract_text(self) -> pd.DataFrame:
        return self._extract_text()

    def _extract_text(self) -> pd.DataFrame:
        with open(self._file) as f:
            text = f.read()

        df = pd.DataFrame({"text": text.split("\n")})
        df["page"] = [[1]] * len(df)

        return df
