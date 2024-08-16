import logging
import os
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .extractors import Extractor, File, PDFExtractor, StrOrPath

logger = logging.getLogger(__name__)

EXTRACTORS: dict[str | list[str], type[Extractor]] = {
    "txt": Extractor,
    "pdf": PDFExtractor,
}


def _extract_file(file: File, type: str) -> pd.DataFrame:
    for ext, extractor in EXTRACTORS.items():
        if isinstance(ext, str) and re.match(ext, type) is None:
            continue
        if isinstance(ext, list) and type not in ext:
            continue

        return extractor(file).extract_text()

    raise NotImplementedError(f"Cannot extract file of type {type}")


def _combine(data: pd.DataFrame) -> pd.Series:
    data = data.copy()
    header_cols = [c for c in data.columns if c.startswith("h")]

    data["res"] = ""

    for i, col in enumerate(header_cols):
        n = len(header_cols) - i + 1
        for _ in range(n):
            data["res"] += data[col].fillna("") + "  "

    data["res"] += data["text"]

    return data["res"]


def extract(file_or_dir: StrOrPath) -> pd.DataFrame:
    extracted_data: pd.DataFrame | None = None

    files: list[Path] = []

    if os.path.isdir(file_or_dir):
        files = [Path(file_or_dir) / file for file in os.listdir(file_or_dir)]
    else:
        files = [Path(file_or_dir)]

    for file in tqdm(files, desc="Extracting files", unit="file"):
        print(file)

        if not file.is_file():
            print("not file")
            continue

        file_type = file.suffix[1:]

        try:
            data = _extract_file(file, file_type)
        except NotImplementedError as e:
            logger.warn(f"Skipping {file}: {e}")
            continue

        data.insert(0, "file", file.name)

        extracted_data = (
            data if extracted_data is None else pd.concat([extracted_data, data])
        )

    if extracted_data is None:
        raise ValueError(f"No files found in {file_or_dir}")

    extracted_data["combined_text"] = _combine(extracted_data)

    return extracted_data
