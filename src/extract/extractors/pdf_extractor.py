import math
from typing import Optional

import pandas as pd
from PyPDF2 import PdfReader

from .extractor import Extractor, File


class PDFExtractor(Extractor):
    _reader: PdfReader
    _df: Optional[pd.DataFrame] = None

    def __init__(self, file: File):
        super().__init__(file)
        self._reader = PdfReader(file)

    def _extract_text(self) -> pd.DataFrame:
        for i, page in enumerate(self._reader.pages):
            page.extract_text(visitor_text=lambda *args: self._visitor_body(i, *args))

        self._df = self._df.pipe(self._classify_text).pipe(self._clean_text)

        return self._df

    def _visitor_body(
        self, page_number: int, text: str, cm, tm, font, fontSize: float
    ) -> None:
        stripped_text: str = text.strip()

        if len(stripped_text) <= 0:
            return

        entry = pd.DataFrame(
            {"page": page_number, "size": fontSize, "text": text},
            index=[len(self._df) if self._df is not None else 0],
        )

        if self._df is None:
            self._df = entry
        else:
            self._df = pd.concat([self._df, entry])

    def _classify_text(self, df: pd.DataFrame) -> pd.DataFrame:
        sizes = df.groupby("size").agg({"text": "count"})
        p_size = sizes["text"].idxmax()

        headers = sizes[sizes.index > p_size].sort_index(ascending=False).reset_index()
        headers["type"] = "h" + (pd.to_numeric(headers.index) + 1).astype(str)

        df_p = df[df["size"] <= p_size].copy()
        df_p["type"] = "p"

        df_h = df[df["size"] > p_size]
        df_h = pd.merge_asof(
            df_h.sort_values("size").reset_index(),
            headers[["size", "type"]].sort_values("size"),
            left_on="size",
            right_on="size",
            direction="backward",
        )
        df_h.index = df_h["index"]
        df_h = df_h.drop(columns=["index"])

        df = pd.concat([df_p, df_h]).sort_index()

        return df

    def _clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        n_pages = df["page"].max() + 1

        repeated_text = (
            df.groupby(["text", "size"])
            .agg({"page": ["count", "nunique"]})
            .sort_values(("page", "nunique"), ascending=False)
        )
        repeated_text = repeated_text[
            repeated_text[("page", "nunique")] >= n_pages - 1
        ]  # Text that appears in all pages except one (notably page header/footer)

        df = df[~df.set_index(["text", "size"]).index.isin(repeated_text.index)].copy()

        headers = (
            df[df["type"].str.startswith("h")]["type"]
            .drop_duplicates()
            .sort_values()
            .reset_index(drop=True)
        )

        for header in headers:
            df.loc[df["type"] == header, header] = df.loc[df["type"] == header, "text"]

        for i, header in enumerate(headers):
            if i < len(headers) - 1:
                next_header: str = headers[i + 1]
                sections: pd.Series = df[header].dropna().drop_duplicates()

                for index, section in sections.items():
                    next_header_val = df.loc[index][next_header]
                    if isinstance(next_header_val, float) and math.isnan(
                        next_header_val
                    ):
                        df.loc[index, next_header] = section

            df[header] = df[header].ffill()

        df = df[df["type"] == "p"].copy().reset_index()
        df = (
            df.groupby(list(headers.values))
            .agg(
                {
                    "text": " ".join,
                    "page": lambda x: pd.Series(x)
                    .drop_duplicates()
                    .sort_values()
                    .tolist(),
                    "index": "first",
                }
            )
            .reset_index()
            .sort_values("index")
            .drop(columns=["index"])
        )
        df["text"] = df["text"].str.strip().str.replace(r"(\s|\n)+", " ", regex=True)

        return df
