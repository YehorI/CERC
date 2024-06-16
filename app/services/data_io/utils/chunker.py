import pandas as pd
import os
import glob
import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel


from app.settings import Settings

settings = Settings()


class ReportTypes(Enum):
    FIELD = "field"
    LANDFILL = "landfill"
    CARRIER = "carrier"


class ChunkIdentifier(BaseModel):
    report_type: ReportTypes
    year: int
    month: int
    submittion_time: datetime.datetime


class Chunker:
    def __init__(self, chunks_folder=settings.app_path / "db" / "chunks",):
        self.chunk_folder = chunks_folder
    
    def _get_chunk_path(self, report_type, report_name, year, month, submitted_order):
        file_pattern = f"{str(report_type.value)}_{report_name}_{year:04}_{month:02}_*.parquet"
        file_paths = sorted(glob.glob(str(self.chunk_folder / file_pattern)))
        if submitted_order == -1 or submitted_order >= len(file_paths):
            return file_paths[-1]  # Get the latest chunk if submitted_order is -1 or too large
        else:
            return file_paths[submitted_order]
    
    def load_chunk(self, report_type, report_name, year, month, submitted_order=-1):
        chunk_path = self._get_chunk_path(report_type, report_name, year, month, submitted_order)
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"No chunk found for {report_type.value} ({report_name}) in {year}-{month} with order {submitted_order}")
        return pd.read_parquet(chunk_path)

    def save_chunk(self, df, report_type, report_name, year, month):
        # Convert WindowsPath objects to strings
        for col in df.columns:
            if df[col].dtype == 'object' and any(isinstance(val, Path) for val in df[col]):
                df[col] = df[col].astype(str)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        chunk_path = self.chunk_folder / f"{report_type.value}_{report_name}_{year:04}_{month:02}_{timestamp}.parquet"
        
        os.makedirs(self.chunk_folder, exist_ok=True)
        df.to_parquet(chunk_path, index=False)

    def clean_chunks(self,
        report_type: list[ReportTypes],
        report_names: list[str],
        year: list[int],
        month: list[int],
        only_non_relevant=True
    ):
        for rt in report_type:
            for rn in report_names:
                for y in year:
                    for m in month:
                        chunk_pattern = f"{str(rt.value)}_{rn}_{y:04}_{m:02}_*.parquet"
                        chunk_files = glob.glob(str(self.chunk_folder / chunk_pattern))
                        
                        if only_non_relevant:
                            if len(chunk_files) > 1:
                                chunk_files = sorted(chunk_files)[:-1]  # Keep the latest one
                            else:
                                continue
                        
                        for file in chunk_files:
                            os.remove(file)
