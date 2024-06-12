import pandas as pd
from loguru import logger


class ExcelLoader():
    """..."""

    def __init__(self, path, *args, **kwargs):
        self.df = None
        self.read_excel_calamine(path, *args, **kwargs)

    def read_excel_calamine(self, path, *args, **kwargs):
        """read_excel calamine engine"""
        df = pd.read_excel(path, *args, **kwargs, engine='calamine')
        self.df = df
        return self.df
