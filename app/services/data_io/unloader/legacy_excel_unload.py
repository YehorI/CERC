from pathlib import Path

import pandas as pd
import openpyxl as xl

from ..utils.yaml import UtilYAML


class BaseExcelUnloader(UtilYAML):
    def __init__(self):
        ...

    @property
    def classpath(self):
        return Path(__file__).resolve().parent

    def write_to_excel(self,
        df: pd.DataFrame,
        filepath: str,
        sheet_name: str = 'Sheet1',
        startrow: int = 0,
        startcol: int = 0,
        header: bool = False,
    ):
        with pd.ExcelWriter(
            filepath,
            engine='openpyxl',
            mode='a',
            if_sheet_exists='overlay'
        ) as writer:
                df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=startrow,
                    startcol=startcol,
                    index=False,
                    header=header,
                )


class ColumnCorrections:
    def __init__(self):
        ...

    @staticmethod
    def to_dotseparated_date(df, column):
        # Check if the column is in the DataFrame
        if column not in df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame.")
        
        # Check if the column is of type datetime64
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            raise TypeError(f"Column '{column}' is not a datetime64 type.")

        # Format the datetime64 column to a string with dot-separated dates
        df[column] = df[column].dt.strftime('%d.%m.%Y')

        return df
    
    @staticmethod
    def to_kilos(df, column):
        df[column] = df[column].apply(lambda x: x * 1000)
        return df


class LegacyExcelUnloader(BaseExcelUnloader):
    def __init__(self,
        df,
        report_name,
        stream_type,
        stream_direction,
        month=None,
    ):

        self.df = df
        self.report_config = self.get_legacy_loader_yaml_config(
            report_name, stream_type, stream_direction
        )
        self.report_name = report_name
        self.stream_type = stream_type
        self.stream_direction = stream_direction

        if month is not None:
            self.df = self.df[self.df["date"].dt.month == month]

        self.exclude_columns()
        self.apply_column_coercions()

    @staticmethod
    def get_columns_to_exclude(report_config):
        try:
            cte = report_config["columns_to_exclude"]
        except KeyError:
            cte = []
        except TypeError as e:
            raise TypeError("The report configuration should be a dictionary.") from e
        if cte is None:
            return []
        return cte

    def exclude_columns(self):
        columns_to_exclude = self.get_columns_to_exclude(self.report_config)
        self.df = self.df.drop(columns=columns_to_exclude)

    @staticmethod
    def get_column_coercions(report_config):
        try:
            c_c = report_config["column_coercions"]
        except KeyError:
            c_c = []
        except TypeError as e:
            raise TypeError("The report configuration should be a dictionary.") from e
        if c_c is None:
            return []
        return c_c

    def apply_column_coercions(self):
        c_c = self.get_column_coercions(self.report_config)
        CC = ColumnCorrections()
        for coercion in c_c:
            coercion_func = getattr(CC, coercion["how"])
            coercion_column = coercion["column"]
            if "kwargs" in coercion:
                kwargs = coercion["kwargs"]
            else:
                kwargs = {}
            self.df = coercion_func(self.df, coercion_column, **kwargs)

    def get_legacy_loader_yaml_config(self,
        report_name, stream_type, stream_direction,
        path=None,
    ):
        if path is None:
            path = self.classpath / "models" / 'legacy_unloader.yaml'
        config = self.load_yaml_file(path)

        return config[report_name][stream_type][stream_direction]

    def column_correction(self, df):
        ...
    
    def write_safe(self, filepath: str, sheet_name: str):
        """
        Appends data to an existing sheet in an Excel file.

        Args:
            filepath (str): Path to the Excel file.
            sheet_name (str): Name of the sheet to write to.
        """
        # Load the existing data from the sheet
        existing_data = pd.read_excel(filepath, sheet_name=sheet_name, header=None)

        # Get the last row and column of the existing data
        last_row = existing_data.last_valid_index()
        if last_row is None:
            last_row = 0
        else:
            last_row += 1

        last_col = existing_data.columns.max()
        if last_col is None:
            last_col = 0

        # Write the new data to the sheet, starting from the next available row
        with pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            self.df.to_excel(writer, sheet_name=sheet_name, startrow=last_row, startcol=0, header=False, index=False)

    def write_unsafe(
        self,
        filepath,
        sheet_name,
    ):
        self.write_to_excel(
            self.df,
            filepath=filepath,
            sheet_name=sheet_name,
            startrow=1,
            startcol=0,
            header=False,
        )

    @staticmethod
    def get_sheet_row_quantity(path, sheet_name):
        # Load the Excel file
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
            # Return the number of rows in the sheet
            return df.shape[0]
        except Exception as e:
            print(f"An error occurred: {e}")
            return -1  # Return -1 or any other error code/value you prefer
