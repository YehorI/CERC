import re
import datetime
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import numpy as np

from .excel_loader import ExcelLoader
from ..utils.yaml import UtilYAML

pd.set_option('future.no_silent_downcasting', True)


class LoadTransformations():
    def __init__(self):
        ...

    @staticmethod
    def get_value_indecies(df, value, value_column_index=0):
        """sumary_line"""
        ...

    @staticmethod
    def select_after_certain_value(df, value, value_column_index=0, value_index=0, offset_after_value=0):
        """sumary_line"""
        df = df.reset_index(drop=True)
        start = df.index[
            df.iloc[:, value_column_index] == value
        ].to_list()[value_index] + offset_after_value
        df.columns = df.iloc[start]
        return df.iloc[start + 1:]

    @staticmethod
    def select_before_certain_value(df, value, value_column_index=0, value_index=0, offset_after_value=0):
        """sumary_line"""
        df = df.reset_index(drop=True)
        index = df.index[df.iloc[:, value_column_index] == value].to_list()[value_index] + offset_after_value
        return df.iloc[:index]

    @staticmethod
    def remove_empty_columns(df):
        """sumary_line"""
        valid_columns = df.columns[df.columns.notna()]
        return df.loc[:, [col for col in valid_columns if not str(col).startswith('Unnamed')]]

    @staticmethod
    def remove_columns(df, columns_to_exclude: list):
        if not isinstance(columns_to_exclude, list):
            raise ValueError("columns_to_exclude must be a list of column names.")
        missing_columns = [col for col in columns_to_exclude if col not in df.columns]
        if missing_columns:
            raise KeyError(f"The following columns are not in the DataFrame: {missing_columns}")
        return df.drop(columns=columns_to_exclude)

    @staticmethod
    def remove_empty_lines(df):
        mask = df.map(lambda x: str(x).strip() == '').all(axis=1) | \
            df.isnull().all(axis=1) | \
            (df == 0).all(axis=1) | \
            ((df == 0) | df.isnull() | (df.map(lambda x: str(x).strip() == ''))).all(axis=1)
        df_cleaned = df[~mask].reset_index(drop=True)
        return df_cleaned

    @staticmethod
    def remove_total(df):
        mask = ~(
            df.iloc[:, 0].str.contains("ИТОГО:") |
            df.iloc[:, 0].str.contains("ВСЕГО:")
        )
        df = df[mask]
        return df

    @staticmethod
    def filter_df_by(df, colname, fieldname):
        mask = df[colname] == fieldname
        df = df[mask]
        return df

    @staticmethod
    def skip_first_lines(df, n_lines):
        df = df.iloc[n_lines:]
        return df

    @staticmethod
    def collapse_multiheader(df, delimiter="_"):
        # Combine the MultiIndex column headers and filter out "Unnamed" parts
        df.columns = [
            delimiter.join([
                str(elem) for elem in col if isinstance(elem, str) and 'Unnamed' not in elem
            ]).strip()
            for col in df.columns.values
        ]
        return df


class Coercions(UtilYAML):
    def __init__(self):
        ...

    @property
    def classpath(self):
        return Path(__file__).resolve().parent

    def landfill_right_name(self, df):
        fields_mapping_path = self.classpath / "models" / "fields.yaml"
        config = self.load_yaml_file(fields_mapping_path)
        mapping = {}
        for item in config['Fields']:
            for right_name, aliases in item.items():
                for alias in aliases:
                    mapping[alias] = right_name

        unique_landfills = df['landfill'].unique()
        unknown_aliases = [alias for alias in unique_landfills if alias not in mapping]
        if unknown_aliases:
            raise ValueError(f"Unknown landfill aliases in DataFrame: {unknown_aliases}")

        df["landfill"] = df["landfill"].map(mapping).fillna(df["landfill"])

        return df

    @staticmethod
    def _to_tonnes(df, target_col):
        df[target_col] = df[target_col] / 1000
        return df

    @staticmethod
    def remove_non_numeric_symbols(df, column_name):
        # Remove non-numeric symbols from the specified column
        df[column_name] = df[column_name].astype(str).str.replace(r'[^0-9.,]', '', regex=True)
        # Convert the column to numeric data type
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        
        return df

    @staticmethod
    def datetime2date(df):
        # Convert the "date" column to datetime format
        df["datetime"] = pd.to_datetime(df["date"], format="%d.%m.%Y %H:%M:%S")
        # Extract the date component from the "datetime" column and create a new "date" column
        df["date"] = df["datetime"].dt.date
        df["date"] = pd.to_datetime(df["date"])
        # Extract the time component from the "datetime" column and create a new "time" column
        df["time"] = df["datetime"].dt.time
        # Drop the temporary "datetime" column
        df.drop("datetime", axis=1, inplace=True)

        return df

    @staticmethod
    def to_datetime(df):
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def date_from_dotseparated(df):
        # Convert the "date" column to datetime format
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="raise")
        return df

    def date_from_dotseparated_super(self, df):
        """
        Handles date correction:
        - Corrects dates with excessive dots, e.g., 02.05.20.24 -> 02.05.2024
        - Corrects improperly formatted years like 0202 to a plausible year (2024 hardcoded), if necessary
        """

        def correct_excessive_dots(date_str):
            return re.sub(r'(\d{2}\.\d{2}\.\d{2})\.(\d{2})', r'\1\2', date_str)
        
        def correct_0202_year(date_str):
            if re.search(r'\.0202$', date_str):
                date_str = re.sub(r'(\d{2}\.\d{2})\.0202$', r'\1.2024', date_str)
            return date_str
        
        def ensure_string(date):
            return date.strftime("%d.%m.%Y") if isinstance(date, pd.Timestamp) else date
        
        correction_fns = [correct_excessive_dots, correct_0202_year]

        def try_correct_and_parse(date_str):
            for correction_fn in correction_fns:
                corrected_date_str = correction_fn(date_str)
                try:
                    return pd.to_datetime(corrected_date_str, format="%d.%m.%Y", errors="raise")
                except (ValueError, pd.errors.OutOfBoundsDatetime):
                    date_str = corrected_date_str
            return pd.to_datetime(date_str, format="%d.%m.%Y", errors="raise")

        for idx, date in df['date'].items():
            date_str = ensure_string(date)
            try:
                df.at[idx, 'date'] = pd.to_datetime(date_str, format="%d.%m.%Y", errors="raise")
            except (ValueError, pd.errors.OutOfBoundsDatetime):
                try:
                    df.at[idx, 'date'] = try_correct_and_parse(date_str)
                except (ValueError, pd.errors.OutOfBoundsDatetime) as final_error:
                    raise ValueError(f"Unable to parse and correct date: {date_str}") from final_error
        df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="raise")
        return df

    @staticmethod
    def remove_rows_with_na_of_certain_columns(df, columns):
        df = df[~df[columns].isna().all(axis=1)]
        return df

    @staticmethod
    def choose_month(df, month_number):
        df = df[df["date"].dt.month == month_number]
        return df

    @staticmethod
    def fix_vehicle_number(df):
        def fix_number(number):
            return str(number).upper().replace(" ", "")

        def transliterate_number(text):
            # Mapping of English-looking letters to their Cyrillic equivalents
            mapping = {
                'A': 'А',  # Cyrillic 'A'
                'B': 'В',  # Cyrillic 'B' (looks like English 'B')
                'E': 'Е',  # Cyrillic 'E'
                'K': 'К',  # Cyrillic 'K'
                'M': 'М',  # Cyrillic 'M'
                'H': 'Н',  # Cyrillic 'H' (looks like English 'N')
                'O': 'О',  # Cyrillic 'O'
                'P': 'Р',  # Cyrillic 'P' (looks like English 'R')
                'C': 'С',  # Cyrillic 'C' (looks like English 'S')
                'T': 'Т',  # Cyrillic 'T'
                'Y': 'У',  # Cyrillic 'Y' (looks like English 'U')
                'X': 'Х',  # Cyrillic 'X' (looks like English 'H')
            }
            # Transliterate each character using the mapping
            transliterated = ''.join(mapping.get(char, char) for char in text)
            return transliterated
        
        func_list = [fix_number, transliterate_number]

        for func in func_list:
            df["vehicle_number"] = df["vehicle_number"].apply(func)
        
        return df

    @staticmethod
    def duplicate_column(df, column, to_column, place='end'):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        if to_column in df.columns:
            raise ValueError(f"Column '{to_column}' already exists in the DataFrame.")
        
        if place not in ['end', 'after']:
            raise ValueError("Invalid value for 'place' parameter. Must be either 'end' or 'after'.")
        
        if place == 'end':
            df[to_column] = df[column]
        elif place == 'after':
            column_index = df.columns.get_loc(column)
            df.insert(column_index + 1, to_column, df[column])
        
        return df


class AutotestLoad():

    def __init__(self):
        ...

    @staticmethod
    def verify_type(df, col, expected_type):
        ...

    def verify_date_datetime64(self, df):
        self.verify_type(df, "date", "datetime64[ns]")


class ReportLoader(ExcelLoader, UtilYAML):
    def __init__(self,
        path,
        config: list[tuple] | OrderedDict,
        *args,
        report_config_overrides: dict = None,
        path_to_reports_yaml=None,
        **kwargs,
    ):

        if str(path).endswith(
            (".xlsx", "xls", "xlsm")
        ):
            self.path = path
            self.config = OrderedDict(config)
            self.report_config = self.load_report_config(
                # report_name, stream_type, stream_direction,
                path_to_reports_yaml=path_to_reports_yaml,
                **self.config,
            )
            if report_config_overrides is not None:
                self.report_config = self.recursive_update(self.report_config, report_config_overrides)
            self.load_kwargs = self.get_load_kwargs(self.report_config)
            self.lts = self.get_load_transformations(self.report_config)
            self.coercions = self.get_coercions(self.report_config)
            self.column_mapping = self.get_column_mapping(self.report_config)

            ExcelLoader.__init__(self, path, *args, **kwargs, **self.load_kwargs)

            self.apply_load_transformations()
            self.rename_columns(columns_map=self.column_mapping)
            self.apply_coercions()
            self.make_meta_columns()
        elif str(path).endswith(
            (".txt", "##")
        ):
            ...

    def recursive_update(self, base_dict, update_dict):
        for key, value in update_dict.items():
            if isinstance(value, dict):
                base_dict[key] = self.recursive_update(base_dict.get(key, {}), value)
            else:
                base_dict[key] = value
        return base_dict

    @property
    def classpath(self):
        return Path(__file__).resolve().parent

    def make_meta_columns(self):
        if 'stream_type' in self.config:
            self.df["stream_type"] = self.config["stream_type"]
        if 'stream_direction' in self.config:
            self.df["stream_direction"] = self.config["stream_direction"]
        self.df["source"] = self.path
        if 'report_name' in self.config:
            self.df["report_name"] = self.config["report_name"]

    def load_report_config(self,
        # report_name, stream_type, stream_direction,
        path_to_reports_yaml=None,
        **kwargs,
    ):
        if path_to_reports_yaml is None:
            path_to_reports_yaml = self.classpath / "models" / "reports.yaml"
        config = self.load_yaml_file(path_to_reports_yaml)
        for key, value in kwargs.items():
            config = config.get(value)
            if config is None:
                raise ValueError(f"Config for {value} not found")
        
        return config

    @staticmethod
    def get_load_kwargs(report_config):
        try:
            load_kwargs = report_config["load_kwargs"]
            load_kwargs = {k: v for k, v in load_kwargs.items() if v is not None}
        except KeyError:
            load_kwargs = {}
        except AttributeError:
            load_kwargs = {}
        except TypeError as e:
            raise TypeError("The report configuration should be a dictionary.") from e
        return load_kwargs

    @staticmethod
    def get_load_transformations(report_config):
        try:
            lts = report_config["load_transformations"]
        except KeyError:
            lts = []
        except TypeError as e:
            raise TypeError("The report configuration should be a dictionary.") from e
        if lts is None:
            return []
        return lts

    def apply_class_funcs_to_df(self,
        ClassContainer,
        func_names
    ):
        class_ = ClassContainer()
        for func in func_names:
            if isinstance(func, dict) and 'how' in func:
                method_name = func['how']
                kwargs = func.get('kwargs', {})
                self.df = getattr(class_, method_name)(self.df, **kwargs)
            else:
                self.df = getattr(class_, func)(self.df)

    def apply_load_transformations(self):
        self.apply_class_funcs_to_df(LoadTransformations, self.lts)

    @staticmethod
    def get_column_mapping(report_config):
        try:
            c_m = report_config["column_mapping"]
        except KeyError:
            c_m = {}
        except TypeError as e:
            raise TypeError("The report configuration should be a dictionary.") from e
        if c_m is None:
            return {}
        return c_m

    def rename_columns(self, columns_map: dict):
        self.df = self.df.rename(columns=columns_map)

    @staticmethod
    def get_coercions(report_config):
        try:
            coercions = report_config["coercions"]
        except KeyError:
            coercions = []
        except TypeError as e:
            raise TypeError("The report configuration should be a dictionary.") from e
        if coercions is None:
            return []
        return coercions

    def apply_coercions(self):
        self.apply_class_funcs_to_df(Coercions, self.coercions)
