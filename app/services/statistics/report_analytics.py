import datetime
import pandas as pd


TimePeriod = tuple[datetime.datetime, datetime.datetime]


class AnalyticsBase:
    def __init__(self, df: pd.DataFrame | None = None, initial_balance: float = 0.0):
        self.current_month_period = self.get_current_month_time_period()
        self.df = df if df is not None else pd.DataFrame()
        self.initial_balance = initial_balance

    @staticmethod
    def get_current_month_time_period() -> TimePeriod:
        """
        Get the time period for the current month.

        :return: A tuple containing the first day of the current month and yesterday's date.
        """
        today = datetime.datetime.combine(
            datetime.datetime.today(), datetime.time.min
        )
        yesterday = today - datetime.timedelta(days=1)
        first_day_of_month = datetime.datetime(today.year, today.month, 1)
        return first_day_of_month, yesterday

    def apply_filters(self,
        time_period: TimePeriod | None = None,
        filters: dict | None = None
    ) -> pd.DataFrame:
        if time_period is None:
            time_period = self.current_month_period
        if filters is None:
            filters = {}

        start_date, end_date = time_period
        period_filter = (self.df["date"] >= start_date) & (self.df["date"] <= end_date)

        additional_filters = [period_filter]
        for column, value in filters.items():
            additional_filters.append(self.df[column] == value)

        combined_filters = additional_filters[0]
        for filter_condition in additional_filters[1:]:
            combined_filters &= filter_condition

        return self.df[combined_filters]

    def get_weight_sum_for_dates(self,
        time_period: TimePeriod | None = None,
        filters: dict = {},
    ) -> float:

        filtered_df = self.apply_filters(time_period, filters)
        return filtered_df["weight"].sum()

    def get_route_count_for_dates(self,
        time_period: TimePeriod | None = None,
        filters: dict = {},
    ) -> int:

        filtered_df = self.apply_filters(time_period, filters)
        return filtered_df["weight"].count() 

    def get_daily_average_weight(self,
        time_period: TimePeriod | None = None,
        filters: dict = {},
    ) -> float:

        filtered_df = self.apply_filters(time_period, filters)
        daily_weights = filtered_df.groupby("date")["weight"].sum()
        average_daily_weight = daily_weights.mean()
        return average_daily_weight

    def get_daily_average_route_count(self,
        time_period: TimePeriod | None = None,
        filters: dict = {},
    ) -> float:

        filtered_df = self.apply_filters(time_period, filters)
        daily_weights = filtered_df.groupby("date")["weight"].count()
        average_daily_weight = daily_weights.mean()
        return average_daily_weight

    def get_balance(self,
        time_period: TimePeriod | None = None
    ) -> float:

        if time_period is None:
            time_period = self.current_month_period

        import_filters = {'stream_direction': 'import'}
        export_filters = {'stream_direction': 'export'}
        import_sum = self.get_weight_sum_for_dates(time_period, import_filters)
        export_sum = self.get_weight_sum_for_dates(time_period, export_filters)

        return self.initial_balance + import_sum - export_sum

    def get_average_route_weight(self, 
        time_period: TimePeriod | None = None,
        filters: dict | None = None,
    ) -> float:
        filtered_df = self.apply_filters(time_period, filters)
        return filtered_df["weight"].mean()


class DailyAnalytics(AnalyticsBase):
    def get_daily_formatted_analytics(self,
        time_period: TimePeriod | None = None,
    ) -> str:
        if time_period is None:
            time_period = self.current_month_period

        f_date_1 = time_period[0].strftime("%d.%m.%Y")
        f_date_2 = time_period[1].strftime("%d.%m.%Y")

        yesterday_period = (time_period[1], time_period[1])
        import_filters = {'stream_direction': 'import'}
        export_filters = {'stream_direction': 'export'}

        yesterday_import_weight = self.get_weight_sum_for_dates(yesterday_period, import_filters)
        yesterday_export_weight = self.get_weight_sum_for_dates(yesterday_period, export_filters)
        yesterday_import_count = self.get_route_count_for_dates(yesterday_period, import_filters)
        yesterday_export_count = self.get_route_count_for_dates(yesterday_period, export_filters)

        average_daily_import_weight = self.get_daily_average_weight(time_period, import_filters)
        average_daily_export_weight = self.get_daily_average_weight(time_period, export_filters)

        average_daily_import_count = self.get_daily_average_route_count(time_period, import_filters)
        average_daily_export_count = self.get_daily_average_route_count(time_period, export_filters)

        balance = self.get_balance(time_period)

        unique_stream_types = self.df["stream_type"].unique()

        match unique_stream_types.tolist():
            case ["processing"]:
                stream_type = "обработка"
            case ["tranship"]:
                stream_type = "перегруз"
            case ["processing", "tranship"]:
                stream_type = "обработка и перегруз"
            case _:
                stream_type = "странный"

        if len(self.df["report_name"].unique()) == 1:
            report_name = self.df["report_name"].iloc[0]
        else:
            report_name = "Странный"

        f_string = (
            f"{report_name} {stream_type} {f_date_1} - {f_date_2}\n"
            "\n"
            f"Ввоз - {round(yesterday_import_weight, 2)}\n"
            f"Вывоз - {round(yesterday_export_weight, 2)}\n"
            "\n"
            f"Кол-во рейсов ввоз - {yesterday_import_count}\n"
            f"Кол-во рейсов вывоз - {yesterday_export_count}\n"
            "\n"
            f"Среднесуточный ввоз - {round(average_daily_import_weight, 2)}\n"
            f"Среднесуточный вывоз - {round(average_daily_export_weight, 2)}\n"
            "\n"
            f"Среднесуточное кол-во рейсов (ввоз) - {round(average_daily_import_count)}\n"
            f"Среднесуточное кол-во рейсов (вывоз) - {round(average_daily_export_count)}\n"
            "\n"
            f"Предварительный остаток - {round(balance, 2)}"
            f"{' без вычета ВМР' if stream_type == 'processing' else ''}"
        )
        return f_string


class ByWeekDayAnalytics(AnalyticsBase):
    def __init__(self, df: pd.DataFrame = None):
        super().__init__(df)
        self.make_weekday_column()

    def make_weekday_column(self) -> None:
        """
        Adds a 'weekday' column to the DataFrame which represents the day of the week.
        """
        self.df["weekday"] = self.df["date"].dt.day_name()

    def _sort_by_weekday(self, series: pd.Series) -> pd.Series:
        """
        Sorts the Series by the order of the weekdays starting from Monday.
        """
        ordered_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        series.index = pd.Categorical(series.index, categories=ordered_weekdays, ordered=True)
        return series.sort_index()

    def get_weekdays_average_counts(self,
        time_period: TimePeriod | None = None,
        filters: dict | None = None,
    ) -> pd.Series:
        if time_period is None:
            time_period = self.current_month_period

        filtered_df = self.apply_filters(time_period, filters)
        
        # Calculate the counts per day first, then average per weekday
        daily_counts = filtered_df.groupby(["date", "weekday"]).size().reset_index(name="count")
        avg_counts = daily_counts.groupby("weekday")["count"].mean()
        
        return self._sort_by_weekday(avg_counts)

    def get_weekdays_average_weights(self,
        time_period: TimePeriod | None = None,
        filters: dict | None = None,
    ) -> pd.Series:
        if time_period is None:
            time_period = self.current_month_period

        filtered_df = self.apply_filters(time_period, filters)
        
        # Calculate the daily sums first, then average per weekday
        daily_weights = filtered_df.groupby(["date", "weekday"])["weight"].sum().reset_index(name="daily_sum")
        avg_weights = daily_weights.groupby("weekday")["daily_sum"].mean()
        
        return self._sort_by_weekday(avg_weights)
