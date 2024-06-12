import datetime

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class DailyChartPlotter:
    def __init__(self,
        im_df=None,
        ex_df=None,
        t_im_df=None,
        t_ex_df=None
    ):
        self.im_df = im_df
        self.ex_df = ex_df
        self.t_im_df = t_im_df
        self.t_ex_df = t_ex_df
        self.combined_df = None
        self.pivot_data = None

    def process_data(self,
        month,
        only_till_yesterday=False,
    ):
        dfs = []

        if self.im_df is not None:
            self.im_df["landfill"] = "Ввоз Обработка"
            dfs.append(self.im_df)

        if self.ex_df is not None:
            dfs.append(self.ex_df)

        if self.t_im_df is not None:
            self.t_im_df["landfill"] = "Ввоз Перегруз"
            dfs.append(self.t_im_df)

        if self.t_ex_df is not None:
            dfs.append(self.t_ex_df)

        if not dfs:
            raise ValueError("At least one DataFrame must be provided.")

        grouped_dfs = []
        for df in dfs:
            grouped_dfs.append(self.group_date_land(df))

        self.combined_df = pd.concat(grouped_dfs)
        self.combined_df = pd.DataFrame(self.combined_df)
        self.combined_df.reset_index(inplace=True)
        self.combined_df.columns = ["date", "landfill", "weight"]

        self.combined_df = self.combined_df[self.combined_df['date'].dt.month == month]

        if only_till_yesterday:
            self.combined_df = self.combined_df[
                self.combined_df['date'].dt.day <= (datetime.datetime.now() - datetime.timedelta(days=1)).day
            ]

        pd.set_option('future.no_silent_downcasting', True)
        self.pivot_data = self.combined_df.pivot_table(
            index='date', columns='landfill',
            values='weight', aggfunc='sum', fill_value=0
        )

    @staticmethod
    def group_date_land(df):
        df = df.groupby(["date", "landfill"])["weight"].sum()
        return df

    @staticmethod
    def format_date_russian(date):
        months = [
            'января', 'февраля', 'марта', 'апреля', 'мая', 'июня',
            'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'
        ]
        return f"{date.day} {months[date.month - 1]}"

    def create_double_bar_stacked_chart(self,
        figsize=(10, 6),
        bar_width=0.35,
        title='',
        ylabel='',
        legend=True,
        import_palette=None, export_palette=None
    ):
        df = self.pivot_data.reset_index()

        unique_landfills = [
            col for col in df.columns \
            if not col.startswith('Ввоз') and col not in ("date", "landfill")
        ]

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Set the bar positions
        index = np.arange(len(df['date']))

        # Create the stacked bar chart for import categories
        import_categories = [col for col in df.columns if col.startswith('Ввоз')]
        bottom = np.zeros(len(df['date']))
        import_bars = []
        import_colors = sns.color_palette(import_palette, len(import_categories))
        for i, category in enumerate(import_categories):
            bar = ax.bar(index, df[category], bar_width, bottom=bottom, label=category, color=import_colors[i], alpha=0.7)
            import_bars.append(bar)
            bottom += df[category]

        # Create the stacked bar chart for export categories
        export_categories = [*unique_landfills]
        bottom = np.zeros(len(df['date']))
        export_bars = []
        export_colors = sns.color_palette(export_palette, len(export_categories))
        for i, category in enumerate(export_categories):
            bar = ax.bar(index + bar_width, df[category], bar_width, bottom=bottom, label=category, color=export_colors[i], alpha=1)
            export_bars.append(bar)
            bottom += df[category]

        # Add labels on the bars
        for date_index, date in enumerate(df['date']):
            current_import_height = 0
            for bar_index, bar in enumerate(import_bars):
                value = df[import_categories[bar_index]][date_index]
                if value != 0:  # Check if value is not zero
                    ax.text(index[date_index], current_import_height + value / 2, f"{value:g}", ha='center', va='center', rotation=90)
                current_import_height += value

            current_export_height = 0
            for bar_index, bar in enumerate(export_bars):
                value = df[export_categories[bar_index]][date_index]
                if value != 0:  # Check if value is not zero
                    ax.text(index[date_index] + bar_width, current_export_height + value / 2, f"{value:g}", ha='center', va='center', rotation=90)
                current_export_height += value

        # Add labels and title
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        # Set x-axis ticks and labels
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels([self.format_date_russian(date) for date in df['date']], rotation=45)

        # Add a legend if specified
        if legend:
            ax.legend()

        # Display the chart
        plt.tight_layout()
        plt.show()


class ByWeekDayPlotter:
    def __init__(self):
        ...
    
    def process_data(self):
        ...
    
    def create_byweekdayplot(self):
        ...


class DailyLandfillPlotter:
    def __init__(self,
        export_dfs: list[pd.DataFrame],
    ):
        self.df = pd.concat(export_dfs, ignore_index=True)
        self.grouped_df = None

    def process_data(self, month: int, only_till_yesterday=False) -> pd.DataFrame:
        # Ensure the date column is in datetime format
        self.df['date'] = pd.to_datetime(self.df['date'])

        # Filter the dataframe for the specified month
        filtered_df = self.df[self.df['date'].dt.month == month]
        if only_till_yesterday:
            filtered_df = filtered_df[
                filtered_df['date'].dt.day <= (datetime.datetime.now() - datetime.timedelta(days=1)).day
            ]

        # Group by landfill, date, and report_name, then sum the weights
        grouped_df = (filtered_df
                      .groupby(['landfill', filtered_df['date'].dt.date, 'report_name'])['landfill_weight']
                      .sum()
                      .reset_index()
                      .rename(columns={

                        })
                    )
        self.grouped_df = grouped_df
        return grouped_df

    def plot_landfills(self):
        if self.grouped_df is None:
            raise ValueError("Data has not been processed. Please run process_data() first.")

        # Get unique landfills
        landfills = self.grouped_df['landfill'].unique()

        for landfill in landfills:
            landfill_data = self.grouped_df[self.grouped_df['landfill'] == landfill]

            # Pivot the DataFrame to have report_dates as index and report_names as columns
            pivot_df = landfill_data.pivot_table(index='date', columns='report_name', values='landfill_weight', fill_value=0)

            # Plotting the data
            ax = pivot_df.plot(kind='bar', stacked=True, figsize=(10, 6))

            # Adding titles and labels
            plt.title(f'Статистика ввоза на {landfill}')
            plt.xlabel('')
            plt.ylabel('Масса, Т')
            plt.xticks(rotation=45)

            # Labeling each bar segment
            for p in ax.patches:
                width = p.get_width()
                height = p.get_height()
                x, y = p.get_xy()
                if height > 0:  # Only label positive bars
                    ax.annotate(f'{round(height, 2)}', (x + width / 2, y + height / 2), 
                                ha='center', va='center', fontsize=9, color='black')
            
            # Labeling the sum for each stacked bar
            for index, total_height in enumerate(pivot_df.sum(axis=1)):
                ax.annotate(f'{round(total_height, 2)}', (index, total_height), 
                            ha='center', va='bottom', fontsize=10, color='black', weight='bold')

            # Show the plot
            plt.show()
