# timeseries_viz.py
from typing import Optional, Sequence
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ipywidgets import IntSlider, Layout, SelectMultiple, VBox, HBox, Label, Output
from IPython.display import display
import holidays
from datetime import timedelta

# -----------------------------
# Original draw function
# -----------------------------
def draw(df: pd.DataFrame,
         sensors: Optional[Sequence[str]] = None,
         slider_width: str = "1000px",
         sensors_width: str = "400px"):
    """
    Interactive plot for df (same as before).
    """
    # Validate index
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    all_cols = list(df.columns)
    if sensors is None:
        sensors = all_cols
    else:
        sensors = [s for s in sensors if s in all_cols]
        if len(sensors) == 0:
            sensors = all_cols

    n = len(df)
    if n < 2:
        raise ValueError("DataFrame must have at least 2 rows.")

    start_slider = IntSlider(min=0, max=n-2, step=1, value=0, description='Start', layout=Layout(width=slider_width))
    end_slider   = IntSlider(min=1, max=n-1, step=1, value=n-1, description='End', layout=Layout(width=slider_width))

    sensor_selector = SelectMultiple(options=all_cols, value=tuple(sensors), description='Sensors', layout=Layout(width=sensors_width))
    start_label = Label(value=str(df.index[start_slider.value]))
    end_label   = Label(value=str(df.index[end_slider.value]))
    out = Output(layout=Layout(width='100%'))

    def render_plot():
        with out:
            out.clear_output(wait=True)
            s_idx = int(start_slider.value)
            e_idx = int(end_slider.value)
            if s_idx >= e_idx:
                print("Start index must be < End index.")
                return None

            start_date = df.index[s_idx]
            end_date = df.index[e_idx]
            df_sel = df.loc[start_date:end_date]

            fig = go.Figure()
            selected = list(sensor_selector.value)
            if len(selected) == 0:
                print("No sensors selected.")
                return None

            for col in selected:
                if col in df_sel:
                    fig.add_trace(go.Scatter(x=df_sel.index, y=df_sel[col], mode='lines', name=col))

            fig.update_layout(
                title=f"Sensor Time Series ({start_date} → {end_date})",
                xaxis_title="Time", yaxis_title="Value",
                hovermode="x unified",
                xaxis=dict(rangeslider=dict(visible=True)),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig.show()
            return fig

    def _on_slider_change(change):
        start_label.value = str(df.index[start_slider.value])
        end_label.value = str(df.index[end_slider.value])
        render_plot()

    def _on_sensors_change(change):
        render_plot()

    start_slider.observe(_on_slider_change, names='value')
    end_slider.observe(_on_slider_change, names='value')
    sensor_selector.observe(_on_sensors_change, names='value')

    render_plot()

    slider_box = VBox([
        HBox([start_slider, VBox([Label("Start timestamp"), start_label])]),
        HBox([end_slider,   VBox([Label("End timestamp"), end_label])])
    ])
    left_col  = VBox([sensor_selector])
    right_col = VBox([slider_box])
    top_row = HBox([left_col, right_col], layout=Layout(justify_content='flex-start', width='100%'))
    display(VBox([top_row, out]))
    return out, render_plot

# -----------------------------
# New function with holidays
# -----------------------------
de_holidays = holidays.Germany(years=range(2020, 2030))


def plot_with_holidays(df: pd.DataFrame,
                       sensors: Optional[Sequence[str]] = None,
                       step: int = 5,
                       slider_width: str = "1000px",
                       sensors_width: str = "400px"):
    """
    Interactive Plotly time series plot with optional holiday vertical lines and tooltips.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex
    sensors : sequence of str, optional
        Sensors/columns to plot. Defaults to all.
    step : int
        Width of holiday vertical lines in time steps.
    slider_width : str
        CSS width of start/end sliders
    sensors_width : str
        CSS width of sensor multi-select
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    all_cols = list(df.columns)
    if sensors is None:
        sensors = all_cols
    else:
        sensors = [s for s in sensors if s in all_cols]
        if len(sensors) == 0:
            sensors = all_cols

    n = len(df)
    if n < 2:
        raise ValueError("DataFrame must have at least 2 rows.")

    # German holidays
    de_holidays = holidays.Germany(years=df.index.year.unique())

    # Widgets
    start_slider = IntSlider(min=0, max=n-2, step=1, value=0,
                             description='Start', layout=Layout(width=slider_width))
    end_slider = IntSlider(min=1, max=n-1, step=1, value=n-1,
                           description='End', layout=Layout(width=slider_width))
    sensor_selector = SelectMultiple(
        options=all_cols,
        value=tuple(sensors),
        description='Sensors',
        layout=Layout(width=sensors_width)
    )
    start_label = Label(value=str(df.index[start_slider.value]))
    end_label = Label(value=str(df.index[end_slider.value]))
    out = Output(layout=Layout(width='100%'))

    def render_plot():
        with out:
            out.clear_output(wait=True)
            s_idx = int(start_slider.value)
            e_idx = int(end_slider.value)
            if s_idx >= e_idx:
                print("Start index must be < End index.")
                return None

            start_date = df.index[s_idx]
            end_date = df.index[e_idx]
            df_sel = df.loc[start_date:end_date]

            fig = go.Figure()
            selected = list(sensor_selector.value)
            if len(selected) == 0:
                print("No sensors selected.")
                return None

            # Plot sensors
            for col in selected:
                fig.add_trace(go.Scatter(
                    x=df_sel.index,
                    y=df_sel[col],
                    mode='lines',
                    name=col
                ))

            # Add holidays
            for h_date, h_name in de_holidays.items():
                h_ts = pd.Timestamp(h_date)
                if start_date <= h_ts <= end_date:
                    try:
                        idx = df_sel.index.get_loc(h_ts)
                        idx_end = min(idx + step, len(df_sel) - 1)
                        # Vertical band
                        fig.add_vrect(
                            x0=df_sel.index[idx],
                            x1=df_sel.index[idx_end],
                            fillcolor="black",
                            opacity=0.6,
                            line_width=.5,
                            layer="below"
                        )
                        # Invisible scatter for tooltip
                        fig.add_trace(go.Scatter(
                            x=[df_sel.index[idx], df_sel.index[idx_end]],
                            y=[df_sel[selected[0]].max(), df_sel[selected[0]].max()],
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='text',
                            hovertext=f"{h_name} ({h_ts.date()})"
                        ))
                    except KeyError:
                        continue

            fig.update_layout(
                title=f"Sensor Time Series ({start_date} → {end_date})",
                xaxis_title="Time",
                yaxis_title="Value",
                hovermode="x unified",
                xaxis=dict(rangeslider=dict(visible=True)),
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="right", x=1)
            )

            fig.show()
            return fig

    # Handlers
    def _on_slider_change(change):
        start_label.value = str(df.index[start_slider.value])
        end_label.value = str(df.index[end_slider.value])
        render_plot()

    def _on_sensors_change(change):
        render_plot()

    start_slider.observe(_on_slider_change, names='value')
    end_slider.observe(_on_slider_change, names='value')
    sensor_selector.observe(_on_sensors_change, names='value')

    render_plot()

    # Layout
    slider_box = VBox([
        HBox([start_slider, VBox([Label("Start timestamp"), start_label])]),
        HBox([end_slider, VBox([Label("End timestamp"), end_label])])
    ])
    left_col = VBox([sensor_selector])
    right_col = VBox([slider_box])
    top_row = HBox([left_col, right_col], layout=Layout(justify_content='flex-start', width='100%'))
    display(VBox([top_row, out]))

def detect_frequency_regions(df, tolerance='1s'):
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex")

    # Compute rounded diffs
    diffs = df.index.to_series().diff().dropna().dt.round(tolerance)

    # Identify where the frequency changes
    change_points = diffs != diffs.shift(1)
    change_points = change_points[1:]

    # Initialize results
    time = df.index.to_series()
    start_time = time.iloc[0]
    start_loc = 0

    for i in range(len(change_points)):
        if change_points.iloc[i]:
            end_time = time.iloc[i+1]
            count = i + 2 - start_loc
            print(f"From {start_time} to {end_time}, frequency is {diffs[time.iloc[i+1]]}, and count is {count}")
            start_time = end_time
            start_loc = i+1
    count = len(time) - start_loc

    print(f"From {start_time} to {time.iloc[-1]}, frequency is {diffs[time.iloc[-1]]}, and count is {count}")




def ETL_single(path,
               column_names,
               date_column,
               column_format=[np.float16,np.float16],
               make_date_index=True,
               header=2,
               add_col_sort_change = True,
               date_format="%Y-%m-%d %H:%M:%S",
):
    """ETL function for a single CSV file.
    Args:
        path (str): Path to the CSV file.
        column_names (list): List of column names to assign to the DataFrame.
        date_column (str): Name of the column containing date information.
        make_date_index (bool, optional): Whether to set the date column as index. Defaults to True.
        date_format (str, optional): Format of the date in the date column. Defaults to "%Y-%m-%d %H:%M:%S".
    Returns:
        pd.DataFrame: Processed DataFrame.

    """
    # Load
    path = path.replace('\\','/')
    file_extension = path.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(path, header=header)
    elif file_extension in ['xlsx', 'xls']:
        df = pd.read_excel(path, header=header)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    
    # Transform
    df.columns = column_names
    df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
    if make_date_index:
        df.set_index(date_column, inplace=True)

    # Describe and check
    print("DataFrame Description and initial cleaning report:")
    print("---------------------------------------")
    print("Shape:", df.shape)
    duplicated_num_timestamps = df.index.duplicated().sum()
    print("===================================================")
    print("Duplicated Timestamps:",duplicated_num_timestamps)
    if duplicated_num_timestamps > 0:
        print(df.index.value_counts()[df.index.value_counts() > 1])
        df = df[~df.index.duplicated(keep='first')]
        print("After removing duplicates, new shape:", df.shape)
    
    # sort items
    print("===================================================")
    if df.index.is_monotonic_increasing:
        print("Index is already sorted.")
    else:
        print("Index is not sorted.")
        if add_col_sort_change:
            df_sorted = df.sort_index()
            df_sorted['Sort_Change'] = (df.index != df_sorted.index).astype(int)
            df = df_sorted
            print("Added 'Sort_Change' column")
            print("Is now sorted:",df.index.is_monotonic_increasing)
        else:
            df = df.sort_index()
            print("Is now sorted:",df.index.is_monotonic_increasing)
        

    freq_counts = df.index.to_series().diff().nunique()
    print("===================================================")
    print("Number of frequency of Dates:" ,freq_counts)
    print(df.index.diff()[1:].value_counts())
    print("Frequency variation point report:")
    print("---------------------------------------")
    detect_frequency_regions(df)
    print("===================================================")
    for i,col in enumerate(df.columns):
        print(col,"dtype before:",df[col].dtype)
        df[col]= df[col].astype(column_format[i])
        print(col,"dtype after:",df[col].dtype)
        print(f"Number of null values for {col} is {df[col].isnull().sum()}")

    return df
    

    

