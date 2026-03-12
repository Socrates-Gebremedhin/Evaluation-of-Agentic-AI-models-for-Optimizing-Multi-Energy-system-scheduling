import pandas as pd


def run_sliding_24h_forecast(
    date_time_series,
    frequency,
    components,
    elec_demand,
    gas_demand,
    pv_generated,
    lp_optimizer,
    **optimizer_kwargs,
):
    """
    Run the LP optimizer sequentially over the full date_time_series
    in non-overlapping 24-hour blocks.

    Inputs:
        date_time_series : pandas.DatetimeIndex (full horizon)
        frequency        : minutes per timestep (e.g. 60)
        components       : same dict used by the optimizer
        elec_demand      : full pandas Series
        gas_demand       : full pandas Series
        pv_generated     : full pandas Series
        lp_optimizer     : callable( date_time_index=..., frequency=..., components=...,
                                    elec_demand=..., gas_demand=..., pv_generated=..., **kwargs ) -> DataFrame
        **optimizer_kwargs : passed through to lp_optimizer (e.g. switch_cost=0.02 for switch-cost version)

    Returns:
        total_df : concatenated pd.DataFrame of all 24h chunks
    """
    W = 24  # fixed 24-hour horizon
    total_df = []

    n = len(date_time_series)
    num_blocks = n // W

    for b in range(num_blocks):
        start_idx = b * W
        start_time = date_time_series[start_idx]
        print(f"Running 24h block {b+1}/{num_blocks} starting at {start_time}")

        df_block = lp_optimizer(
            date_time_index=start_time,
            frequency=frequency,
            components=components,
            elec_demand=elec_demand,
            gas_demand=gas_demand,
            pv_generated=pv_generated,
            **optimizer_kwargs,
        )
        total_df.append(df_block)

        # Propagate end-of-block SOC to components for next block (avoids infeasibility from wrong initial state)
        if "soc_elec_end" in df_block.columns and "soc_heat_end" in df_block.columns:
            status = df_block.get("status", pd.Series(dtype=object)).iloc[0]
            if status in ("optimal", "optimal_inaccurate"):
                battery = components.get("battery")
                buffer = components.get("buffer")
                if battery is not None:
                    battery.soc_elec = float(df_block["soc_elec_end"].iloc[-1])
                if buffer is not None:
                    buffer.soc_heat = float(df_block["soc_heat_end"].iloc[-1])

    return pd.concat(total_df)


# def run_sliding_24h_forecast_with_switch_cost(
#     date_time_series,
#     frequency,
#     components,
#     elec_demand,
#     gas_demand,
#     pv_generated,
#     lp_optimizer_with_switch_cost,
#     switch_cost=0.02,
# ):
#     """Convenience wrapper: same as run_sliding_24h_forecast with switch_cost passed to the optimizer."""
#     return run_sliding_24h_forecast(
#         date_time_series,
#         frequency,
#         components,
#         elec_demand,
#         gas_demand,
#         pv_generated,
#         lp_optimizer_with_switch_cost,
#         switch_cost=switch_cost,
#     )
