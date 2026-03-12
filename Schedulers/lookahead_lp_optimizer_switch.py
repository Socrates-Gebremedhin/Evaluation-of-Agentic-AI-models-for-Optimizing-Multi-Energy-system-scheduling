import io
import contextlib
import io
import contextlib

def optimize_24h_with_switching_cost(date_time_index, frequency, components, elec_demand, gas_demand, pv_generated,feed_in_price,gas_price,grid_price, switch_cost=10):
    """
    24-hour optimizer starting at `date_time_index`.

    Inputs:
      - date_time_index: single pandas.Timestamp (start of 24h horizon)
      - frequency: minutes per timestep (e.g. 60)
      - components: dict with keys "boiler","chp","ee","pv","battery","buffer" and optionally "prices"
      - elec_demand, gas_demand, pv_generated: pandas.Series indexed by a DatetimeIndex

    Returns:
      - pd.DataFrame with 24 rows indexed by timestamps
        Columns: boiler, Grid, chp_out_de, chp_out_ee, chp_out_charge, chp_out_heat,
                 pv_out_de, pv_out_ee, pv_out_charge, buffer_out, batt_out_de, batt_out_ee,
                 soc_elec, soc_heat, ee_heat, elec_price_Grid, gas_price_total, feed_in_revenue, status
    """
    import cvxpy as cp
    import numpy as np
    import pandas as pd
    import logging

    cp.settings.LOGGER.setLevel(logging.WARNING)

    W = 24
    time_delta = float(frequency) / 60.0  # hours

    # Prices
    #price_dict = components.get("prices", {}) if isinstance(components, dict) else {}
    #gas_price = price_dict.get("gas_price", 0.11)
    #grid_price = price_dict.get("grid_price", 0.30)
    #feed_in_price = price_dict.get("feed_in_price", 0.065)

    # Components
    boiler = components["boiler"]
    chp = components["chp"]
    ee = components["ee"]
    pv_comp = components.get("pv", None)
    battery = components["battery"]
    buffer = components["buffer"]

    # Slice data
    start_pos = elec_demand.index.get_loc(date_time_index)
    if start_pos + W > len(elec_demand):
        raise ValueError("Not enough data to cover full 24h horizon")

    elec_win = elec_demand.iloc[start_pos : start_pos + W].to_numpy(dtype=float)
    heat_win = gas_demand.iloc[start_pos : start_pos + W].to_numpy(dtype=float)
    pv_win = pv_generated.iloc[start_pos : start_pos + W].to_numpy(dtype=float)

    # Demand must be nonnegative (forecasts can be slightly negative); otherwise balance constraints are infeasible
    elec_win = np.maximum(elec_win, 0.0)
    heat_win = np.maximum(heat_win, 0.0)
    pv_win = np.maximum(pv_win, 0.0)

    # --- Variables ---
    chp_hd = cp.Variable(W, nonneg=True)
    boiler_hd = cp.Variable(W, nonneg=True)
    buffer_to_discharge_hd = cp.Variable(W, nonneg=True)
    ee_hd = cp.Variable(W, nonneg=True)

    pv_ed = cp.Variable(W, nonneg=True)
    battery_to_discharge_ed = cp.Variable(W, nonneg=True)
    chp_ed = cp.Variable(W, nonneg=True)
    grid_out = cp.Variable(W, nonneg=True)

    chp_batt = cp.Variable(W, nonneg=True)
    pv_batt = cp.Variable(W, nonneg=True)
    chp_buff = cp.Variable(W, nonneg=True)
    boiler_buff = cp.Variable(W, nonneg=True)

    chp_ee = cp.Variable(W, nonneg=True)
    pv_ee = cp.Variable(W, nonneg=True)
    batt_to_discharge_ee = cp.Variable(W, nonneg=True)

    batt_to_charge = cp.Variable(W, nonneg=True)
    buff_to_charge = cp.Variable(W, nonneg=True)
    ee_ed = cp.Variable(W, nonneg=True)

    batt_to_discharge_total = cp.Variable(W, nonneg=True)
    chp_heat_total = cp.Variable(W, nonneg=True)
    chp_elec_total = cp.Variable(W, nonneg=True)
    chp_in = cp.Variable(W, nonneg=True)

    soc_batt = cp.Variable(W + 1)
    soc_buff = cp.Variable(W + 1)
    pv_feed_in = cp.Variable(W, nonneg=True)
    chp_feed_in = cp.Variable(W, nonneg=True)

    # --- Switching cost binaries ---
    chp_on = cp.Variable(W, boolean=True)
    boiler_on = cp.Variable(W, boolean=True)

    chp_switch = cp.Variable(W, boolean=True)
    boiler_switch = cp.Variable(W, boolean=True)

    unmet_heat = cp.Variable(W, nonneg=True)
    unmet_elec = cp.Variable(W, nonneg=True)

    cons = []

    # Initial SOC
    init_soc_batt = min(max(float(battery.soc_elec), 0.0), battery.capacity)
    init_soc_buff = min(max(float(buffer.soc_heat), 0.0), buffer.capacity)
    cons += [soc_batt[0] == init_soc_batt]
    cons += [soc_buff[0] == init_soc_buff]

    # CHP ratios
    if chp.rated_input_heat <= 0:
        raise ValueError("CHP rated_input_heat must be >0")
    chp_heat_per_fuel = chp.rated_output_heat / chp.rated_input_heat
    chp_el_per_fuel = chp.rated_output_elec / chp.rated_input_heat

    # --- Hourly constraints ---
    for i in range(W):
        # Heat & electricity balance
        cons += [chp_hd[i] + boiler_hd[i] + buffer_to_discharge_hd[i] + ee_hd[i] + unmet_heat[i] == heat_win[i]]
        cons += [pv_ed[i] + battery_to_discharge_ed[i] + chp_ed[i] + grid_out[i] + unmet_elec[i] == elec_win[i]]

        # Linking flows
        cons += [chp_batt[i] + pv_batt[i] == batt_to_charge[i]]
        cons += [chp_buff[i] + boiler_buff[i] == buff_to_charge[i]]
        cons += [pv_ee[i] + chp_ee[i] + batt_to_discharge_ee[i] == ee_ed[i]]

        # PV limit
        pv_avail_i = max(0,float(min(pv_win[i], pv_comp.capacity)) if pv_comp else float(pv_win[i]))
        cons += [pv_ed[i] + pv_ee[i] + pv_batt[i] <= pv_avail_i]

        # Boiler limits
        cons += [
            boiler_hd[i] + boiler_buff[i] <= boiler_on[i] * boiler.rated_output_heat,
            boiler_hd[i] + boiler_buff[i] >= 0.2 * boiler.rated_output_heat * boiler_on[i]
        ]

        # EE limits
        cons += [ee_ed[i] <= ee.rated_input_elec]
        cons += [ee_hd[i] == ee_ed[i] * (ee.rated_output_heat / ee.rated_input_elec)]

        # Battery discharge
        cons += [batt_to_discharge_total[i] == batt_to_discharge_ee[i] + battery_to_discharge_ed[i]]
        cons += [batt_to_discharge_total[i] <= soc_batt[i] * battery.eta_discharge_elec]
        cons += [batt_to_discharge_total[i] <= battery.max_discharge_rate_elec * time_delta * battery.eta_discharge_elec]
        cons += [batt_to_charge[i] <= (battery.capacity - soc_batt[i]) * battery.eta_charge_elec]
        cons += [batt_to_charge[i] <= battery.max_charge_rate_elec * time_delta * battery.eta_charge_elec]

        # Buffer discharge & charge
        cons += [buffer_to_discharge_hd[i] <= soc_buff[i] * buffer.eta_discharge_heat]
        #cons += [buffer_to_discharge_hd[i] <= buffer.discharge_rate_heat * time_delta * buffer.eta_discharge_heat]
        cons += [buff_to_charge[i] <= (buffer.capacity - soc_buff[i]) * buffer.eta_charge_heat]
        #cons += [buff_to_charge[i] <= buffer.charge_rate_heat * time_delta * buffer.eta_charge_heat]

        # CHP coupling
        cons += [chp_heat_total[i] == chp_in[i] * chp_heat_per_fuel]
        cons += [chp_elec_total[i] == chp_in[i] * chp_el_per_fuel]
        cons += [chp_heat_total[i] >= chp_hd[i] + chp_buff[i]]
        cons += [chp_elec_total[i] >= chp_ed[i] + chp_ee[i] + chp_batt[i]]
        cons += [chp_heat_total[i] <= chp.rated_output_heat]
        cons += [chp_elec_total[i] <= chp.rated_output_elec]
        cons += [
            chp_in[i] <= chp_on[i] * chp.rated_input_heat,
            chp_in[i] >= 0.6 * chp.rated_input_heat * chp_on[i]
        ]

        # SOC dynamics
        cons += [soc_batt[i+1] == soc_batt[i] + batt_to_charge[i] * battery.eta_charge_elec
                 - batt_to_discharge_total[i]/battery.eta_discharge_elec]
        cons += [soc_buff[i+1] == soc_buff[i] + buff_to_charge[i] * buffer.eta_charge_heat
                 - buffer_to_discharge_hd[i]/buffer.eta_discharge_heat]
        cons += [soc_batt[i+1] >=0, soc_batt[i+1] <= battery.capacity]
        cons += [soc_buff[i+1] >=0, soc_buff[i+1] <= buffer.capacity]

        # feed in dynamics
        cons += [
        pv_feed_in[i] <= pv_avail_i - (pv_ed[i] + pv_ee[i] + pv_batt[i]),
        chp_feed_in[i] <= chp_elec_total[i] - (chp_ed[i] + chp_ee[i] + chp_batt[i]),
    ]
        # ---------- EXACT switching definition ----------
        if i == 0:
            cons += [
                chp_switch[i] >= chp_on[i],
                boiler_switch[i] >= boiler_on[i]
            ]
        else:
            cons += [
                chp_switch[i] >= chp_on[i] - chp_on[i-1],
                boiler_switch[i] >= boiler_on[i] - boiler_on[i-1]
            ]

    # --- Objective ---
    big_penalty = 1e6
    chp_fuel_cost = cp.sum(chp_in) * time_delta * gas_price
    boiler_fuel_input = cp.sum(boiler_hd + boiler_buff) * (boiler.rated_input_heat / boiler.rated_output_heat)
    boiler_cost = boiler_fuel_input * time_delta * gas_price
    grid_cost = cp.sum(grid_out) * time_delta * grid_price

    # Feed-in revenue
    # pv_excess_expr = cp.sum([cp.pos(pv_avail_i - (pv_ed[i] + pv_ee[i] + pv_batt[i]))
    #                          for i, pv_avail_i in enumerate(
    #                              [float(min(pv_win[j], pv_comp.capacity) if pv_comp else pv_win[j])
    #                               for j in range(W)]
    #                          )])
    # chp_excess_expr = cp.sum([cp.pos(chp_elec_total[i] - (chp_ed[i] + chp_ee[i] + chp_batt[i])) for i in range(W)])
    feed_in_revenue = feed_in_price * time_delta * cp.sum(pv_feed_in + chp_feed_in)
    total_switch_cost = switch_cost * (cp.sum(chp_switch) + cp.sum(boiler_switch))

    unmet_penalty = big_penalty * (cp.sum(unmet_heat) + cp.sum(unmet_elec))
    objective = cp.Minimize(chp_fuel_cost + boiler_cost + grid_cost - feed_in_revenue + unmet_penalty + total_switch_cost)

    prob = cp.Problem(objective, cons)

    # --- Solve ---
    status = None
    solver_used = None
    solver_verbose = ""

    # all_solvers = [cp.GLPK_MI,cp.CLARABEL,cp.CVXOPT, cp.ECOS, cp.GLPK, cp.OSQP, cp.SCIPY, cp.SCS,cp.ECOS_BB]

    # for solver in all_solvers:
    #     try:
    #         f = io.StringIO()
    #         with contextlib.redirect_stdout(f):
    #             prob.solve(solver=solver, verbose=True)  # always verbose
    #         status = prob.status
    #         solver_used = solver.__name__ if hasattr(solver, "__name__") else str(solver)
    #         solver_verbose = f.getvalue()
    #         if status in ("optimal", "optimal_inaccurate"):
    #             break  # stop at first successful solver
    #         else:
    #             continue
    #     except Exception as e:
    #         status = "error"
    #         solver_used = solver.__name__ if hasattr(solver, "__name__") else str(solver)
    #         solver_verbose = str(e)
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):  # capture verbose output
            prob.solve(solver=cp.GUROBI, verbose=False,reoptimize=True)
        status = prob.status
        solver_used = "GUROBI"
        solver_verbose = f.getvalue()
    except Exception as e:
        status = "error"
        solver_used = "GUROBI"
        solver_verbose = str(e)
    # --- Extract safe arrays ---
    def safe_arr(var, length):
        if var is None or getattr(var, "value", None) is None:
            return np.zeros(length, dtype=float)
        arr = np.array(var.value, dtype=float)
        if arr.shape == (): return np.full(length, float(arr))
        if arr.size >= length: return arr[:length]
        out = np.zeros(length, dtype=float)
        out[:arr.size] = arr
        return out

    idx = elec_demand.index[start_pos : start_pos + W]

    boiler_arr = safe_arr(boiler_hd, W)
    grid_arr = safe_arr(grid_out, W)
    chp_ed_arr = safe_arr(chp_ed, W)
    chp_ee_arr = safe_arr(chp_ee, W)
    chp_batt_arr = safe_arr(chp_batt, W)
    chp_hd_arr = safe_arr(chp_hd, W)
    pv_ed_arr = safe_arr(pv_ed, W)
    pv_ee_arr = safe_arr(pv_ee, W)
    pv_batt_arr = safe_arr(pv_batt, W)
    buffer_out_arr = safe_arr(buffer_to_discharge_hd, W)
    batt_de_arr = safe_arr(battery_to_discharge_ed, W)
    batt_ee_arr = safe_arr(batt_to_discharge_ee, W)
    ee_hd_arr = safe_arr(ee_hd, W)
    soc_batt_arr = safe_arr(soc_batt, W + 1)
    soc_buff_arr = safe_arr(soc_buff, W + 1)

    # --- Hourly prices ---
    elec_price_grid = grid_arr * grid_price * time_delta
    b_in = (boiler_arr + safe_arr(boiler_buff, W)) * (boiler.rated_input_heat/boiler.rated_output_heat)
    price_boiler = b_in * time_delta * gas_price
    
    gas_price_total = ((chp_in.value if chp_in.value is not None else np.zeros(W)) * gas_price * time_delta
                       + ((boiler_arr + safe_arr(boiler_buff, W)) * (boiler.rated_input_heat/boiler.rated_output_heat) * gas_price * time_delta))
    pv_feedin = np.maximum([float(min(pv_win[i], pv_comp.capacity)) if pv_comp else pv_win[i]
                            for i in range(W)] - (pv_ed_arr + pv_ee_arr + pv_batt_arr), 0)
    chp_feedin = np.maximum(chp_elec_total.value - (chp_ed_arr + chp_ee_arr + chp_batt_arr), 0) if chp_elec_total.value is not None else np.zeros(W)
    feed_in_revenue_arr = feed_in_price * time_delta * (pv_feedin + chp_feedin)
    price_chp = (chp_in.value if chp_in.value is not None else np.zeros(W)) * gas_price * time_delta
    price_elec_total = elec_price_grid - feed_in_revenue_arr
    

    # --- Build DataFrame ---
    out_df = pd.DataFrame({
        "boiler": boiler_arr,
        "boiler_to_buffer": safe_arr(boiler_buff,W),
        "boiler_out_total":safe_arr(boiler_hd,W) + safe_arr(boiler_buff,W),
        "boiler_in_total":b_in,
        "Grid": grid_arr,
        "chp_out_de": chp_ed_arr,
        "chp_out_ee": chp_ee_arr,
        "chp_out_charge": chp_batt_arr,
        "chp_out_heat": safe_arr(chp_heat_total,W),
        "chp_out_hd":chp_hd_arr,
        "chp_out_buff":safe_arr(chp_buff,W),
        "chp_in": safe_arr(chp_in,W),
        "chp_total_elec": safe_arr(chp_elec_total,W),
        "pv_out_de":safe_arr(pv_ed,W),
        "pv_out_ee":safe_arr(pv_ee,W),
        "pv_out_charge":safe_arr(pv_batt,W),
        "buffer_out":buffer_out_arr,
        "total_stored_heat":safe_arr(buff_to_charge,W),
        "actual_stored_heat":safe_arr(buff_to_charge,W),
        "batt_out_de":batt_de_arr,
        "batt_out_ee":batt_ee_arr,
        "total_battery_discharge":safe_arr(batt_to_discharge_total,W),
        "soc_elec":soc_batt_arr[:-1],
        "actual_charged_batt":safe_arr(batt_to_charge,W),
        "soc_heat": soc_buff_arr[:-1],
        "ee_in_elec_actual":safe_arr(ee_ed,W),
        "ee_heat":ee_hd_arr,
        "gas_price_boiler": price_boiler,
        "elec_price_Grid": elec_price_grid,
        "gas_price_chp": price_chp,
        "gas_price_total": gas_price_total,
        "elec_price_total": price_elec_total,
        "status": np.full(W, status if status is not None else "unknown"),
        "solver_verbose": np.full(W, solver_verbose),
        "solver_used": np.full(W, solver_used if solver_used else "unknown"),
        "chp_on": safe_arr(chp_on,W),
        "boiler_on": safe_arr(boiler_on,W),
        "chp_switch": safe_arr(chp_switch,W),
        "boiler_switch": safe_arr(boiler_switch,W),
        "soc_elec_end": np.full(W, soc_batt_arr[-1]),
        "soc_heat_end": np.full(W, soc_buff_arr[-1]),
    }, index=idx)



    return out_df
