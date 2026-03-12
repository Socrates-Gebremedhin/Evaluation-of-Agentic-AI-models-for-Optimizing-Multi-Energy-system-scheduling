def feasibility_layer_RL(
    row, elec_real, heat_real, pv_real, components,
    gas_price=0.11, grid_price=0.30, feed_in_price=0.065,
    lambda_res=100.0,
):
    import cvxpy as cp
    import numpy as np
    import io
    import contextlib

    chp = components["chp"]
    boiler = components["boiler"]
    battery = components["battery"]
    buffer = components["buffer"]
    ee = components["ee"]

    time_delta = 1.0

    elec_d = float(elec_real)
    heat_d = float(heat_real)
    pv_cap = max(0.0, float(pv_real))

    # SOC source priority:
    # 1) "previous_soc"/"previous_soc_buff" from a chained LP trajectory (row[t-1])
    # 2) "soc_elec_start"/"soc_heat_start" from the env (start-of-step SOC)
    # 3) Fallback to row["soc_elec"]/row["soc_heat"] if nothing else is present
    soc_elec = float(row.get("previous_soc", row.get("soc_elec_start", row["soc_elec"])))
    soc_heat = float(row.get("previous_soc_buff", row.get("soc_heat_start", row["soc_heat"])))

    # ======================
    # Variables
    # ======================
    chp_in = cp.Variable(nonneg=True)
    chp_out_heat = cp.Variable(nonneg=True)
    chp_out_elec = cp.Variable(nonneg=True)

    boiler_out_dh = cp.Variable(nonneg=True)
    boiler_to_buffer = cp.Variable(nonneg=True)

    batt_out_de = cp.Variable(nonneg=True)
    batt_charge = cp.Variable(nonneg=True)

    buff_out = cp.Variable(nonneg=True)
    buff_charge = cp.Variable(nonneg=True)

    ee_in_elec = cp.Variable(nonneg=True)
    ee_out_heat = cp.Variable(nonneg=True)

    grid = cp.Variable(nonneg=True)

    pv_used = cp.Variable(nonneg=True)
    pv_feed_in = cp.Variable(nonneg=True)

    soc_batt_next = cp.Variable()
    soc_buff_next = cp.Variable()

    # Binary on/off for min-load constraints (CHP 60–100%, boiler 20–100%)
    chp_on = cp.Variable(boolean=True)
    boiler_on = cp.Variable(boolean=True)

    cons = []

    # ======================
    # CHP coupling + min-load: either 0 or 60–100% of rated input
    # ======================
    chp_hpf = chp.rated_output_heat / chp.rated_input_heat
    chp_epf = chp.rated_output_elec / chp.rated_input_heat
    chp_min_in = 0.6 * chp.rated_input_heat

    cons += [
        chp_in <= chp.rated_input_heat * chp_on,
        chp_in >= chp_min_in * chp_on,
        chp_out_heat == chp_in * chp_hpf,
        chp_out_elec == chp_in * chp_epf,
    ]

    # ======================
    # Heat balance (no slack here: strictly meet real demand)
    # ======================
    cons += [
        chp_out_heat + ee_out_heat + buff_out + boiler_out_dh == heat_d
    ]

    # ======================
    # Electricity balance
    # ======================
    cons += [
        chp_out_elec + pv_used + batt_out_de + grid
        == elec_d + ee_in_elec + batt_charge
    ]

    # ======================
    # PV limit (real PV)
    # ======================
    cons += [pv_used + pv_feed_in <= pv_cap]

    # ======================
    # EE
    # ======================
    cons += [
        ee_in_elec <= ee.rated_input_elec,
        ee_out_heat == ee_in_elec * (ee.rated_output_heat / ee.rated_input_elec),
    ]

    # ======================
    # Battery
    # ======================
    cons += [
        batt_out_de <= battery.max_discharge_rate_elec * time_delta,
        batt_out_de <= soc_elec * battery.eta_discharge_elec,

        batt_charge <= battery.max_charge_rate_elec * time_delta,
        batt_charge <= (battery.capacity - soc_elec) / battery.eta_charge_elec,
    ]

    # ======================
    # Buffer
    # ======================
    cons += [
        buff_out <= soc_heat * buffer.eta_discharge_heat,
        buff_charge <= (buffer.capacity - soc_heat) / buffer.eta_charge_heat,
    ]

    # Link boiler_to_buffer to buffer charging (simple version: all buffer charge from boiler)
    cons += [
        buff_charge == boiler_to_buffer
    ]

    # ======================
    # Boiler + min-load: either 0 or 20–100% of rated heat output
    # ======================
    boiler_out_total_var = boiler_out_dh + boiler_to_buffer
    boiler_min_out = 0.2 * boiler.rated_output_heat

    cons += [
        boiler_out_total_var <= boiler.rated_output_heat * boiler_on,
        boiler_out_total_var >= boiler_min_out * boiler_on,
    ]

    # ======================
    # SOC updates
    # ======================
    cons += [
        soc_batt_next ==
        soc_elec
        + batt_charge * battery.eta_charge_elec
        - batt_out_de / battery.eta_discharge_elec,

        soc_buff_next ==
        soc_heat
        + buff_charge * buffer.eta_charge_heat
        - buff_out / buffer.eta_discharge_heat,

        soc_batt_next >= 0,
        soc_batt_next <= battery.capacity,
        soc_buff_next >= 0,
        soc_buff_next <= buffer.capacity,
    ]

    # ======================
    # Objective: (real) cost + deviation from RL
    # ======================
    # Fuel inputs for cost (total boiler output = direct heat + to buffer)
    boiler_out_total_lp = boiler_out_dh + boiler_to_buffer
    boiler_in = boiler_out_total_lp * boiler.rated_input_heat / boiler.rated_output_heat

    # Economic cost (one hour)
    cost = (
        gas_price * (chp_in + boiler_in)
        + grid_price * grid
        - feed_in_price * pv_feed_in
    )

    # Deviation from RL actions / control decisions (not balance outcomes like grid, pv_used)
    resemble = lambda_res * cp.norm1(cp.hstack([
        chp_in - row["chp_in"],
        boiler_out_total_lp - row["boiler_out_total"],
        batt_out_de - row["batt_out_de"],
        buff_out - row["buffer_out"],
        boiler_to_buffer - row["boiler_to_buffer"],
    ]))

    prob = cp.Problem(cp.Minimize(cost + resemble), cons)

    # Solve like feasibility_layer_LP / 24h LP: capture status, do not raise.
    status = None
    solver_used = "GUROBI"
    solver_verbose = ""
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            prob.solve(solver=cp.GUROBI, verbose=False)
        status = prob.status
        solver_verbose = f.getvalue()
    except Exception as e:
        status = "error"
        solver_verbose = str(e)

    def safe_val(var, default=0.0):
        v = getattr(var, "value", None)
        return float(v) if v is not None else float(default)

    b_dh = safe_val(boiler_out_dh, 0.0)
    b_buf = safe_val(boiler_to_buffer, 0.0)
    b_in = (b_dh + b_buf) * boiler.rated_input_heat / boiler.rated_output_heat
    g = safe_val(grid, 0.0)
    chp_in_v = safe_val(chp_in, 0.0)

    return {
        "boiler": b_dh,
        "boiler_to_buffer": b_buf,
        "boiler_out_total": b_dh + b_buf,
        "boiler_in_total": b_in,
        "Grid": g,
        "chp_out_de": safe_val(chp_out_elec, 0.0),
        "chp_out_heat": safe_val(chp_out_heat, 0.0),
        "chp_out_hd": safe_val(chp_out_heat, 0.0),
        "chp_in": chp_in_v,
        "chp_total_elec": safe_val(chp_out_elec, 0.0),
        "pv_out_de": safe_val(pv_used, 0.0),
        "buffer_out": safe_val(buff_out, 0.0),
        "total_stored_heat": safe_val(buff_charge, 0.0),
        "actual_stored_heat": safe_val(buff_charge, 0.0),
        "batt_out_de": safe_val(batt_out_de, 0.0),
        "total_battery_discharge": safe_val(batt_out_de, 0.0),
        "soc_elec": safe_val(soc_batt_next, soc_elec),
        "actual_charged_batt": safe_val(batt_charge, 0.0),
        "soc_heat": safe_val(soc_buff_next, soc_heat),
        "soc_elec_start": float(soc_elec),
        "soc_heat_start": float(soc_heat),
        "ee_in_elec_actual": safe_val(ee_in_elec, 0.0),
        "ee_heat": safe_val(ee_out_heat, 0.0),
        "gas_price_boiler": b_in * gas_price,
        "elec_price_Grid": g * grid_price,
        "gas_price_chp": chp_in_v * gas_price,
        "gas_price_total": (chp_in_v + b_in) * gas_price,
        "elec_price_total": g * grid_price,
        "status": status,
        "solver_used": solver_used,
        "solver_verbose": solver_verbose,
    }