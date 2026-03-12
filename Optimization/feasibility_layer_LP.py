def feasibility_layer_LP(df, elec_real, heat_real, pv_real, components, t=0, lambda_res=100.0,gas_price=0.11,grid_price=0.3,feed_in_price=0.065):  
    import cvxpy as cp
    import numpy as np
    import io
    import contextlib

    # ======================
    # Components
    # ======================
    boiler = components["boiler"]
    chp = components["chp"]
    ee = components["ee"]
    battery = components["battery"]
    buffer = components["buffer"]
    pv_comp = components["pv"]

    plan = df.iloc[t]

    # Clip demand to nonnegative (real/forecast can be slightly negative → would make balance infeasible)
    elec_d = max(0.0, float(elec_real.iloc[t]))
    heat_d = max(0.0, float(heat_real.iloc[t]))
    pv_avail = max(0.0, float(pv_real.iloc[t]))

    # SOC source for this LP step:
    # - if the caller passes a chained trajectory via "previous_soc"/"previous_soc_buff"
    #   on the plan row, use those (LP step t-1 result);
    # - otherwise fall back to the plan's own soc_elec/soc_heat.
    soc_elec = float(plan.get("previous_soc", plan["soc_elec"]))
    soc_heat = float(plan.get("previous_soc_buff", plan["soc_heat"]))

    time_delta = 1.0  # 1 hour

    # ======================
    # Variables (scalar)
    # ======================
    chp_hd = cp.Variable(nonneg=True)
    boiler_hd = cp.Variable(nonneg=True)
    buffer_to_discharge_hd = cp.Variable(nonneg=True)
    ee_hd = cp.Variable(nonneg=True)

    pv_ed = cp.Variable(nonneg=True)
    battery_to_discharge_ed = cp.Variable(nonneg=True)
    chp_ed = cp.Variable(nonneg=True)
    grid_out = cp.Variable(nonneg=True)

    chp_batt = cp.Variable(nonneg=True)
    pv_batt = cp.Variable(nonneg=True)
    chp_buff = cp.Variable(nonneg=True)
    boiler_buff = cp.Variable(nonneg=True)

    chp_ee = cp.Variable(nonneg=True)
    pv_ee = cp.Variable(nonneg=True)
    batt_to_discharge_ee = cp.Variable(nonneg=True)

    batt_to_charge = cp.Variable(nonneg=True)
    buff_to_charge = cp.Variable(nonneg=True)
    ee_ed = cp.Variable(nonneg=True)

    batt_to_discharge_total = cp.Variable(nonneg=True)
    chp_heat_total = cp.Variable(nonneg=True)
    chp_elec_total = cp.Variable(nonneg=True)
    chp_in = cp.Variable(nonneg=True)

    soc_batt_next = cp.Variable()
    soc_buff_next = cp.Variable()

    pv_feed_in = cp.Variable(nonneg=True)
    chp_feed_in = cp.Variable(nonneg=True)

    # Binary on/off for min-load (CHP 60–100%, boiler 20–100%)
    chp_on = cp.Variable(boolean=True)
    boiler_on = cp.Variable(boolean=True)

    cons = []

    # ======================
    # Balances (NO unmet)
    # ======================
    cons += [
        chp_hd + boiler_hd + buffer_to_discharge_hd + ee_hd == heat_d,
        pv_ed + battery_to_discharge_ed + chp_ed + grid_out == elec_d
    ]

    # ======================
    # Linking flows
    # ======================
    cons += [
        chp_batt + pv_batt == batt_to_charge,
        chp_buff + boiler_buff == buff_to_charge,
        pv_ee + chp_ee + batt_to_discharge_ee == ee_ed
    ]

    # ======================
    # PV limit
    # ======================
    pv_cap = min(pv_avail, pv_comp.capacity) if pv_comp else pv_avail
    cons += [pv_ed + pv_ee + pv_batt <= pv_cap]

    # ======================
    # Boiler + min-load: either 0 or 20–100% of rated heat output
    # ======================
    boiler_out_total = boiler_hd + boiler_buff
    boiler_min_out = 0.2 * boiler.rated_output_heat
    cons += [
        boiler_out_total <= boiler.rated_output_heat * boiler_on,
        boiler_out_total >= boiler_min_out * boiler_on,
    ]

    # ======================
    # EE
    # ======================
    cons += [
        ee_ed <= ee.rated_input_elec,
        ee_hd == ee_ed * (ee.rated_output_heat / ee.rated_input_elec)
    ]

    # ======================
    # Battery
    # ======================
    cons += [
        batt_to_discharge_total == batt_to_discharge_ee + battery_to_discharge_ed,
        batt_to_discharge_total <= soc_elec * battery.eta_discharge_elec,
        batt_to_discharge_total <= battery.max_discharge_rate_elec * time_delta * battery.eta_discharge_elec,
        batt_to_charge <= (battery.capacity - soc_elec) * battery.eta_charge_elec
    ]

    # ======================
    # Buffer
    # ======================
    cons += [
        buffer_to_discharge_hd <= soc_heat * buffer.eta_discharge_heat,
        buff_to_charge <= (buffer.capacity - soc_heat) * buffer.eta_charge_heat
    ]

    # ======================
    # CHP coupling + min-load: either 0 or 60–100% of rated input
    # ======================
    chp_heat_per_fuel = chp.rated_output_heat / chp.rated_input_heat
    chp_el_per_fuel = chp.rated_output_elec / chp.rated_input_heat
    chp_min_in = 0.6 * chp.rated_input_heat

    cons += [
        chp_in <= chp.rated_input_heat * chp_on,
        chp_in >= chp_min_in * chp_on,
        chp_heat_total == chp_in * chp_heat_per_fuel,
        chp_elec_total == chp_in * chp_el_per_fuel,
        chp_heat_total >= chp_hd + chp_buff,
        chp_elec_total >= chp_ed + chp_ee + chp_batt
    ]

    # ======================
    # SOC update
    # ======================
    cons += [
        soc_batt_next == soc_elec
            + batt_to_charge * battery.eta_charge_elec
            - batt_to_discharge_total / battery.eta_discharge_elec,

        soc_buff_next == soc_heat
            + buff_to_charge * buffer.eta_charge_heat
            - buffer_to_discharge_hd / buffer.eta_discharge_heat,

        soc_batt_next >= 0, soc_batt_next <= battery.capacity,
        soc_buff_next >= 0, soc_buff_next <= buffer.capacity
    ]

    # ======================
    # Feed-in
    # ======================
    cons += [
        pv_feed_in <= pv_cap - (pv_ed + pv_ee + pv_batt),
        chp_feed_in <= chp_elec_total - (chp_ed + chp_ee + chp_batt)
    ]

    # ======================
    # Objective (use fallback prices if plan has NaN, e.g. from infeasible 24h block)
    # ======================
    def _safe_plan(key, default):
        v = plan.get(key, default)
        return float(v) if np.isfinite(v) else float(default)

    price_chp = _safe_plan("gas_price_chp", gas_price)
    price_boiler = _safe_plan("gas_price_boiler", gas_price)
    price_grid = _safe_plan("elec_price_Grid", grid_price)

    cost = (
        chp_in * price_chp
        + boiler_hd * price_boiler
        + grid_out * price_grid
    )

    # Resemble: use 0 for plan entries that are NaN so we don't corrupt the objective
    resemble = lambda_res * cp.norm1(cp.hstack([
        chp_hd - _safe_plan("chp_out_hd", 0),
        boiler_hd - _safe_plan("boiler", 0),
        buffer_to_discharge_hd - _safe_plan("buffer_out", 0),
        battery_to_discharge_ed - _safe_plan("batt_out_de", 0),
        grid_out - _safe_plan("Grid", 0),
        pv_ed - _safe_plan("pv_out_de", 0),
        pv_ee - _safe_plan("pv_out_ee", 0),
        pv_batt - _safe_plan("pv_out_charge", 0),
    ]))

    prob = cp.Problem(cp.Minimize(cost + resemble), cons)

    # Solve like the 24h LP: capture status and solver output, do not raise.
    status = None
    solver_used = "GUROBI"
    solver_verbose = ""
    try:
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            prob.solve(solver=cp.GUROBI, verbose=False,reoptimize=True)
        status = prob.status
        solver_verbose = f.getvalue()
    except Exception as e:
        status = "error"
        solver_verbose = str(e)

    # Safe extraction of scalar variable values; use sensible defaults if no solution.
    def safe_val(var, default=0.0):
        v = getattr(var, "value", None)
        return float(v) if v is not None else float(default)

    return {

    # BOILER
    "boiler": safe_val(boiler_hd, 0.0),
    "boiler_to_buffer": safe_val(boiler_buff, 0.0),
    "boiler_out_total": safe_val(boiler_hd, 0.0) + safe_val(boiler_buff, 0.0),
    "boiler_in_total": (safe_val(boiler_hd, 0.0) + safe_val(boiler_buff, 0.0))
                       * boiler.rated_input_heat / boiler.rated_output_heat,

    # GRID
    "Grid": safe_val(grid_out, 0.0),

    # CHP
    "chp_out_de": safe_val(chp_ed, 0.0),
    "chp_out_ee": safe_val(chp_ee, 0.0),
    "chp_out_charge": safe_val(chp_batt, 0.0),
    "chp_out_heat": safe_val(chp_heat_total, 0.0),
    "chp_out_hd": safe_val(chp_hd, 0.0),
    "chp_out_buff": safe_val(chp_buff, 0.0),
    "chp_in": safe_val(chp_in, 0.0),
    "chp_total_elec": safe_val(chp_elec_total, 0.0),

    # PV
    "pv_out_de": safe_val(pv_ed, 0.0),
    "pv_out_ee": safe_val(pv_ee, 0.0),
    "pv_out_charge": safe_val(pv_batt, 0.0),

    # HEAT BUFFER
    "buffer_out": safe_val(buffer_to_discharge_hd, 0.0),
    "total_stored_heat": safe_val(buff_to_charge, 0.0),
    "actual_stored_heat": safe_val(buff_to_charge, 0.0),
    "soc_heat": safe_val(soc_buff_next, soc_heat),

    # BATTERY
    "batt_out_de": safe_val(battery_to_discharge_ed, 0.0),
    "batt_out_ee": safe_val(batt_to_discharge_ee, 0.0),
    "total_battery_discharge": safe_val(batt_to_discharge_total, 0.0),
    "actual_charged_batt": safe_val(batt_to_charge, 0.0),
    "soc_elec": safe_val(soc_batt_next, soc_elec),

    # ELECTRIC HEATER
    "ee_in_elec_actual": safe_val(ee_ed, 0.0),
    "ee_heat": safe_val(ee_hd, 0.0),

    
    "gas_price_boiler": (safe_val(boiler_hd, 0.0) * boiler.rated_input_heat / boiler.rated_output_heat) * gas_price,
    "elec_price_Grid": safe_val(grid_out, 0.0) * grid_price,
    "gas_price_chp": safe_val(chp_in, 0.0) * gas_price,
    "gas_price_total": (
        safe_val(chp_in, 0.0) * gas_price
        + (safe_val(boiler_hd, 0.0) * boiler.rated_input_heat / boiler.rated_output_heat) * gas_price
    ),
    "elec_price_total": safe_val(grid_out, 0.0) * grid_price,

    # META
    "status": status,
    "solver_verbose": solver_verbose,
    "solver_used": solver_used,
}

