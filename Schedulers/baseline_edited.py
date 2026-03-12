"""
Baseline scheduler: one-timestep dispatch for heat and electricity.
Priority: electricity PV -> CHP -> Battery -> Grid; heat EE -> CHP -> buffer -> boiler.
Uses a draft pass to determine CHP proportions (for min-load), then actual pass with battery reset.
"""

CHP_MIN_LOAD_RATIO = 0.6
BOILER_MIN_LOAD_RATIO = 0.2


def _draft_chp_proportions(
    chp, pv, battery, ee,
    heat_demand, elec_demand, pv_gen, charge_value,
):
    """
    Draft pass: run electricity and heat balance with battery mutations to get
    CHP fuel proportions (demand vs EE vs charge vs heat). CHP min load depends
    on total CHP fuel. Mutates battery; caller must reset battery.soc_elec before actual pass.
    Returns dict with: re_de, re_ee, pv_out_de, pv_ex_de, pv_exc_ee, ee_in_elec_req,
    chp_in_de_ef, chp_in_ee_ef, chp_in_charge_ef, tota_chp_in_ef, chp_in_heat_hf, ee_out_heat_actual_ef.
    """
    # Electricity demand (after PV)
    pv_out_de = min(pv_gen, elec_demand)
    pv_ex_de = max(0, pv_gen - pv_out_de)
    re_de = elec_demand - pv_out_de

    chp_out_de_ef = min(chp.rated_output_elec, re_de)
    chp_in_de_ef = chp.get_input_heat_frm_elec(chp_out_de_ef)
    re_de2 = re_de - chp_out_de_ef
    battery.discharge_elec(re_de2)

    # EE electrical need and supply
    ee_out_heat_in = min(ee.rated_output_heat, heat_demand)
    ee_in_elec_req = ee.get_input_elec(ee_out_heat_in)
    pv_out_ee = min(pv_ex_de, ee_in_elec_req)
    pv_exc_ee = max(0, pv_ex_de - pv_out_ee)
    re_ee = ee_in_elec_req - pv_out_ee

    chp_out_ee_ef = min(chp.rated_output_elec - chp_out_de_ef, re_ee)
    chp_in_ee_ef = chp.get_input_heat_frm_elec(chp_out_ee_ef)
    re_ee2 = re_ee - chp_out_ee_ef
    batt_out_ee_ef = battery.discharge_elec(re_ee2)
    re_ee3 = re_ee2 - batt_out_ee_ef
    ee_in_elec_actual_ef = ee_in_elec_req - re_ee3
    ee_out_heat_actual_ef = ee.get_output_heat(ee_in_elec_actual_ef)

    # Battery charge (draft)
    amount_to_charge = max(0, charge_value - battery.soc_elec)
    pv_out_charge = min(pv_exc_ee, amount_to_charge, battery.max_charge_rate_elec)
    re_charge1 = amount_to_charge - pv_out_charge
    chp_out_charge_ef = min(
        chp.rated_output_elec - chp_out_de_ef - chp_out_ee_ef,
        re_charge1,
        battery.max_charge_rate_elec - pv_out_charge,
    )
    chp_in_charge_ef = chp.get_input_heat_frm_elec(chp_out_charge_ef)
    re_charge2 = re_charge1 - chp_out_charge_ef
    actual_amount_to_charge = amount_to_charge - re_charge2
    battery.charge_elec(actual_amount_to_charge)

    tota_chp_in_ef = (
        chp_in_de_ef + chp_in_ee_ef + chp_in_charge_ef
        if (chp_in_de_ef + chp_in_ee_ef + chp_in_charge_ef) >= CHP_MIN_LOAD_RATIO * chp.rated_input_heat
        else 0
    )

    # Heat demand (CHP share for heat)
    re_he = heat_demand - ee_out_heat_actual_ef
    chp_out_heat_hf = min(chp.rated_output_heat, re_he)
    chp_in_heat_hf = (
        chp.get_input_heat_frm_heat(chp_out_heat_hf)
        if chp.get_input_heat_frm_heat(chp_out_heat_hf) >= CHP_MIN_LOAD_RATIO * chp.rated_input_heat
        else 0
    )

    return {
        "re_de": re_de,
        "re_ee": re_ee,
        "pv_out_de": pv_out_de,
        "pv_out_ee": pv_out_ee,
        "pv_ex_de": pv_ex_de,
        "pv_exc_ee": pv_exc_ee,
        "ee_in_elec_req": ee_in_elec_req,
        "chp_in_de_ef": chp_in_de_ef,
        "chp_in_ee_ef": chp_in_ee_ef,
        "chp_in_charge_ef": chp_in_charge_ef,
        "tota_chp_in_ef": tota_chp_in_ef,
        "chp_in_heat_hf": chp_in_heat_hf,
        "ee_out_heat_actual_ef": ee_out_heat_actual_ef,
    }


def _actual_chp_split(chp, draft):
    """
    From draft CHP fuel proportions, compute actual CHP input and outputs
    (respecting min load). Returns actual_chp_in and CHP in/out for elec and heat.
    """
    tota = draft["tota_chp_in_ef"]
    chp_heat = draft["chp_in_heat_hf"]
    actual_chp_in = max(tota, chp_heat)

    if tota != 0:
        r = actual_chp_in / tota
        chp_in_de = draft["chp_in_de_ef"] * r
        chp_in_ee = draft["chp_in_ee_ef"] * r
        chp_in_charge = draft["chp_in_charge_ef"] * r
    else:
        third = actual_chp_in / 3
        chp_in_de = chp_in_ee = chp_in_charge = third

    eta_elec = chp.rated_output_elec / chp.rated_input_heat
    eta_heat = chp.rated_output_heat / chp.rated_input_heat
    chp_out_de = chp_in_de * eta_elec
    chp_out_ee = chp_in_ee * eta_elec
    chp_out_charge = chp_in_charge * eta_elec
    chp_out_heat = actual_chp_in * eta_heat

    return {
        "actual_chp_in": actual_chp_in,
        "chp_in_de": chp_in_de,
        "chp_in_ee": chp_in_ee,
        "chp_in_charge": chp_in_charge,
        "chp_out_de": chp_out_de,
        "chp_out_ee": chp_out_ee,
        "chp_out_charge": chp_out_charge,
        "chp_out_heat": chp_out_heat,
    }


def _actual_elec_dispatch(
    battery, chp_out_de, chp_out_ee, chp_out_charge,
    re_de, re_ee, pv_exc_ee, ee_in_elec_req, charge_value,
    chp, ee,
):
    """
    Actual electricity dispatch: meet demand, EE, then charge battery.
    Caller must set battery.soc_elec to start value before calling.
    Mutates battery. Returns flow dict for downstream and return struct.
    """
    # Demand met
    re_de2 = max(0, re_de - chp_out_de)
    excess_chp_de = chp_out_de - re_de if re_de - chp_out_de < 0 else 0
    chp_to_de = chp_out_de - excess_chp_de
    batt_out_de = battery.discharge_elec(re_de2)
    re_de3 = re_de2 - batt_out_de
    grid_out = max(0, re_de3)

    # EE met
    re_ee2 = max(0, re_ee - chp_out_ee)
    excess_chp_ee = chp_out_ee - re_ee if re_ee - chp_out_ee < 0 else 0
    chp_to_ee = chp_out_ee - excess_chp_ee
    batt_out_ee = battery.discharge_elec(re_ee2)
    re_ee3 = re_ee2 - batt_out_ee
    ee_in_elec_actual = ee_in_elec_req - re_ee3
    ee_out_heat_actual = ee.get_output_heat(ee_in_elec_actual)

    # Charge battery
    amount_to_charge = max(0, charge_value - battery.soc_elec)
    pv_out_charge = min(pv_exc_ee, amount_to_charge, battery.max_charge_rate_elec)
    pv_excess_charge = max(0, pv_exc_ee - pv_out_charge)
    re_charge1 = amount_to_charge - pv_out_charge
    chp_out_charge_req = min(
        chp.rated_output_elec - chp_out_de - chp_out_ee,
        re_charge1,
        battery.max_charge_rate_elec - pv_out_charge,
    )
    chp_elec_excess = chp_out_charge - chp_out_charge_req
    re_charge2 = re_charge1 - chp_out_charge_req
    actual_amount_to_charge = amount_to_charge - re_charge2
    battery.charge_elec(actual_amount_to_charge)

    return {
        "grid_out": grid_out,
        "batt_out_de": batt_out_de,
        "batt_out_ee": batt_out_ee,
        "chp_to_de": chp_to_de,
        "chp_to_ee": chp_to_ee,
        "excess_chp_de": excess_chp_de,
        "excess_chp_ee": excess_chp_ee,
        "chp_elec_excess": chp_elec_excess,
        "ee_in_elec_actual": ee_in_elec_actual,
        "ee_out_heat_actual": ee_out_heat_actual,
        "pv_out_charge": pv_out_charge,
        "pv_excess_charge": pv_excess_charge,
        "chp_out_charge_req": chp_out_charge_req,
        "actual_amount_to_charge": actual_amount_to_charge,
    }


def _actual_heat_dispatch(
    boiler, buffer, chp,
    heat_demand, ee_out_heat_actual, chp_out_heat,
    charge_at_buff, chp_charge_only,
):
    """
    Actual heat dispatch: meet remaining heat with CHP, buffer, boiler; store excess.
    Mutates buffer. Returns flow dict.
    """
    re_he = heat_demand - ee_out_heat_actual
    chp_out_heat_req = min(chp.rated_output_heat, re_he)
    chp_out_excess = chp_out_heat - chp_out_heat_req
    re_he2 = re_he - chp_out_heat_req
    heat_discharged = buffer.discharge_heat(re_he2)
    re_he3 = re_he2 - heat_discharged
    boiler_out = min(boiler.rated_output_heat, re_he3)
    gas_input_boiler = boiler.get_input_heat(boiler_out)

    amount_to_store_heat = max(buffer.capacity * charge_at_buff - buffer.soc_heat, 0)
    amount_to_store_from_chp = min(amount_to_store_heat, chp_out_excess)
    amount_to_store_from_boiler = (
        0
        if chp_charge_only
        else max(0, min(boiler.rated_output_heat - boiler_out, amount_to_store_heat - amount_to_store_from_chp))
    )
    chp_waste_heat = chp_out_excess - amount_to_store_from_chp

    gas_input_store_chp = amount_to_store_from_chp * (chp.rated_input_heat / chp.rated_output_heat)
    gas_input_store_boiler = boiler.get_input_heat(amount_to_store_from_boiler)
    total_in_boiler = gas_input_boiler + gas_input_store_boiler

    if total_in_boiler < BOILER_MIN_LOAD_RATIO * boiler.rated_input_heat:
        total_in_boiler = 0
        boiler_out = 0
        gas_input_boiler = 0
        amount_to_store_from_boiler = 0
        gas_input_store_boiler = 0

    total_stored_heat = amount_to_store_from_boiler + amount_to_store_from_chp
    actual_stored_heat = buffer.charge_heat(total_stored_heat)

    return {
        "boiler_out": boiler_out,
        "gas_input_boiler": gas_input_boiler,
        "gas_input_store_boiler": gas_input_store_boiler,
        "gas_input_store_chp": gas_input_store_chp,
        "total_in_boiler": total_in_boiler,
        "amount_to_store_from_boiler": amount_to_store_from_boiler,
        "amount_to_store_from_chp": amount_to_store_from_chp,
        "chp_out_heat_req": chp_out_heat_req,
        "chp_out_excess": chp_out_excess,
        "chp_waste_heat": chp_waste_heat,
        "heat_discharged": heat_discharged,
        "total_stored_heat": total_stored_heat,
        "actual_stored_heat": actual_stored_heat,
    }


def _compute_costs(
    time_delta,
    actual_chp_in, total_in_boiler, grid_out,
    feed_in_amount, chp_in_charge, chp_in_ee,
    gas_input_store_chp, gas_input_store_boiler,
    gas_price, grid_price, feed_in_price,
):
    """Compute gas and electricity cost components (€ or same unit)."""
    price_chp = actual_chp_in * time_delta * gas_price
    price_boiler = total_in_boiler * time_delta * gas_price
    price_gas_total = price_chp + price_boiler
    price_elec_grid = grid_out * time_delta * grid_price
    price_feed_in = feed_in_amount * time_delta * feed_in_price
    price_elec_total = price_elec_grid - price_feed_in
    gas_price_batt = chp_in_charge * time_delta * gas_price
    gas_price_buffer = (gas_input_store_chp + gas_input_store_boiler) * time_delta * gas_price
    gas_price_ee = chp_in_ee * time_delta * gas_price
    return {
        "price_chp": price_chp,
        "price_boiler": price_boiler,
        "price_gas_total": price_gas_total,
        "price_elec_grid": price_elec_grid,
        "price_elec_total": price_elec_total,
        "gas_price_batt": gas_price_batt,
        "gas_price_buffer": gas_price_buffer,
        "gas_price_ee": gas_price_ee,
    }


def _round_result(d, decimals=2):
    """Round numeric values in a dict for output."""
    return {k: round(float(v), decimals) for k, v in d.items()}


def Baseline_edited(
    date_time_index,
    frequency,
    components,
    elec_demand,
    gas_demand,
    pv_generated,
    gas_price,
    grid_price,
    feed_in_price,
    charge_at=0.5,
    charge_at_buff=0.5,
    chp_charge_only=False,
):
    """
    Compute outputs for one timestep.
    - Boiler, CHP, buffer and electric heating element (via buffer) meet heat demand.
    - Grid, PV, battery and CHP meet electrical demand.
    - Priority: heat EE -> CHP -> buffer -> boiler; electricity PV -> CHP -> Battery -> Grid.
    - Draft pass determines CHP proportions (min load); actual pass resets battery and dispatches.
    - gas_price, grid_price, feed_in_price: €/kWh (or same unit) for cost calculation.
    """
    time_delta = frequency / 60  # hours
    heat_demand = gas_demand[date_time_index]
    elec_demand_ts = elec_demand[date_time_index]

    boiler = components["boiler"]
    chp = components["chp"]
    pv = components["pv"]
    battery = components["battery"]
    buffer = components["buffer"]
    ee = components["ee"]
    charge_value = charge_at * battery.capacity
    battery_soc_start = battery.soc_elec
    pv_gen = max(0, min(pv_generated[date_time_index], pv.capacity))

    # Draft pass: get CHP proportions (mutates battery)
    draft = _draft_chp_proportions(
        chp, pv, battery, ee,
        heat_demand, elec_demand_ts, pv_gen, charge_value,
    )

    # Actual pass: reset battery, then dispatch
    battery.soc_elec = battery_soc_start
    chp_out = _actual_chp_split(chp, draft)

    elec = _actual_elec_dispatch(
        battery,
        chp_out["chp_out_de"], chp_out["chp_out_ee"], chp_out["chp_out_charge"],
        draft["re_de"], draft["re_ee"], draft["pv_exc_ee"], draft["ee_in_elec_req"],
        charge_value, chp, ee,
    )

    heat = _actual_heat_dispatch(
        boiler, buffer, chp,
        heat_demand, elec["ee_out_heat_actual"], chp_out["chp_out_heat"],
        charge_at_buff, chp_charge_only,
    )

    feed_in_amount = (
        elec["chp_elec_excess"] + elec["pv_excess_charge"]
        + elec["excess_chp_ee"] + elec["excess_chp_de"]
    )
    costs = _compute_costs(
        time_delta,
        chp_out["actual_chp_in"], heat["total_in_boiler"], elec["grid_out"],
        feed_in_amount, chp_out["chp_in_charge"], chp_out["chp_in_ee"],
        heat["gas_input_store_chp"], heat["gas_input_store_boiler"],
        gas_price, grid_price, feed_in_price,
    )

    raw = {
        "boiler": heat["boiler_out"],
        "boiler_to_buffer": heat["amount_to_store_from_boiler"],
        "boiler_out_total": heat["boiler_out"] + heat["amount_to_store_from_boiler"],
        "boiler_in": heat["gas_input_boiler"],
        "boiler_in_to_buffer": heat["gas_input_store_boiler"],
        "boiler_in_total": heat["gas_input_boiler"] + heat["gas_input_store_boiler"],
        "Grid": elec["grid_out"],
        "chp_out_de": elec["chp_to_de"],
        "chp_out_ee": elec["chp_to_ee"],
        "chp_out_charge": elec["chp_out_charge_req"],
        "chp_out_heat": chp_out["chp_out_heat"],
        "chp_waste_heat": heat["chp_waste_heat"],
        "chp_out_hd": heat["chp_out_heat_req"],
        "chp_out_buff": heat["amount_to_store_from_chp"],
        "chp_excess_elec": elec["chp_elec_excess"] + elec["excess_chp_ee"] + elec["excess_chp_de"],
        "chp_in": chp_out["actual_chp_in"],
        "chp_total_elec": chp_out["chp_out_de"] + chp_out["chp_out_ee"] + chp_out["chp_out_charge"],
        "pv_out_de": draft["pv_out_de"],
        "pv_out_ee": draft["pv_out_ee"],
        "pv_out_charge": elec["pv_out_charge"],
        "pv_excess": elec["pv_excess_charge"],
        "buffer_out": heat["heat_discharged"],
        "total_stored_heat": heat["total_stored_heat"],
        "actual_stored_heat": heat["actual_stored_heat"],
        "batt_out_de": elec["batt_out_de"],
        "batt_out_ee": elec["batt_out_ee"],
        "total_battery_discharge": elec["batt_out_de"] + elec["batt_out_ee"],
        "soc_elec": battery.soc_elec,
        "actual_charged_batt": elec["actual_amount_to_charge"],
        "soc_heat": buffer.soc_heat,
        "ee_in_elec_actual": elec["ee_in_elec_actual"],
        "ee_heat": elec["ee_out_heat_actual"],
        "gas_price_boiler": costs["price_boiler"],
        "elec_price_Grid": costs["price_elec_grid"],
        "gas_price_chp": costs["price_chp"],
        "gas_price_batt": costs["gas_price_batt"],
        "gas_price_buffer": costs["gas_price_buffer"],
        "gas_price_ee": costs["gas_price_ee"],
        "gas_price_total": costs["price_gas_total"],
        "elec_price_total": costs["price_elec_total"],
    }
    return _round_result(raw)
