def Baseline(date_time_index, frequency, components,elec_demand,gas_demand,pv_generated,charge_at=0.5,charge_at_buff=1,chp_charge_only=False,gas_price=None,grid_price=None,feed_in_price=None):
    """
    Compute outputs for one timestep.
    - Implement all components as per the diagram provided
    - Boiler,CHP,buffer and electric heating element (through buffer) meets heat demand
    - Grid,PV,battery and CHP meets electrical demand
    - We use a sequential approach for heat EE -> CHP -> buffer -> boiler
    - For electricity, PV -> CHP - > Battery -> GRID
    - Assumes discharging of battery is possible while charging for electricity
    - All components operate every day at least at min gauge level
    - CHP,PV also charges the battery (priority respected)
    - Battery,CHP and PV are sources to the electricity heating element (ee)
    - parameters include initial SOC (defined outside this function) for buffer and battery and charge_at
    """
    time_delta  = frequency / 60 # in hours
    heat_demand = gas_demand[date_time_index]
    elec_demand = elec_demand[date_time_index]

    boiler = components["boiler"]
    chp = components["chp"]
    pv = components["pv"]
    battery = components["battery"]
    buffer = components["buffer"]
    ee = components["ee"]
    charge_value = charge_at * battery.capacity
    battery_soc_start = battery.soc_elec

    # electricity demand met
    pv_gen = max(0,min(pv_generated[date_time_index],pv.capacity)) # does not change with chp
    pv_out_de = min(pv_gen, elec_demand) # does not change with chp
    pv_ex_de = max(0, pv_gen - pv_out_de) # does not change with chp
    re_de = elec_demand - pv_out_de # does not change with chp

    chp_out_de_ef = min(chp.rated_output_elec, re_de)
    chp_in_de_ef = chp.get_input_heat_frm_elec(chp_out_de_ef)
    re_de2 = re_de - chp_out_de_ef
    batt_out_de = battery.discharge_elec(re_de2)
    re_de3 = re_de2 - batt_out_de
    grid_out = max(0,re_de3) # all remaining elec demand met by grid


    ee_out_heat_in = min(ee.rated_output_heat, heat_demand) # does not change with chp
    ee_in_elec_req = ee.get_input_elec(ee_out_heat_in) # does not change with chp

    # ee demand meet
    pv_out_ee = min(pv_ex_de, ee_in_elec_req)   # does not change with chp
    pv_exc_ee = max(0, pv_ex_de - pv_out_ee)  # does not change with chp
    re_ee = ee_in_elec_req - pv_out_ee  # does not change with chp
    chp_out_ee_ef = min(chp.rated_output_elec - chp_out_de_ef, re_ee)
    chp_in_ee_ef = chp.get_input_heat_frm_elec(chp_out_ee_ef)
    re_ee2 = re_ee - chp_out_ee_ef
    batt_out_ee = battery.discharge_elec(re_ee2)
    re_ee3 = re_ee2 - batt_out_ee

    ee_in_elec_actual_ef = ee_in_elec_req - re_ee3
    ee_out_heat_actual_ef = ee.get_output_heat(ee_in_elec_actual_ef)

    # charge battery with excess pv and chp
    amount_to_charge = max(0,charge_value - battery.soc_elec)
    pv_out_charge = min(pv_exc_ee, amount_to_charge, battery.max_charge_rate_elec) # excess pv can be generated
    re_charge1 = amount_to_charge - pv_out_charge
    chp_out_charge_ef = min(chp.rated_output_elec - chp_out_de_ef - chp_out_ee_ef, re_charge1, battery.max_charge_rate_elec - pv_out_charge)
    chp_in_charge_ef = chp.get_input_heat_frm_elec(chp_out_charge_ef)
    re_charge2 = re_charge1 - chp_out_charge_ef
    actual_amount_to_charge = amount_to_charge - re_charge2
    act_charged = battery.charge_elec(actual_amount_to_charge)
    batt_rej = actual_amount_to_charge - act_charged/battery.eta_charge_elec 

    tota_chp_in_ef = chp_in_de_ef + chp_in_ee_ef + chp_in_charge_ef if (chp_in_de_ef + chp_in_ee_ef + chp_in_charge_ef)>=0.6*chp.rated_input_heat else 0

    # heat demand met
    re_he = heat_demand - ee_out_heat_actual_ef
    chp_out_heat_hf = min(chp.rated_output_heat, re_he)
    chp_in_heat_hf = chp.get_input_heat_frm_heat(chp_out_heat_hf) if chp.get_input_heat_frm_heat(chp_out_heat_hf)>=0.6*chp.rated_input_heat else 0 

    # compare and finally do actual transactions

    # initialize battery status
    battery.soc_elec = battery_soc_start
    # print("Battery initial: ", battery_soc_start)

    actual_chp_in = max(tota_chp_in_ef, chp_in_heat_hf)
    
    if tota_chp_in_ef !=0:
        chp_in_de = (chp_in_de_ef/tota_chp_in_ef)*actual_chp_in
        chp_in_ee = (chp_in_ee_ef/tota_chp_in_ef)*actual_chp_in
        chp_in_charge = (chp_in_charge_ef/tota_chp_in_ef)*actual_chp_in
    else:
        chp_in_de = (1/3)*actual_chp_in
        chp_in_ee = (1/3)*actual_chp_in
        chp_in_charge = (1/3)*actual_chp_in


    chp_out_de = chp_in_de * (chp.rated_output_elec / chp.rated_input_heat)
    chp_out_ee = chp_in_ee * (chp.rated_output_elec / chp.rated_input_heat)
    chp_out_charge = chp_in_charge * (chp.rated_output_elec / chp.rated_input_heat)
    chp_out_heat = actual_chp_in * (chp.rated_output_heat / chp.rated_input_heat)

    # demand met
    re_de2 = max(0,re_de - chp_out_de) # excess electricity can be generated
    excess_chp_de = chp_out_de - re_de if  re_de - chp_out_de <0 else 0
    chp_to_de = chp_out_de - excess_chp_de
    batt_out_de = battery.discharge_elec(re_de2)
    re_de3 = re_de2 - batt_out_de
    grid_out = max(0,re_de3) # all remaining elec demand met by grid

    # ee demand met
    re_ee2 = max(0,re_ee - chp_out_ee) # excess electricity can be generated
    excess_chp_ee = chp_out_ee - re_ee if  re_ee - chp_out_ee <0 else 0
    chp_to_ee = chp_out_ee - excess_chp_ee
    batt_out_ee = battery.discharge_elec(re_ee2)
    re_ee3 = re_ee2 - batt_out_ee
    ee_in_elec_actual = ee_in_elec_req - re_ee3
    ee_out_heat_actual = ee.get_output_heat(ee_in_elec_actual)

    # charge the battery
    amount_to_charge = max(0,charge_value - battery.soc_elec)
    pv_out_charge = min(pv_exc_ee, amount_to_charge, battery.max_charge_rate_elec) # excess pv can be generated
    pv_excess_charge = max(0,pv_exc_ee - pv_out_charge)
    re_charge1 = amount_to_charge - pv_out_charge
    chp_out_charge_req = min(chp.rated_output_elec - chp_out_de - chp_out_ee, re_charge1, battery.max_charge_rate_elec - pv_out_charge)
    chp_elec_excess = chp_out_charge - chp_out_charge_req
    re_charge2 = re_charge1 - chp_out_charge_req
    actual_amount_to_charge = amount_to_charge - re_charge2
    battery.charge_elec(actual_amount_to_charge)

    # heat demand met
    re_he = heat_demand - ee_out_heat_actual
    chp_out_heat_req = min(chp.rated_output_heat, re_he)
    chp_out_excess = chp_out_heat - chp_out_heat_req
    re_he2 = re_he - chp_out_heat_req
    #print("Buffer_initial:",buffer.soc_heat)
    #print("chp_out_excess_heat:",chp_out_excess)
    heat_discharged = buffer.discharge_heat(re_he2)
    #print("Buffer_discharged:",heat_discharged)
    #print("Buffer_soc_after_discharge:",buffer.soc_heat)
    re_he3 = re_he2 - heat_discharged
    boiler_out = min(boiler.rated_output_heat, re_he3)
    gas_input_boiler = boiler.get_input_heat(boiler_out)
    amount_to_store_heat = max(buffer.capacity*charge_at_buff - buffer.soc_heat,0)
    amount_to_store_from_chp = min(amount_to_store_heat,chp_out_excess)
    amount_to_store_from_boiler = 0 if chp_charge_only else max(0,min(boiler.rated_output_heat - boiler_out,amount_to_store_heat - amount_to_store_from_chp))
    chp_waste_heat = chp_out_excess - amount_to_store_from_chp
    

    gas_input_store_chp = amount_to_store_from_chp * (chp.rated_input_heat / chp.rated_output_elec)
    gas_input_store_boiler = boiler.get_input_heat(amount_to_store_from_boiler)
    total_in_boiler = (gas_input_boiler+gas_input_store_boiler)
    if total_in_boiler < 0.2*boiler.rated_input_heat:
        total_in_boiler = 0
        boiler_out = 0
        gas_input_boiler = 0
        amount_to_store_from_boiler = 0
        gas_input_store_boiler = 0

    
    total_stored_heat = amount_to_store_from_boiler + amount_to_store_from_chp
    
    actual_stored_heat = buffer.charge_heat(total_stored_heat)
    #print("stored_heat:",total_stored_heat)
    #print("Buffer after charge:",buffer.soc_heat)

    final_soc_batt = battery.soc_elec
    final_soc_buffer = buffer.soc_heat

    # Calculate prices
    price_chp = actual_chp_in * time_delta * gas_price
    price_boiler = total_in_boiler *time_delta* gas_price
    price_gas_total = price_chp + price_boiler

    price_elec_grid = grid_out * time_delta * grid_price

    #calculate generated electricity gain
    excess_chp_elec = chp_elec_excess + excess_chp_ee + excess_chp_de
    feed_in_amount = chp_elec_excess + pv_excess_charge + excess_chp_ee + excess_chp_de
    price_feed_in = feed_in_amount * time_delta * feed_in_price
    price_elec_total = price_elec_grid - price_feed_in
    gas_price_batt = chp_in_charge  * time_delta * feed_in_price
    gas_price_buffer = (gas_input_store_chp + gas_input_store_boiler)*time_delta* gas_price
    gas_price_ee = chp_in_ee* time_delta * gas_price

    def round_val(val):
        val = float(val)
        return round(val,2)

    


    return {
        "boiler": round_val(boiler_out),
        "boiler_to_buffer": round_val(amount_to_store_from_boiler),
        "boiler_out_total":round_val(boiler_out+amount_to_store_from_boiler),
        "boiler_in":round_val(gas_input_boiler),
        "boiler_in_to_buffer":round_val(gas_input_store_boiler),
        "boiler_in_total":round_val(gas_input_boiler+gas_input_store_boiler),
        "Grid": round_val(grid_out),
        "chp_out_de": round_val(chp_to_de),
        "chp_out_ee": round_val(chp_to_ee),
        "chp_out_charge": round_val(chp_out_charge_req),
        "chp_out_heat": round_val(chp_out_heat),
        "chp_waste_heat":round_val(chp_waste_heat),
        "chp_out_hd":round_val(chp_out_heat_req),
        "chp_out_buff":round_val(amount_to_store_from_chp),
        "chp_excess_elec":round_val(excess_chp_elec),
        "chp_in": round_val(actual_chp_in),
        "chp_total_elec": round_val(chp_out_de + chp_out_ee + chp_out_charge),
        "pv_out_de":round_val(pv_out_de),
        "pv_out_ee":round_val(pv_out_ee),
        "pv_out_charge":round_val(pv_out_charge),
        "pv_excess":round_val(pv_excess_charge),
        "buffer_out":round_val(heat_discharged),
        "total_stored_heat":round_val(total_stored_heat),
        "actual_stored_heat":round_val(actual_stored_heat),
        "batt_out_de":round_val(batt_out_de),
        "batt_out_ee":round_val(batt_out_ee),
        "total_battery_discharge":round_val(batt_out_de + batt_out_ee),
        "soc_elec":round_val(final_soc_batt),
        "actual_charged_batt":round_val(actual_amount_to_charge),
        "soc_heat":round_val(final_soc_buffer),
        "ee_in_elec_actual":round_val(ee_in_elec_actual),
        "ee_heat":round_val(ee_out_heat_actual),
        "gas_price_boiler": round_val(price_boiler),
        "elec_price_Grid": round_val(price_elec_grid),
        "gas_price_chp": round_val(price_chp),
        "gas_price_batt":round_val(gas_price_batt),
        "gas_price_buffer":round_val(gas_price_buffer),
        "gas_price_ee":round_val(gas_price_ee),
        "gas_price_total": round_val(price_gas_total),
        "elec_price_total": round_val(price_elec_total)
    }