import numpy as np
import pandas as pd


def feasibility_summary(constraint_result_df, tol=0.1):
    """
    For each constraint column, compute % feasible and the indexes where it is infeasible.

    - Boolean columns: feasible = True; infeasible indexes = where False.
    - Numeric columns: feasible = |value| <= tol; infeasible indexes = where |value| > tol.

    Parameters
    ----------
    constraint_result_df : pd.DataFrame
        Output of check_results(...) with columns like heat_demand_met, elec_demand_met, etc.
    tol : float
        Tolerance for numeric residuals; rows with |value| <= tol are considered feasible.

    Returns
    -------
    pd.DataFrame
        Index = constraint column names.
        Columns:
          - pct: percentage of rows that are feasible (0–100).
          - infeasible_indexes: list of index labels where the constraint is violated.
    """
    rows = []
    for col in constraint_result_df.columns:
        s = constraint_result_df[col].dropna()
        n = len(s)
        if n == 0:
            rows.append({"pct": 100.0, "infeasible_indexes": []})
            continue
        if pd.api.types.is_bool_dtype(s):
            feasible = s
        else:
            s = pd.to_numeric(s, errors="coerce")
            feasible = np.abs(s) <= tol
        pct = 100.0 * feasible.sum() / n
        infeasible_idx = s.index[~feasible].tolist()
        rows.append({"pct": float(pct), "infeasible_indexes": infeasible_idx})
    return pd.DataFrame(rows, index=constraint_result_df.columns)


def constraint_checker_row(result_df,index,constraint_name,components,tol = 0.1):
    boiler = components["boiler"]
    chp = components["chp"]
    battery = components["battery"]
    buffer = components["buffer"]
    ee = components["ee"]
    
    if constraint_name == "heat_demand_met":
        return (result_df.loc[index,"gas_demand"] - (result_df.loc[index,"chp_out_hd"] + 
                                                result_df.loc[index,"boiler"] + 
                                                result_df.loc[index,"buffer_out"] + 
                                                result_df.loc[index,"ee_heat"]
                                            )
         
                )
    
    elif constraint_name == "chp_threshold":
        return ( ((result_df.loc[index,"chp_out_heat"] + tol >= 0.6*chp.rated_output_heat ) | (result_df.loc[index,"chp_out_heat"] == 0)) & 
                ((result_df.loc[index,"chp_in"] + tol >= 0.6*chp.rated_input_heat) | (result_df.loc[index,"chp_in"] == 0)) &
                ((result_df.loc[index,"chp_total_elec"] + tol >= 0.6*chp.rated_output_elec) | (result_df.loc[index,"chp_total_elec"] == 0))


        )
    elif constraint_name == "boiler_threshold":
        return ( ((result_df.loc[index,"boiler_out_total"] + tol >= 0.2*boiler.rated_output_heat) | (result_df.loc[index,"boiler_out_total"] == 0)) & 
                ((result_df.loc[index,"boiler_in_total"] + tol >= 0.2*boiler.rated_input_heat) | (result_df.loc[index,"boiler_in_total"] == 0))


        )
    
    elif constraint_name == "elec_demand_met":        
        return (result_df.loc[index,"elec_demand"] - (result_df.loc[index,"pv_out_de"] + 
                                        result_df.loc[index,"batt_out_de"] + 
                                        result_df.loc[index,"chp_out_de"] +
                                        result_df.loc[index,"Grid"]
                                    )
    
                )
    elif constraint_name == "batt_charge_chp_pv":        
        return (result_df.loc[index,"actual_charged_batt"] - (result_df.loc[index,"chp_out_charge"] + 
                                        result_df.loc[index,"pv_out_charge"]
                                    )
    
                )

    elif constraint_name == "buff_charge_chp_boil":        
        return (result_df.loc[index,"total_stored_heat"] - (result_df.loc[index,"boiler_to_buffer"] + 
                                        result_df.loc[index,"chp_out_buff"]
                                    )
    
                )

    elif constraint_name == "elec_to_ee_by_pv_chp_batt":        
        return (result_df.loc[index,"ee_in_elec_actual"] - (result_df.loc[index,"chp_out_ee"] + 
                                        result_df.loc[index,"batt_out_ee"] + 
                                        result_df.loc[index,"pv_out_ee"]
                                    )
    
                )
    elif constraint_name == "pv_generated_alloc":        
        return (result_df.loc[index,"pv_gen"] >= (result_df.loc[index,"pv_out_ee"] + 
                                        result_df.loc[index,"pv_out_de"] + 
                                        result_df.loc[index,"pv_out_charge"] #+
                                        #result_df.loc[index,"pv_excess"]
                                    )
    
                )
    
    elif constraint_name == "boiler_alloc":        
        return (result_df.loc[index,"boiler_out_total"] - (result_df.loc[index,"boiler_to_buffer"] + 
                                        result_df.loc[index,"boiler"]
                                    )
    
                )
    elif constraint_name == "boiler_out_ratings":        
        return (boiler.rated_output_heat >= (result_df.loc[index,"boiler_out_total"]))

    elif constraint_name == "boiler_in_out_rln":        
        return (result_df.loc[index,"boiler_in_total"] - (boiler.get_input_heat(result_df.loc[index,"boiler_out_total"])
                                    )
                )
    
    elif constraint_name == "ee_elec_ratings":        
        return (ee.capacity >= (result_df.loc[index,"ee_in_elec_actual"]))


    elif constraint_name == "ee_in_out_rln":        
        return (result_df.loc[index,"ee_in_elec_actual"] - (ee.get_input_elec(result_df.loc[index,"ee_heat"])
                                    )
                )

    elif constraint_name == "batt_disch_to_de_ee":        
        return (result_df.loc[index,"total_battery_discharge"] - (result_df.loc[index,"batt_out_de"] +
                                                                  result_df.loc[index,"batt_out_ee"]
                                                                )
                )

    elif constraint_name == "batt_disch_ratings":
        result_df["previous_soc"] = result_df["soc_elec"].shift(1).fillna(30)       
        return ((result_df.loc[index,"total_battery_discharge"] <= (result_df.loc[index,"previous_soc"]) + tol) 
                and (
                        result_df.loc[index,"total_battery_discharge"] <= battery.max_discharge_rate_elec + tol

                )
                )

    elif constraint_name == "buff_disch_ratings":
        result_df["previous_soc_buff"] = result_df["soc_heat"].shift(1).fillna(50)        
        return ((result_df.loc[index,"buffer_out"] <= (result_df.loc[index,"previous_soc_buff"]) + tol) 

                )
    
    elif constraint_name == "buff_charge_ratings":
        result_df["previous_soc_buff"] = result_df["soc_heat"].shift(1).fillna(50)          
        return ((result_df.loc[index,"actual_stored_heat"] <= (buffer.capacity - result_df.loc[index,"previous_soc_buff"] +
                                                              result_df.loc[index,"buffer_out"]/buffer.eta_discharge_heat) + tol) 

                )

    elif constraint_name == "batt_charge_ratings":
        result_df["previous_soc"] = result_df["soc_elec"].shift(1).fillna(30)       
        return ((result_df.loc[index,"actual_charged_batt"] <= (battery.capacity - result_df.loc[index,"previous_soc"] + 
                                                                result_df.loc[index,"total_battery_discharge"]/battery.eta_discharge_elec) + tol) 
                and (
                        result_df.loc[index,"actual_charged_batt"] <= battery.max_charge_rate_elec + tol

                )
                )

    elif constraint_name == "chp_to_heat_buff":       
        return (result_df.loc[index,"chp_out_heat"] + tol >= (result_df.loc[index,"chp_out_hd"] + 
                                        result_df.loc[index,"chp_out_buff"] #+
                                        #result_df.loc[index,"chp_waste_heat"]
                                    )
    
                )

    elif constraint_name == "chp_to_ele_batt_ee":       
        return (result_df.loc[index,"chp_total_elec"] + tol >= (result_df.loc[index,"chp_out_de"] + 
                                        result_df.loc[index,"chp_out_charge"] + 
                                        result_df.loc[index,"chp_out_ee"] #+ 
                                        #result_df.loc[index,"chp_excess_elec"]
                                    )
    
                )

    elif constraint_name == "chp_in_out_rln_heat":        
        return (result_df.loc[index,"chp_out_heat"] - (chp.get_output_heat(result_df.loc[index,"chp_in"])
                                    )
                )

    elif constraint_name == "chp_in_out_rln_elec":        
        return (result_df.loc[index,"chp_total_elec"] - (chp.get_output_elec(result_df.loc[index,"chp_in"])
                                    )
                )

    elif constraint_name == "chp_out_elec_ratings":        
        return (chp.rated_output_elec >= (result_df.loc[index,"chp_total_elec"]))

    elif constraint_name == "chp_out_heat_ratings":        
        return (chp.rated_output_heat >= (result_df.loc[index,"chp_out_heat"]))
    

def check_results(result_df,components,ignore=[],tol=0.1):
    """There are 22 constraints with ignore to indicate which constraints to ignore
    1:"heat_demand_met",
    2:"elec_demand_met",
    3:"batt_charge_chp_pv",
    4:"buff_charge_chp_boil",
    5:"elec_to_ee_by_pv_chp_batt",
    6:"pv_generated_alloc",
    7:"boiler_alloc",
    8:"boiler_out_ratings",
    9:"boiler_in_out_rln",
    10:"ee_elec_ratings",
    11:"ee_in_out_rln",
    12:"batt_disch_to_de_ee",
    13:"batt_disch_ratings",
    14:"buff_disch_ratings",
    15:"buff_charge_ratings",
    16:"batt_charge_ratings",
    17:"chp_to_heat_buff",
    18:"chp_to_ele_batt_ee",
    19:"chp_in_out_rln_heat",
    20:"chp_in_out_rln_elec",
    21:"chp_out_elec_ratings",
    22:"chp_out_heat_ratings,
    23:"chp_threshold",
    24:"boiler_threshold",

    Args:
        result_df (_type_): column names must be standardized
    """
    constraints = {1:"heat_demand_met",
                   2:"elec_demand_met",
                   3:"batt_charge_chp_pv",
                   4:"buff_charge_chp_boil",
                   5:"elec_to_ee_by_pv_chp_batt",
                   6:"pv_generated_alloc",
                   7:"boiler_alloc",
                   8:"boiler_out_ratings",
                   9:"boiler_in_out_rln",
                   10:"ee_elec_ratings",
                   11:"ee_in_out_rln",
                   12:"batt_disch_to_de_ee",
                   13:"batt_disch_ratings",
                   14:"buff_disch_ratings",
                   15:"buff_charge_ratings",
                   16:"batt_charge_ratings",
                   17:"chp_to_heat_buff",
                   18:"chp_to_ele_batt_ee",
                   19:"chp_in_out_rln_heat",
                   20:"chp_in_out_rln_elec",
                   21:"chp_out_elec_ratings",
                   22:"chp_out_heat_ratings",
                   23:"chp_threshold",
                   24:"boiler_threshold"
                   }
    row_wise_results = pd.DataFrame(index=result_df.index)
    #row wise checker
    for ind in result_df.index:
        for const_num,constraint_name in constraints.items():
            if const_num in ignore:
                continue
            else:
                row_wise_results.loc[ind,constraint_name] = constraint_checker_row(result_df,ind,constraint_name,components=components,tol=tol)



    #column wise checker

    return row_wise_results


def count_switch_ons(df, columns=None):
    """
    Count how many times each component "switched on" (previous state 0, current state != 0).

    Parameters
    ----------
    df : pd.DataFrame
        Must have a row order (e.g. time index).
    columns : list of str, optional
        Column names to count. Default: ["boiler_in_total", "chp_in"].

    Returns
    -------
    pd.Series
        Index = column name, value = number of switch-on events.
    """
    prev = df[columns].shift(1)
    curr = df[columns]
    switched = ((prev == 0) | prev.isna()) & (curr != 0)
    return switched.sum()
