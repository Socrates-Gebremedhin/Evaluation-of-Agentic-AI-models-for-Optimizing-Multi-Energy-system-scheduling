from typing import Dict, Callable, Any

import numpy as np
import pandas as pd


def freq_to_timedelta(freq: str) -> pd.Timedelta:
    """
    Convert a pandas offset alias (e.g. 'H', '30T') to a Timedelta.
    """
    if freq.isalpha():
        freq = "1" + freq
    return pd.Timedelta(freq)


class DigitalTwin:
    """
    Digital twin of an energy system, orchestrating component rules and
    producing time series of loads and prices.
    """

    def __init__(self, components: Dict[str, Any]):
        """
        components: mapping from component name to EnergyObject (or similar).
        """
        self.components = components
        self.rules: Dict[str, Callable[..., Dict[str, float]]] = {}
        self.load_results: Dict[str, np.ndarray] = {}
        self.gas_price_results: Dict[str, Dict[str, np.ndarray]] = {}
        self.elec_price_results: Dict[str, Dict[str, np.ndarray]] = {}

    # ---------------- RULE DECORATOR ----------------
    def create_rule(self, name: str = None):
        """
        Decorator to register a user-defined rule.
        Rule function must accept (hour_or_array, components, **kwargs)
        and return a dict of allocations: component_name -> np.ndarray or float.
        """

        def decorator(func: Callable):
            rule_name = name or func.__name__
            self.rules[rule_name] = func
            return func

        return decorator

    # ---------------- SIMPLE LOAD SIMULATION ----------------
    def simulate_load(self, hours: int, **kwargs):
        """
        Simple hourly simulation over a fixed number of hours.
        """
        self.hours = hours
        self.load_results = {name: np.zeros(hours) for name in self.components}
        self.load_results["Grid"] = np.zeros(hours)

        for h in range(hours):
            for _, rule_func in self.rules.items():
                allocations = rule_func(h, self.components, **kwargs)
                for cname, val in allocations.items():
                    if cname not in self.load_results:
                        self.load_results[cname] = np.zeros(hours)
                    self.load_results[cname][h] = val

        return self.load_results

    # ---------------- FULL DIGITAL TWIN SIMULATION ----------------
    def simulate_twin(
        self,
        rule_name: str,
        components: Dict[str, Any],
        date_time_index: pd.DatetimeIndex,
        **kwargs,
    ):
        """
        Simulate a registered rule across a DatetimeIndex.
        Tracks load results and optional gas/electricity price series.
        """
        index = len(date_time_index)
        rule = self.rules[rule_name]

        freq = date_time_index.inferred_freq
        freq_delta = freq_to_timedelta(freq)
        minutes = freq_delta.total_seconds() / 60

        # --- First timestep (t = 0) ---
        first_allocation = rule(
            date_time_index[0],
            minutes,
            components,
            **kwargs,
        )

        load_result: Dict[str, np.ndarray] = {}
        gas_price_result: Dict[str, np.ndarray] = {}
        elec_price_result: Dict[str, np.ndarray] = {}

        for key, value in first_allocation.items():
            if key.startswith("gas_price_"):
                gas_price_result[key] = np.zeros(index)
                gas_price_result[key][0] = value
            elif key.startswith("elec_price_"):
                elec_price_result[key] = np.zeros(index)
                elec_price_result[key][0] = value
            else:
                load_result[key] = np.zeros(index)
                load_result[key][0] = value

        # --- Remaining timesteps (t = 1 ... T) ---
        for i in range(1, index):
            date_time = date_time_index[i]
            allocations = rule(date_time, minutes, components, **kwargs)

            for cname, val in allocations.items():
                if cname.startswith("gas_price_"):
                    gas_price_result[cname][i] = val
                elif cname.startswith("elec_price_"):
                    elec_price_result[cname][i] = val
                else:
                    load_result[cname][i] = val

        self.load_results[rule_name] = load_result
        self.gas_price_results[rule_name] = gas_price_result
        self.elec_price_results[rule_name] = elec_price_result

        return load_result, gas_price_result, elec_price_result

