from .base import EnergyObject


class Battery(EnergyObject):
    """
    Electrical battery storage.
    """

    def __init__(
        self,
        name: str,
        capacity: float,
        soc_elec: float,
        max_charge_rate_elec: float,
        max_discharge_rate_elec: float,
        eta_charge_elec: float,
        eta_discharge_elec: float,
    ):
        super().__init__(name, capacity)
        self.soc_elec = soc_elec
        self.max_charge_rate_elec = max_charge_rate_elec
        self.max_discharge_rate_elec = max_discharge_rate_elec
        self.eta_charge_elec = eta_charge_elec
        self.eta_discharge_elec = eta_discharge_elec

    def charge_elec(self, desired: float) -> float:
        """
        Charge battery with desired electrical power [kW].
        Returns actual stored energy [kW].
        """
        input_elec = max(0.0, min(desired, self.max_charge_rate_elec))
        to_store = input_elec * self.eta_charge_elec
        available_capacity = self.capacity - self.soc_elec
        actual_charge = min(to_store, available_capacity)
        self.soc_elec += actual_charge
        return actual_charge

    def discharge_elec(self, desired: float) -> float:
        """
        Discharge battery with desired electrical power [kW].
        Returns actual delivered power [kW].
        """
        output_elec = max(0.0, min(desired, self.max_discharge_rate_elec))
        to_provide = min(output_elec, self.soc_elec)
        actual_discharge = to_provide * self.eta_discharge_elec
        self.soc_elec -= to_provide
        return actual_discharge

