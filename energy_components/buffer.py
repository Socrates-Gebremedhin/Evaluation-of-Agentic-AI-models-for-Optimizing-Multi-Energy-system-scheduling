from .base import EnergyObject


class Buffer(EnergyObject):
    """
    Thermal buffer / heat storage.
    """

    def __init__(
        self,
        name: str,
        capacity: float,
        soc_heat: float,
        eta_charge_heat: float,
        eta_discharge_heat: float,
    ):
        super().__init__(name, capacity)
        self.soc_heat = soc_heat  # state of charge [kWh-equivalent]
        self.eta_charge_heat = eta_charge_heat
        self.eta_discharge_heat = eta_discharge_heat

    def charge_heat(self, desired: float) -> float:
        """
        Charge buffer with desired heat input [kW].
        Returns actual stored heat [kW].
        """
        input_heat = max(0.0, desired)
        to_store = input_heat * self.eta_charge_heat
        available_capacity = self.capacity - self.soc_heat
        actual_charge = min(to_store, available_capacity)
        self.soc_heat += actual_charge
        return actual_charge

    def discharge_heat(self, desired: float) -> float:
        """
        Discharge buffer with desired heat output [kW].
        Returns actual delivered heat [kW].
        """
        output_heat = max(desired, 0.0)
        to_provide = min(output_heat, self.soc_heat)
        actual_discharge = to_provide * self.eta_discharge_heat
        self.soc_heat -= to_provide
        return actual_discharge

