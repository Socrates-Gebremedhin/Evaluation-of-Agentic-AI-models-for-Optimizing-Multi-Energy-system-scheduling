from .base import EnergyObject


class Boiler(EnergyObject):
    """
    Conventional boiler (fuel to heat).
    """

    def __init__(self, name: str, capacity: float, rated_input_heat: float):
        super().__init__(name, capacity)
        self.rated_input_heat = rated_input_heat  # fuel input [kW]
        self.rated_output_heat = capacity         # heat output [kW]
        self.eta_boiler = capacity / rated_input_heat

    def get_output_heat(self, in_fuel: float) -> float:
        """Output heat [kW] for given fuel input [kW]."""
        return in_fuel * self.eta_boiler

    def get_input_heat(self, out_heat: float) -> float:
        """Required fuel input [kW] for desired heat output [kW]."""
        return out_heat / self.eta_boiler

