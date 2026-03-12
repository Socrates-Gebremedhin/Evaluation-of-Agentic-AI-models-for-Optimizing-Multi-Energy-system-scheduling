from .base import EnergyObject


class CHP(EnergyObject):
    """
    Combined heat and power (CHP) unit.
    """

    def __init__(
        self,
        name: str,
        capacity: float,
        rated_output_heat: float,
        rated_output_elec: float,
    ):
        super().__init__(name, capacity)
        self.rated_output_heat = rated_output_heat
        self.rated_output_elec = rated_output_elec
        self.rated_input_heat = capacity  # fuel input in kW

    def get_output_heat(self, in_fuel: float) -> float:
        """Output heat [kW] for given fuel input [kW]."""
        efficiency = self.rated_output_heat / self.rated_input_heat
        return in_fuel * efficiency

    def get_output_elec(self, in_fuel: float) -> float:
        """Output electricity [kW] for given fuel input [kW]."""
        efficiency = self.rated_output_elec / self.rated_input_heat
        return in_fuel * efficiency

    def get_input_heat_frm_elec(self, in_elec: float) -> float:
        """Required fuel input [kW] for desired electrical output [kW]."""
        efficiency = self.rated_output_elec / self.rated_input_heat
        return in_elec / efficiency

    def get_input_heat_frm_heat(self, in_heat: float) -> float:
        """Required fuel input [kW] for desired heat output [kW]."""
        efficiency = self.rated_output_heat / self.rated_input_heat
        return in_heat / efficiency

