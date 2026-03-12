from .base import EnergyObject


class ElectricHeatingElement(EnergyObject):
    """
    Electric heating element converting electricity to heat.
    """

    def __init__(self, name: str, capacity: float, conv_efficiency: float):
        super().__init__(name, capacity)
        self.rated_input_elec = capacity
        self.conv_efficiency = conv_efficiency
        self.rated_output_heat = capacity * conv_efficiency

    def get_output_heat(self, in_elec: float) -> float:
        """
        Heat output [kW] for a given electrical input [kW].
        """
        return min(in_elec, self.rated_input_elec) * self.conv_efficiency

    def get_input_elec(self, in_heat: float) -> float:
        """
        Required electrical input [kW] for desired heat output [kW].
        """
        return min(in_heat / self.conv_efficiency, self.rated_input_elec)

