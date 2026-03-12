from .base import EnergyObject


class PV(EnergyObject):
    """
    Photovoltaic (PV) generator.
    """

    def __init__(self, name: str, capacity: float):
        super().__init__(name, capacity)

