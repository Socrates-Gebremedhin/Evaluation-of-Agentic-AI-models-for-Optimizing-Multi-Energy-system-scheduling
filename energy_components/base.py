from dataclasses import dataclass


@dataclass
class EnergyObject:
    """
    Base class for energy system components.

    Attributes:
        name: Identifier for the component.
        capacity: Rated capacity in kW.
    """

    name: str
    capacity: float  # kW

