"""Simple factory so CLI scripts stay tiny."""
from models.two_tower import TwoTowerModel
from models.nrms       import NRMSModel

def create_model(name: str, **kwargs):
    name = name.lower()
    if name == "two_tower":
        return TwoTowerModel(**kwargs)
    if name == "nrms":
        return NRMSModel(**kwargs)
    raise ValueError(f"Unknown model: {name}")