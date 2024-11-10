from .er_baseline import ER
from .mvp import MVP

__all__ = [
    "ER",
    "MVP",
]

def get_method(name):
    name = name.lower()
    try:
        return {
            "er": ER,
            "mvp": MVP,
        }[name]
    except KeyError:
        raise NotImplementedError(f"Method {name} not implemented")