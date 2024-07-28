from dataclasses import dataclass

@dataclass
class LogLstEl:
    name: str
    value: float
    args: list
    kwargs: dict