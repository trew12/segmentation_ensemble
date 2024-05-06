from dataclasses import dataclass


@dataclass
class Model:
    name: str
    short_name: str
    queries: bool
    dataset: str


@dataclass
class Config:
    device: str
    img_dir: str
    mask_dir: str
    batch_size: int
    img_size: int
    model: Model


