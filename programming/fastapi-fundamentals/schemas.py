
from pydantic import BaseModel

import json

# data schema
class Car(BaseModel):
    id: int
    size: str
    fuel: str | None = 'electric'
    doors: int
    transmission: str | None = 'auto'


def load_db():
    with open('cars.json', 'r') as f:
        return [Car.parse_obj(obj) for obj in json.load(f)]


def save_db(cars: list[Car]):
    with open('cars.json', 'r') as f:
        json.dump([car.dict() for car in cars], f, indent=4)