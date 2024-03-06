
from typing import Optional, List
from datetime import datetime
from fastapi import FastAPI, HTTPException


app = FastAPI()

db = [{'id': 1, 'size': 's', 'fuel': 'gasoline', 'doors': 4, 'transmission': 'auto'},
      {'id': 2, 'size': 's', 'fuel': 'gasoline', 'doors': 3, 'transmission': 'auto'},
      {'id': 3, 'size': 's', 'fuel': 'gasoline', 'doors': 2, 'transmission': 'auto'}]


@app.get('/')
def welcome():
    """Return a welcome message."""
    return {'message': 'Welcome to the car sharing service!'}

@app.get('/date')
def date():
    return {'date': datetime.now()}

@app.get('/api/cars')
def get_cars(size: Optional[str]=None, doors: Optional[int]=None) -> List:
    result = db
    if size:
        result = [car for car in result if car['size'] == size]
    if doors and doors <=5:
        result = [car for car in result if car['doors'] >= doors]
    else:
        raise HTTPException(status_code=404, detail='No car with doors greater than 5!')

    return result

@app.get('/api/cars/{id}')
def car_by_id(id: int) -> dict:
    result = [car for car in db if car['id'] == id]
    if result:
        return result[0]
    else:
        raise HTTPException(status_code=404, detail=f'No car with id= {id}.')
