from fastapi import FastAPI
from first_practice import Exercises
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}