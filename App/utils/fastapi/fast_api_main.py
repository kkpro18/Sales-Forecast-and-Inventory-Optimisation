from concurrent.futures import ProcessPoolExecutor
from fastapi import FastAPI
from .routers import data_preprocessing_route, data_forecasting_route

app = FastAPI()
executor = ProcessPoolExecutor(max_workers=2)  # do one less than the number of cores || no. jobs in parallel


app.include_router(data_preprocessing_route.router)
app.include_router(data_forecasting_route.router)