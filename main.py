import os

from loguru import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from app.routes.access import router as access_router
from app.routes.compute import router as compute_router


description_text = """

### General Description:
This API is part of the software developed within GreenDatAI
project aiming to predict multiple components of an EV
charging session. The API is used to compute forecasts for:

- **EV charging session total energy consumption** (i.e., consumed by EV during a session)
- **EV charging session duration** (i.e., parked time)
- **Next EV charging session start** (i.e., date and time and returnal probabilities)

### Contacts:
- José Dias (jose.j.dias@inesctec.pt)
- José Andrade (jose.r.andrade@inesctec.pt)
- Wellington Fonseca (wellington.w.fonseca@inesctec.pt)

"""

app = FastAPI(
    title="EV Charging Session Data Forecaster API",
    description=description_text
)

# List of allowed origins
origins = ["*"]  # todo: Change to allow only chargers to interact with this

# Add CORSMiddleware to the application instance
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

log_format = "{time:YYYY-MM-DD HH:mm:ss} | {level:<5} | {message}"
logger.add(os.path.join("files", "logfile.log"), format=log_format,
           level='DEBUG', backtrace=True)
logger.info("-" * 79)

# app.include_router(access_router, prefix="/access", tags=["Data Access"])
app.include_router(compute_router, prefix="/compute", tags=["Compute Forecasts"])


@app.get("/")
def test() -> dict:
    return {"message": "Its alive!"}
