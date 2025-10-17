from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.core.config import config
from src.api.endpoints import api_router
from src.api.middleware import RequestIDMiddleware

import logging


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


app = FastAPI()
app.include_router(api_router)


app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "API"}
