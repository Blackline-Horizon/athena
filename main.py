from typing import Union
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get values from environment variables
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 3001))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=PlainTextResponse)
def read_root():
    return f"Reporting from Athena Service!"

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
