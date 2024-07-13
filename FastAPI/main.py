from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

class Device(BaseModel):
    device_name: str
    year: int
    start_point: int
    end_point: int
    

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific origin if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"message": "hello world"}

@app.post("/predict/")
def get_prediction(device: Device):
    print(device)
    return {"prediction": "This is a prediction"}