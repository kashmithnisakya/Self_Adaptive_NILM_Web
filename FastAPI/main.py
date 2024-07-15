from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_year_3 import get_predictions_0123
from model_year_rest import get_predictions_rest

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
    print("request:",device)
    if device.year == 0 or device.year == 1 or device.year == 2 or device.year == 3:
        output = get_predictions_0123(device_name=device.device_name, device_age=device.year, start_point=device.start_point, end_point=device.end_point)
    else:
        output = get_predictions_rest(device_name=device.device_name, device_age=device.year, start_point=device.start_point, end_point=device.end_point)
    return {"response": str(output)}