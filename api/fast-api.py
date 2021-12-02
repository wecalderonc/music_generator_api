from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import build_music
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(user_input_notes, user_input_durations):
  random_input_notes = json.loads(user_input_notes)
  random_input_durations = json.loads(user_input_durations)

  return build_music.create_song(random_input_notes, random_input_durations)

@app.get("/random_notes")
def random_notes():
  return build_music.randomizing_user_input()
