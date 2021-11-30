from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import build_music
from random import randrange

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

@app.get("/genres")
def index():
    return {"1": "rock", "2": "pop"}

@app.get("/predict")
def predict():
  song_name = f"test_song_number_{randrange(1000)}"
  new_song = build_music.create_song(song_name)
  return new_song
