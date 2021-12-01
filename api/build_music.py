import random
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import json

model = load_model("models/model_40seq_196voc_5tracks.h5")

def create_song():
  Music_notes= Malody_Generator(200)
  return { "notes": json.dumps(Music_notes) }

def Malody_Generator(Note_Count):
    pickle_model = pickle.load(open('models/wil_deploy_dummy.pickle', 'rb'))
    network_input = pickle_model['network_input']
    pitchnames = pickle_model['pitchnames']

    seed = network_input[np.random.randint(0, len(network_input)-1)]
    Music = ""
    Notes_Generated = []
    int_to_note = dict((number, note)
                       for number, note in enumerate(pitchnames))
    for i in range(Note_Count):
        n_vocab = len(int_to_note)
        seed = seed.reshape(1, 40, 1)
        prediction = model.predict(seed, verbose=0)[0]
        prediction = np.log(prediction)  # diversity
        exp_preds = np.exp(prediction) / 1
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index / float(n_vocab)
        Notes_Generated.append(index)
        Music = [int_to_note[char] for char in Notes_Generated]
        seed = np.insert(seed[0], len(seed[0]), index_N)
        seed = seed[1:]
    #Now, we have music in form or a list of chords and notes and we want to be a midi file.

    #Melody = chords_n_notes(Music)
    #Melody_midi = stream.Stream(Melody)
    return Music
