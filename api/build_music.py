import random
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import json

model_notes = load_model("models/model_notes_beeth_50k.h5")
model_dur = load_model("models/model_dur_beeth_50k.h5")
pickle_dur = pickle.load(open("models/beethoven_dur.pickle", 'rb'))
pickle_notes = pickle.load(open("models/beethoven_notes.pickle", 'rb'))
pitchname_notes_train = pickle_notes['pitchnames_notes']
n_vocab_notes_test = pickle_notes['n_vocab_notes']
pitchnames_dur_train = pickle_dur['pitchnames_dur']
n_vocab_dur_test = pickle_dur['n_vocab_dur']

def create_song(random_input_notes, random_input_durations):
  network_input_note_test = pickle_notes['network_input_test']
  network_input_dur_test = pickle_dur['network_input_test']
  Music_notes, Music_durations = Malody_Generator(50, network_input_note_test, network_input_dur_test, random_input_notes, random_input_durations)

  return { "notes": json.dumps(Music_notes), "durations": json.dumps(Music_durations) }

def randomizing_user_input():
  user_input_notes = []
  user_input_duration = []

  for i in range(3):
    user_input_notes.append(random.randint(168, n_vocab_notes_test))
    user_input_duration.append(random.randint(0, n_vocab_dur_test))
    int_to_notes = dict((number, note) for number, note in enumerate(pitchname_notes_train))
    int_to_dur = dict((number, note) for number, note in enumerate(pitchnames_dur_train))
    random_int_to_notes = [int_to_notes[char] for char in user_input_notes]
    random_int_to_dur = [int_to_dur[char] for char in user_input_duration]

  return {"user_input_notes": json.dumps(random_int_to_notes), "user_input_durations": json.dumps(random_int_to_dur)}

def Malody_Generator(Note_Count, X_notes, X_dur, user_input_notes, user_input_duration):
    ## if sequence already defined
    # seed_note = X_notes
    # seed_dur = X_dur
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchname_notes_train))
    dur_to_int = dict((dur, number)
                      for number, dur in enumerate(pitchnames_dur_train))

    user_input_notes = [note_to_int[char] for char in user_input_notes]
    user_input_duration = [dur_to_int[char] for char in user_input_duration]
    ## if random start
    start = np.random.randint(0, len(X_notes)-1)
    seed_note = X_notes[start]
    seed_dur = X_dur[start]
    #adding the randomized last notes to the end of prediction input and to beginning of output
    for i in range(0, len(user_input_notes)):
      seed_note[-len(user_input_notes)+i] = user_input_notes[0 +
                                                             i] / n_vocab_notes_test
      seed_dur[-len(user_input_duration) +
               i] = user_input_duration[0+i] / n_vocab_dur_test
    Notes_Generated = user_input_notes
    Durations_Generated = user_input_duration
    Music_notes = ''
    Music_durations = ''
    int_to_note = dict((number, note)
                       for number, note in enumerate(pitchname_notes_train))
    int_to_dur = dict((number, note)
                      for number, note in enumerate(pitchnames_dur_train))
    for i in range(Note_Count):
        # for notes
        seed_note = seed_note.reshape(1, 45, 1)
        prediction = model_notes.predict(seed_note, verbose=0)[0]
        prediction = np.log(prediction)  # diversity
        exp_preds = np.exp(prediction) / 1
        prediction = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction)
        index_N = index / float(n_vocab_notes_test)
        Notes_Generated.append(index)
        Music_notes = [int_to_note[char] for char in Notes_Generated]
        seed_note = np.insert(seed_note[0], len(seed_note[0]), index_N)
        seed_note = seed_note[1:]
      # for durations
        seed_dur = seed_dur.reshape(1, 45, 1)
        prediction_dur = model_dur.predict(seed_dur, verbose=0)[0]
        prediction_dur = np.log(prediction_dur)  # diversity
        exp_preds = np.exp(prediction_dur)
        prediction_dur = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction_dur)
        index_N = index / float(n_vocab_dur_test)
        Durations_Generated.append(index)
        Music_durations = [int_to_dur[char] for char in Durations_Generated]
        seed_dur = np.insert(seed_dur[0], len(seed_dur[0]), index_N)
        seed_dur = seed_dur[1:]
    #Now, we have music in form or a list of chords and notes and we want to be a midi file.
    # Melody = chords_n_notes(Music_notes, Music_durations)
    # Melody_midi = stream.Stream(Melody)
    return Music_notes, Music_durations
