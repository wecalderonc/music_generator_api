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

def create_song():
  # import ipdb
  # ipdb.set_trace()

  network_input_note_test = pickle_notes['network_input_test']
  network_input_dur_test = pickle_dur['network_input_test']
  Music_notes, Music_durations = Malody_Generator(50, network_input_note_test, network_input_dur_test)

  return { "notes": json.dumps(Music_notes), "durations": json.dumps(Music_durations) }

# def Malody_Generator(Note_Count):
#     pickle_model = pickle.load(open('models/wil_deploy_dummy.pickle', 'rb'))
#     network_input = pickle_model['network_input']
#     pitchnames = pickle_model['pitchnames']

#     seed = network_input[np.random.randint(0, len(network_input)-1)]
#     Music = ""
#     Notes_Generated = []
#     int_to_note = dict((number, note)
#                        for number, note in enumerate(pitchnames))
#     for i in range(Note_Count):
#         n_vocab = len(int_to_note)
#         seed = seed.reshape(1, 40, 1)
#         prediction = model.predict(seed, verbose=0)[0]
#         prediction = np.log(prediction)  # diversity
#         exp_preds = np.exp(prediction) / 1
#         prediction = exp_preds / np.sum(exp_preds)
#         index = np.argmax(prediction)
#         index_N = index / float(n_vocab)
#         Notes_Generated.append(index)
#         Music = [int_to_note[char] for char in Notes_Generated]
#         seed = np.insert(seed[0], len(seed[0]), index_N)
#         seed = seed[1:]
#     #Now, we have music in form or a list of chords and notes and we want to be a midi file.

#     #Melody = chords_n_notes(Music)
#     #Melody_midi = stream.Stream(Melody)
#     return Music


def Malody_Generator(Note_Count, X_notes, X_dur):
    pitchname_notes_train = pickle_notes['pitchnames_notes']
    n_vocab_notes_test = pickle_notes['n_vocab_notes']
    pitchnames_dur_train = pickle_dur['pitchnames_dur']
    n_vocab_dur_test = pickle_dur['n_vocab_dur']

    start = np.random.randint(0, len(X_notes)-1)
    # # seed = network_input[np.random.randint(0,len(network_input)-1)]
    seed_note = X_notes[start]
    seed_dur = X_dur[start]
    # seed_dur = X_dur
    # seed_note = X_notes
    Music_notes = ""
    Music_durations = ""
    Notes_Generated = []
    Durations_Generated = []
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
        # print(seed_note)
        # print(len(seed_note))
        seed_note = np.insert(seed_note[0], len(seed_note[0]), index_N)
        seed_note = seed_note[1:]
      # for durations
        seed_dur = seed_dur.reshape(1, 45, 1)
        prediction_dur = model_dur.predict(seed_dur, verbose=0)[0]
        # prediction_dur = np.log(prediction_dur) #diversity
        # exp_preds = np.exp(prediction_dur)
        # prediction_dur = exp_preds / np.sum(exp_preds)
        index = np.argmax(prediction_dur)
        index_N = index / float(n_vocab_dur_test)
        Durations_Generated.append(index)
        Music_durations = [int_to_dur[char] for char in Durations_Generated]
        # print(seed_dur)
        # print(len(seed_dur))
        seed_dur = np.insert(seed_dur[0], len(seed_dur[0]), index_N)
        seed_dur = seed_dur[1:]
    #Now, we have music in form or a list of chords and notes and we want to be a midi file.

    return Music_notes, Music_durations
