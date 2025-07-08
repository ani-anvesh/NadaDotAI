import pickle
import numpy as np
from tensorflow.keras.models import load_model
from config import SEQUENCE_LENGTH


def generate_notes(model_path='../models/lstm_music_model.h5',
                   mappings_path='../models/note_mappings.pkl',
                   sequence_length=SEQUENCE_LENGTH,
                   num_notes=100):

    with open(mappings_path, 'rb') as f:
        mappings = pickle.load(f)

    note_to_int = mappings['note_to_int']
    int_to_note = mappings['int_to_note']
    vocab_size = len(note_to_int)

    model = load_model(model_path)

    # Pick random seed
    start_index = np.random.randint(0, len(note_to_int) - sequence_length)
    pattern = list(note_to_int.values())[
        start_index: start_index + sequence_length]
    generated = []

    for _ in range(num_notes):
        input_seq = np.reshape(pattern, (1, sequence_length, 1))
        input_seq = input_seq / float(vocab_size)

        prediction = model.predict(input_seq, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        generated.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    return generated
