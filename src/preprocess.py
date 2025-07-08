from music21 import converter, instrument, note, chord
import numpy as np
from tensorflow.keras.utils import to_categorical
import glob
import pickle
import os


def extract_notes(midi_folder='../data/raw/', save_path='../data/processed/notes.pkl'):
    notes = []
    midi_files = glob.glob(f"{midi_folder}/*.mid")

    for file in midi_files:
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            elements = parts.parts[0].recurse() if parts else midi.flat.notes

            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(notes, f)
    return notes


def prepare_sequences(notes, sequence_length=50):
    # Get unique notes
    vocab = sorted(set(notes))
    note_to_int = {n: i for i, n in enumerate(vocab)}
    int_to_note = {i: n for n, i in note_to_int.items()}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(vocab))  # Normalize
    network_output = to_categorical(network_output)

    return network_input, network_output, note_to_int, int_to_note
