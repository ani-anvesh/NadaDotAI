import pickle
from preprocess import extract_notes, prepare_sequences
from config import SEQUENCE_LENGTH, PROCESSED_NOTES_PATH

notes = extract_notes()
X, y, note_to_int, int_to_note = prepare_sequences(notes, SEQUENCE_LENGTH)

print("Input shape:", X.shape)
print("Output shape:", y.shape)
