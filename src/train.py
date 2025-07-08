import pickle
from config import SEQUENCE_LENGTH, PROCESSED_NOTES_PATH
from preprocess import extract_notes, prepare_sequences
from model import create_model


def train_model():
    notes = extract_notes()
    X, y, note_to_int, int_to_note = prepare_sequences(notes, SEQUENCE_LENGTH)

    model = create_model(input_shape=(
        X.shape[1], X.shape[2]), output_size=y.shape[1])
    model.summary()

    model.fit(X, y, epochs=50, batch_size=64)

    # Save model + mappings
    model.save('../models/lstm_music_model.h5')
    with open('../models/note_mappings.pkl', 'wb') as f:
        pickle.dump({'note_to_int': note_to_int,
                    'int_to_note': int_to_note}, f)

    print("✅ Model trained and saved.")


if __name__ == "__main__":
    train_model()
