# NadaDotAI 🎵 – LSTM Music Generator

This project uses a Long Short-Term Memory (LSTM) neural network to generate music based on MIDI data. The model learns patterns from a set of training MIDI files and then produces new compositions by predicting note sequences.

---

## 🧠 Project Workflow

1. **Extract MIDI Notes** – Parses `.mid` files to get sequences of notes and chords.
2. **Preprocess Sequences** – Converts notes into integer sequences and prepares input/output for the LSTM.
3. **Train LSTM Model** – Builds and trains an LSTM on the sequences.
4. **Generate New Music** – Uses trained model to predict new sequences of notes.
5. **Convert to MIDI** – Outputs a new `.mid` file from the predicted notes.
6. **(Optional) Streamlit UI** – Adds a web-based interface to interact with the generator.

---

## 📁 Folder Structure

```
NadaDotAI/
├── data/
│   ├── raw/                # Original MIDI files
│   └── processed/          # Pickled note sequences
├── models/                 # Saved models (.h5, mappings)
├── notebooks/              # Prototyping notebooks
├── outputs/                # Generated MIDI files
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── midi_utils.py
├── app/                    # Streamlit app
├── requirements.txt
└── README.md
```

---

## 🔮 Future Enhancements

- [ ] **Real-time Music Generation** – Predict and stream notes live with `mido` or `pygame.midi`.
- [ ] **Transformer Model** – Replace LSTM with Music Transformer for longer context generation.
- [ ] **User-provided MIDI Upload** – Allow users to upload their own MIDI and train/generate from it.
- [ ] **Genre Conditioning** – Let model learn based on genre tags and generate in specific styles.
- [ ] **Streamlit App** – Upload file, generate music, and preview or download `.mid` output via UI.
- [ ] **Keyboard Input** – Play a short melody and let the model continue it.
- [ ] **Visualizer** – Display generated notes using `music21` or MIDI visualization tools.
- [ ] **Cloud Deploy (HuggingFace/Render)** – Make it public-facing with a persistent backend.

---

## 🚀 Getting Started

1. Place MIDI files inside `data/raw/`
2. Run `train_model()` from `src/train.py`
3. Generate with `generate_notes()` from `src/generate.py`
4. Convert output to MIDI with `notes_to_midi()` from `src/midi_utils.py`

---

## 🧰 Requirements

- Python 3.8+
- TensorFlow
- music21
- numpy
- pickle
- mido (for future real-time playback)
- streamlit (for optional UI)

---

## 📝 Author

NadaDotAI | Built with love and melody 💙🎼
