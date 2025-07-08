from generate import generate_notes
from midi_utils import notes_to_midi

notes = generate_notes(num_notes=100)
notes_to_midi(notes)
