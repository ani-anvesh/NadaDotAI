from music21 import stream, note, chord


def notes_to_midi(note_sequence, output_file='../outputs/generated_output.mid'):
    output_notes = []

    for pattern in note_sequence:
        if '.' in pattern or pattern.isdigit():  # It's a chord
            notes_in_chord = [note.Note(int(n)) for n in pattern.split('.')]
            new_chord = chord.Chord(notes_in_chord)
            new_chord.duration.quarterLength = 0.5
            output_notes.append(new_chord)
        else:  # It's a note
            new_note = note.Note(pattern)
            new_note.duration.quarterLength = 0.5
            output_notes.append(new_note)

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"✅ MIDI file saved: {output_file}")
