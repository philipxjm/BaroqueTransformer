import mido

mid = mido.MidiFile("data/fp-1all.mid")

for i, track in enumerate(mid.tracks):
    if track.name == "Solo Cello":
        for msg in track:
            if msg.type == 'note_on':
                print(msg)
