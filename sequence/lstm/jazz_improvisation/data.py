"""Modified from the original data_utils.py file from the Deep Learning Specialization assignment"""
from grammar import unparse_grammar
from music import data_processing
from music21 import *
from preprocess import get_corpus_data, get_musical_data
from qa import clean_up_notes, prune_grammar, prune_notes

chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)


def load_data():
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)

    return (X, Y, N_tones, indices_tones)


def generate_music(sampling_fn):
    """
    Generates music using the given sampling function that returns the indices of musical values.
    Creates an audio stream to save the music.
    """

    # Set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0  # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)  # number of different set of chords
    
    print('Predicting new values for different set of chords.')
    # Loop over all 18 set of chords. At each iteration generate a sequence of tones
    # and use the current chords to convert it into actual sounds 
    for i in range(1, num_chords):
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        # Loop over the chords of the current set of chords
        for j in chords[i]:
            # Add chord to the current chords with the adequate offset, no need to understand this
            curr_chords.insert((j.offset % 4), j)

        # Generate a sequence of tones using the model
        indices = sampling_fn()
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
                
        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones. It is a common choice.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = prune_grammar(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        num_sounds = len([k for k in sounds if isinstance(k, note.Note)])
        print(f'Generated {num_sounds} sounds using the predicted values for the set of chords ("{i}") and after pruning')
        
        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to fine
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open('output/jazz.midi', 'wb')
    mf.write()
    print('Generated music is saved in output/jazz.midi')
    mf.close()
