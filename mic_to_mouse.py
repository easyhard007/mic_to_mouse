import pynput.mouse
from pynput.mouse import Button
import pynput.keyboard
from multiprocessing import Process,Value,Manager
import numpy as np
import math
import pyaudio
import json
import scipy.fft
import matplotlib.pyplot as plt
import time
from pynput import keyboard

######################################################################
# mic_to_mouse.py - control the mouse using audio from the microphone
######################################################################
# Author:  Yinan Chen
# Date:    July 2022
# Codes are partially based on: https://github.com/mzucker/python-tuner
######################################################################

# ref base frequency of a melodica
number_to_freq_dict = {
    60: 262.3,
    61: 277.2,
    62: 292.3,
    63: 309.8,
    64: 329.3,
    65: 348.1,
    66: 368.4,
    67: 388.0,
    68: 411.1,
    69: 437.3,
    70: 464.9,
    71: 494.3,
    72: 521.1,
    73: 553.3,
    74: 586.0,
    75: 620.7,
    76: 653.8,
    77: 688.4,
    78: 735.0,
    79: 776.5,
    80: 824.7,
    81: 873.3,
    82: 928.0,
    83: 988.3,
    84: 1042.0
}

# amps(weights) of each harmonics of a melodica
weights = [0.8, 0.4, 1.0, 0.7, 1.1, 0.9, 1.4, 0.2, 0.3, 0.1]

def freq_to_number(f): return 69 + 12 * np.log2(f / 440.0)

def number_to_freq(n): return 440 * 2.0 ** ((n - 69) / 12.0)

def note_name(n): return NOTE_NAMES[n % 12] + str(math.floor(n / 12) - 1)

def note_to_fftbin(n): return number_to_freq(n) / FREQ_STEP

def index_to_freq(index, imin, FREQ_STEP): return (index + imin) * FREQ_STEP

def freq_to_index(freq, imin, FREQ_STEP): return round(freq / FREQ_STEP - imin)

#
def listen_keyboard(params):
    q = params
    with pynput.keyboard.Events() as event:
        for i in event:
            key_event = i
            print(key_event.key)
            if(key_event.key==keyboard.Key.esc): # press esc to exit the program
                q.put(-1)
                return False
            if(key_event.key==keyboard.Key.pause): # pause
                q.put(0)
            if (key_event.key == keyboard.Key.insert): # resume
                q.put(1)


# map a value from input_range to output_y_range, adjust [983,171] to fit your screen resolution(mine is 2560 x 1440)
def map_input(value,input_range=[60,84],output_y_range=[983,171]):
    pos = (value-input_range[0]) / (input_range[1]-input_range[0])
    return output_y_range[0]+ (output_y_range[1]-output_y_range[0]) * pos


# a feature file describes the highest k frequencies of each note of a musical instrument (ref. melodica_freqs.txt)
# the keys (indexes) of these frequencies are the index of them in fft buffer (list).
# the weight are the average amplitudes of these frequencies, normalized to make sure that the sum of them is 1.0
def write_features_to_file(path, d):
    f = open(path, 'a',encoding='utf-8')
    json.dump(d, f)
    f.close()

def read_features_from_file(path):
    f = open(path, 'r',encoding='utf-8')
    s = json.load(f)
    print(s)
    f.close()
    return s

# given frequencie feature of notes and current fft, find the note that best fit current fft
# freq_indexes: list of the first k freq indexes(in fft) for a note
# weights: list of weight corresponding to the freqs
def note_score(fft,freq_indexes,weights):
    scores = []
    for i in range(0,len(freq_indexes)):
        if freq_indexes[i]>=len(fft):
            print("warning:",freq_indexes[i],"larger than fft size")
            break
        scores.append(abs(fft[freq_indexes[i]])*weights[i])
    return np.mean(scores)

# calc the first k harmonic frequencies of a note
def harms_of_note(note_index,k):
    harms = []
    base = number_to_freq_dict[note_index]
    for i in range(1,k+1):
        harms.append(base*i)
    return harms


if __name__ == '__main__':

    NOTE_MIN = 60 # C5, the lowest note you want to detect
    NOTE_MAX = 84  # C7, the highest note you want to detect
    FSAMP = 22050 # Sampling frequency in Hz
    FRAME_SIZE = 128  # How many samples per frame
    FRAMES_PER_FFT = 8  # FFT takes average across how many frames
    SAMPLES_PER_FFT = FRAME_SIZE * FRAMES_PER_FFT
    FREQ_STEP = float(FSAMP) / SAMPLES_PER_FFT

    NOTE_NAMES = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']

    print(NOTE_NAMES)

    print("note_to_fftbin(60)",note_to_fftbin(60))

    imin = max(0, int(np.floor(note_to_fftbin(NOTE_MIN-1))))
    imax = min(SAMPLES_PER_FFT, int(np.ceil(note_to_fftbin(NOTE_MAX+1))))

    print("imin",imin,"imax",imax)

    buf = np.zeros(SAMPLES_PER_FFT, dtype=np.float32)
    num_frames = 0

    # Initialize audio device
    p = pyaudio.PyAudio()

    # print devices info, change the "input_device_index=22" to the index of the device as the input
    for i in range(0,p.get_device_count()):
        print(p.get_device_info_by_index(i))

    stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                    channels=1,
                                    rate=FSAMP,
                                    input=True,
                                    input_device_index=22, #use fl studio asio device (index 22), if deleted, default device will be selected
                                    frames_per_buffer=FRAME_SIZE)
    stream.start_stream()

    # Create Hanning window function
    window = 0.5 * (1 - np.cos(np.linspace(0, 2*np.pi, SAMPLES_PER_FFT, False)))


    # Listen the input of the keyboard and Control the mouse
    mouse = pynput.mouse.Controller()
    q = Manager().Queue()
    p1 = Process(target=listen_keyboard,args=(q,))  # 必须加,号
    p1.start()

    # status of the program, can be 1(running), 0(pause) and -1(trigger exit)
    active = 1

    # build the freqs list of each note, which saves the index of each harmonic frequency in FFT window (starting from NOTE_MIN)
    note_list = range(NOTE_MIN,NOTE_MAX+1)
    freqs_dict = {}
    for note in note_list:
        freqs_dict[note] = []
        for freq in harms_of_note(note,10):
            index = freq_to_index(freq,imin,FREQ_STEP)
            freqs_dict[note].append(index)

    last_scan_time = time.time()

    while stream.is_active():

        if not q.empty():
            status = q.get()

            if (status==0): # pause
                active = 0
            if (status==1): # resume
                active = 1
            if (status==-1): # exit
                break

        if active==0:
            continue

        # Shift the buffer down and new data in
        buf[:-FRAME_SIZE] = buf[FRAME_SIZE:]
        buf[-FRAME_SIZE:] = np.fromstring(stream.read(FRAME_SIZE,exception_on_overflow = False), np.int16)

        # Run the FFT on the windowed buffer
        fft = scipy.fft.fft(buf * window)

        # used as the volume of input
        amp = max(np.abs(fft[imin:imax]))

        # Console output once we have a full buffer
        num_frames += 1

        # print(time.time()-last_scan_time)
        last_scan_time=time.time()


        if num_frames >= FRAMES_PER_FFT:

            # volume larger than 50000 will trigger the note detecting and mouse control
            if (amp > 50000):
                scores = []
                # calc the score of each note
                for note_index in note_list:
                    score = note_score(fft[imin:],freqs_dict[note_index],weights)
                    scores.append(score)

                scores_index = np.array(scores).argmax()
                best_fit_note_index = note_list[scores_index]
                # print(note_name(best_fit_note_index), amp, scores[scores_index])

                mouse.position = (mouse.position[0], math.floor(map_input(best_fit_note_index,input_range=[NOTE_MIN,NOTE_MAX],output_y_range=[983,171])))
                mouse.press(Button.left)  # press left button of the mouse

            # volume smaller than 20000 will release the mouse
            if (amp < 20000):
                mouse.release(Button.left)  # release left button of the mouse





