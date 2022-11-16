import pynput.mouse
from pynput.mouse import Button
import pynput.keyboard
from multiprocessing import Process,Value,Manager
import numpy as np
import math
import pyaudio
import json
import librosa
import scipy.fft
import matplotlib.pyplot as plt
import time
from pynput import keyboard
from scipy.signal import find_peaks

######################################################################
# mic_to_mouse.py - control the mouse using audio from the microphone
######################################################################
# Author:  Yinan Chen
# Date:    July 2022
# Codes are partially based on: https://github.com/mzucker/python-tuner
######################################################################



# volume threshold of detecting
amp_threshold_on = 50000
amp_threshold_off = 30000

# 440.0 Hz (+0.0) by default
tune= +0.0


def freq_to_number(f): return 69 + 12 * np.log2(f / 440.0+tune)

def number_to_freq(n): return (440+tune) * 2.0 ** ((n - 69) / 12.0)

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



if __name__ == '__main__':

    NOTE_MIN = 60 # C5, the lowest note you want to detect
    NOTE_MAX = 84  # C7, the highest note you want to detect
    FSAMP = 22050 # Sampling frequency in Hz
    FRAME_SIZE = 256 # How many samples per frame
    FRAMES_PER_FFT = 16  # FFT takes average across how many frames
    SAMPLES_PER_FFT = FRAME_SIZE * FRAMES_PER_FFT
    FREQ_STEP = float(FSAMP) / SAMPLES_PER_FFT

    NOTE_NAMES = ['C','C#','D','Eb','E','F','F#','G','Ab','A','Bb','B']

    print(NOTE_NAMES)

    print("note_to_fftbin(60)",note_to_fftbin(60))
    print("freq_to_index(261.63)",freq_to_index(261.63,1,FREQ_STEP))

    freq_index_min = freq_to_index(number_to_freq(NOTE_MIN), 0, FREQ_STEP)
    freq_index_max = freq_to_index(number_to_freq(NOTE_MAX), 0, FREQ_STEP)

    buf = np.zeros(SAMPLES_PER_FFT, dtype=np.int16)
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
        absfft = np.abs(fft)
        # used as the volume of input
        amp = max(absfft[freq_index_min:freq_index_max*5])

        # Console output once we have a full buffer
        num_frames += 1

        # print(time.time()-last_scan_time)
        last_scan_time=time.time()


        if num_frames >= FRAMES_PER_FFT:

            # volume larger than 50000, trigger the note detecting and mouse control
            if (amp > amp_threshold_on):

                peaks = find_peaks(absfft[:FRAME_SIZE*3],prominence=amp/10)[0]

                peaks = [peak for peak in peaks if (peak < freq_index_max and peak >= freq_index_min - 1)]

                if len(peaks)==0: continue

                note_detected = round(freq_to_number(index_to_freq(peaks[0],0,FREQ_STEP)))
                print(note_name(note_detected),amp)

                # DEBUG
                # if(note_name(note_detected)=='E4'):
                #     plt.plot(absfft[:FRAME_SIZE*3])
                #     plt.show()


                # the mouse will move between (983,171) for my full screen resolution, output_y_range for your resolution
                mouse.position = (mouse.position[0], math.floor(map_input(note_detected,input_range=[NOTE_MIN,NOTE_MAX],output_y_range=[983,171])))
                mouse.press(Button.left)  # press left button of the mouse

            # volume smaller than 20000, release the mouse
            if (amp < amp_threshold_off):
                mouse.release(Button.left)  # release left button of the mouse





