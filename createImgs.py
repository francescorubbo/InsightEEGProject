import numpy as np

from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.time_frequency import psd_multitaper

from scipy.interpolate import griddata

from PIL import Image,ImageOps

import argparse
parser = argparse.ArgumentParser(description='Process subject.')
parser.add_argument('subject',type=int,default=1)
args = parser.parse_args()

tmin, tmax = -1., 4.
event_id = dict(hands=2, feet=3)

subject = args.subject
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
             raw_fnames]
raw = concatenate_raws(raw_files)

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))

# Apply band-pass filter
raw.filter(2, 35., fir_design='firwin', skip_by_annotation='edge')

events = find_events(raw, shortest_event=0, stim_channel='STI 014')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)

from ImgUtils import getimages

layout = read_layout('EEG1005')
images = getimages(epochs,layout)

basepath = '../data/images/subject%d/'%subject
pathsig = basepath+'signal/'
pathbkg = basepath+'background/'

import os
for directory in [pathsig,pathbkg]:
    if not os.path.exists(directory):
        os.makedirs(directory)

labels = epochs.events[:, -1] - 2

for ii,(l,img) in enumerate(zip(labels,images)):
    path = pathsig if l else pathbkg
    rgbimage = Image.fromarray(np.uint8(img*255),'RGB')
    rgbimage.save(path+'/img%s.jpg'%ii)
