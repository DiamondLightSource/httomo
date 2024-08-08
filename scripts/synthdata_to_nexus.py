#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
* Script to generate 3D analytical phantoms and their projection data using TomoPhantom
* Synthetic flat fields are also genererated and noise incorporated into data 
together with normalisation errors. This simulates more challeneging data for 
reconstruction.
* the data is rearranged and saved in the NeXus format

@author: Daniil Kazantsev
"""
import timeit
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py

import tomophantom
from tomophantom import TomoP3D
from tomophantom.flatsgen import synth_flats

print("Building 3D phantom using TomoPhantom software")
tic = timeit.default_timer()
model = 13  # select a model number from the library
N_size = 256  # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "phantomlib", "Phantom3DLibrary.dat")
# This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)
toc = timeit.default_timer()
Run_time = toc - tic
print("Phantom has been built in {} seconds".format(Run_time))

sliceSel = int(0.5 * N_size)
# plt.gray()
plt.figure()
plt.subplot(131)
plt.imshow(phantom_tm[sliceSel, :, :], vmin=0, vmax=1)
plt.title("3D Phantom, axial view")

plt.subplot(132)
plt.imshow(phantom_tm[:, sliceSel, :], vmin=0, vmax=1)
plt.title("3D Phantom, coronal view")

plt.subplot(133)
plt.imshow(phantom_tm[:, :, sliceSel], vmin=0, vmax=1)
plt.title("3D Phantom, sagittal view")
plt.show()

# Projection geometry related parameters:
Horiz_det = int(np.sqrt(2) * N_size)  # detector column count (horizontal)
Vert_det = N_size  # detector row count (vertical) (no reason for it to be > N)
angles_num = int(0.5 * np.pi * N_size)
# angles number
angles = np.linspace(0.0, 179.9, angles_num, dtype="float32")  # in degrees
angles_rad = angles * (np.pi / 180.0)
# %%
print("Building 3D analytical projection data with TomoPhantom")
projData3D_analyt = TomoP3D.ModelSino(
    model, N_size, Horiz_det, Vert_det, angles, path_library3D
)

intens_max_clean = np.max(projData3D_analyt)
sliceSel = (int)(N_size * 0.5)
plt.figure()
plt.subplot(131)
plt.imshow(projData3D_analyt[:, sliceSel, :], vmin=0, vmax=intens_max_clean)
plt.title("2D Projection (analytical)")
plt.subplot(132)
plt.imshow(projData3D_analyt[sliceSel, :, :], vmin=0, vmax=intens_max_clean)
plt.title("Sinogram view")
plt.subplot(133)
plt.imshow(projData3D_analyt[:, :, sliceSel], vmin=0, vmax=intens_max_clean)
plt.title("Tangentogram view")
plt.show()
# %%
print(
    "Simulate synthetic flat fields, add flat field background to the projections and add noise"
)
I0 = 50000
# full-beam photon flux intensity
flatsnum = 20  # the number of the flat fields required

[projData3D_raw, flats, speckel_map] = synth_flats(
    projData3D_analyt,
    source_intensity=I0,
    detectors_miscallibration=0.05,
    variations_number=3,
    arguments_Bessel=(1, 25),
    specklesize=2,
    kbar=2,
    jitter_projections=0.0,
    sigmasmooth=3,
    flatsnum=flatsnum,
)
# del projData3D_analyt
plt.figure()
plt.subplot(121)
plt.imshow(projData3D_raw[:, 0, :])
plt.title("2D Projection (before normalisation)")
plt.subplot(122)
plt.imshow(flats[:, 0, :])
plt.title("A selected simulated flat-field")
plt.show()
# %%
projData3D_raw = np.require(np.swapaxes(projData3D_raw, 0, 1), requirements="C")
projData3D_analyt = np.require(np.swapaxes(projData3D_analyt, 0, 1), requirements="C")
flats = np.require(np.swapaxes(flats, 0, 1), requirements="C")
# %%
# generating darks
from tomophantom.artefacts import noise
darks = np.ones(np.shape(flats), dtype="uint16")
darks = noise(darks, 0.00001, noisetype="Gaussian")
darks /= np.max(darks)
darks = np.uint16(darks * 65535)
darks -= np.min(darks)
#plt.figure()
#plt.imshow(darks[10, :, :])
#plt.show()
# %%
data_full = np.zeros((angles_num + 2*flatsnum, Vert_det, Horiz_det), dtype="uint16")
data_full[:angles_num, :, :] = projData3D_raw
data_full[angles_num:angles_num+flatsnum, :, :] = flats
data_full[angles_num+flatsnum::, :, :] = darks
# %%
# crop the data vertically
start_ind_c = 85
end_ind_c = 170
darks = darks[:,start_ind_c:end_ind_c]
flats = flats[:,start_ind_c:end_ind_c]
data_full = data_full[:,start_ind_c:end_ind_c,:]
projData3D_analyt = projData3D_analyt[:,start_ind_c:end_ind_c,:]
phantom_tm = phantom_tm[start_ind_c:end_ind_c,:,:]
# %%
# extend angles (degrees)
angles_full = np.zeros(angles_num + 2*flatsnum, dtype="float32")
angles_full[:angles_num] = angles
angles_full[angles_num::] = 179.9
#%%
# https://tomotools.gitlab-pages.esrf.fr/nxtomo/index.html
# https://tomotools.gitlab-pages.esrf.fr/nxtomo/tutorials/create_from_scratch.html
import nxtomo
from nxtomo import NXtomo
from nxtomo.nxobject.nxdetector import ImageKey

my_nxtomo = NXtomo()

# then register the data to the detector
my_nxtomo.instrument.detector.data = data_full


image_key_control = np.concatenate([
    [ImageKey.PROJECTION] * angles_num,
    [ImageKey.FLAT_FIELD] * flatsnum,
    [ImageKey.DARK_FIELD] * flatsnum,    
])
print("flats indexes are", np.where(image_key_control == ImageKey.FLAT_FIELD))
# then register the image keys to the detector
my_nxtomo.instrument.detector.image_key_control = image_key_control


my_nxtomo.sample.rotation_angle = -angles_full

my_nxtomo.instrument.detector.field_of_view = "Full"
#%%
os.makedirs("output_nexus", exist_ok=True)
nx_tomo_file_path = os.path.join("output_nexus", "nxtomo.nxs")
my_nxtomo.save(file_path=nx_tomo_file_path, data_path="entry1/tomo_entry", overwrite=True)
del my_nxtomo
#%%
# adding more data to the existing file
h5f = h5py.File('/home/algol/output_nexus/nxtomo.nxs', 'r+')
h5f.create_dataset('/entry1/tomo_entry/data/ground_truth_projections', data=projData3D_analyt)
h5f.create_dataset('/entry1/tomo_entry/data/phantom', data=phantom_tm)
h5f.close()
# %%