from mne.time_frequency import psd_multitaper
import numpy as np
from scipy.interpolate import griddata

def getimages(epochs,layout,imgsize=32):

    #compute power bands
    psds, freqs = psd_multitaper(epochs)
    psds /= psds.sum(axis=-1)[..., None] #normalize

    #power band masks
    masks = {}
    masks['theta'] = (4 < freqs) & (freqs < 7)
    masks['alpha'] = (8 < freqs) & (freqs < 13)
    masks['beta'] = (13 < freqs) & (freqs < 30)

    #extract positions of electrodes
    chs = epochs.info['chs']
    chnames = np.array([ch['ch_name'].lower() for ch in chs])
    ltnames = np.array([name.lower() for name in layout.names])
    indices = np.squeeze([np.where(ltnames==name) for name in chnames])
    positions = layout.pos[indices][:,:2]

    #square grid for interpolation
#    grid_x, grid_y = np.mgrid[min(positions[0]):max(positions[0]):imgsize*1j,
#                              min(positions[1]):max(positions[1]):imgsize*1j]
    grid_x, grid_y = np.mgrid[0.:1.:imgsize*1j,
                              0.:1.:imgsize*1j]
    
    rgbimages = []
    for ii,psd in enumerate(psds):
        
        rgbimage = np.stack([
            griddata(positions, psd[:,masks[band]].sum(axis=1),
                     (grid_x,grid_y),
                     method='cubic', fill_value=np.nan)
            for band in ['theta','alpha','beta']]).T
        
        rgbimages.append(rgbimage)

    return rgbimages
