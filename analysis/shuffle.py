import numpy as np

def circular(spikes):
    """
    Circularly shuffles a (neur, frames) array of spikes, neuron by neuron.
    """

    spikes = spikes.copy()
    shift = np.random.choice(np.arange(1, spikes.size))
    if len(spikes.shape) == 2:
        num_neur, num_frames = spikes.shape
        for neur in range(num_neur):
            shift = np.random.choice(np.arange(1, num_frames))
            spikes[neur,:] = np.roll(spikes[neur,:], shift)
        return spikes
    else:
        return np.roll(spikes, shift)

