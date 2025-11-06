from matplotlib import pyplot as plot
import os, sys
import librosa
from Binaspect import binaspect


def main():
    wav = "dataset_parallel/sample_0001.wav"
    data, sr = librosa.load(str(wav), sr=44100, mono=False)
    spec = binaspect.ITD_spect(data, sr, start_freq=10, stop_freq=1500, plots=True)
    print(spec.shape)

    plot.savefig('output_visualization.png', dpi=150, bbox_inches='tight')
    spec = binaspect.ILD_spect(data, sr, start_freq=5000, stop_freq=20000, plots=True)
    print(spec.shape)
    plot.savefig('output_visualization_levelD.png', dpi=150, bbox_inches='tight')


if __name__ == "__main__":
    main()
