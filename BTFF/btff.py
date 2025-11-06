import torchaudio
import torchaudio.transforms as T
import torch

class btff_transoform:
    def __init__(self , input_audio_path , win_length = 512):
        self.win_length = win_length
        waveform, self.sr = torchaudio.load(input_audio_path)

        self.bin_width = self.sr/self.win_length
        self.hop_size = round(win_length/4)

        self.eps = 1e-6

        if waveform.shape[0] == 1:
            ValueError("the audio file that was passed in mono not stereo")
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        stft_transform = T.Spectrogram(
            n_fft=self.win_length,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            power=None,  # None returns complex spectrogram
            normalized=False,
            center=True,
            pad_mode="reflect",
            onesided=True,
        )
        self.complex_spectrogram_left = stft_transform(waveform[0])
        self.complex_spectrogram_right = stft_transform(waveform[1])

        self.left_mag = torch.abs( self.complex_spectrogram_left)
        self.left_phase = torch.angle(self.complex_spectrogram_left)

        self.right_mag = torch.abs(self.complex_spectrogram_right)
        self.right_phase = torch.angle(self.complex_spectrogram_right)

        self.left_mag = torch.clamp_min(self.left_mag, min=self.eps)
        self.right_mag = torch.clamp_min(self.right_mag, min=self.eps)

        [self.numbins, self.numframes] =  self.left_mag.shape
        # self.spectra =torch.zeros((self.numbins , self.numframes)) for reference on  how to initilizse a spectra space
        self.intensity = self.left_mag + self.right_mag
        self.phasediffs = self.left_phase - self.right_phase



    def ITD_spect(self , start_freq=50, stop_freq=1500):
        if start_freq < 0 or stop_freq > self.sr/2:
            raise ValueError("Invalid frequency range. Valid range is [0, {}]".format(self.sr/2))
        if start_freq >= stop_freq:
            raise ValueError("Invalid frequency range. Valid range is [0, {}]".format(self.sr/2))

        ITD_spectra = torch.zeros((self.numbins , self.numframes))

        startbin = int(torch.round(start_freq/self.bin_width))
        stopbin = int(torch.round(stop_freq/self.bin_width))

        for frame in range(self.numframes):
            for bin in range(startbin, stopbin):
                phasediff =  self.phasediffs[bin, frame]

                wrapped_phase_diff = torch.remainder(phasediff + torch.pi, 2 * torch.pi) - torch.pi

                if bin == 0:
                    bin = 1

                bindelay = wrapped_phase_diff / (2 * torch.pi * self.bin_width * bin)

                if self.intensity[bin , frame] >= self.eps:
                    ITD_spectra[bin, frame] = bindelay
        return ITD_spectra





    def ILD_spect(self, start_freq=5000, stop_freq=20000):
        if start_freq <= 0 or stop_freq > self.sr/2:
            raise ValueError("Invalid frequency range. Valid range is [0, {}]".format(self.sr/2))
        if start_freq >= stop_freq:
            raise ValueError("Invalid frequency range. Valid range is [0, {}]".format(self.sr/2))
        startbin = int(torch.round(start_freq/self.bin_width))
        stopbin = int(torch.round(stop_freq/self.bin_width))

        ILD_spectra = (20 * torch.log10(self.left_mag)) - (20 * torch.log10(self.right_mag))








if __name__ == "__main__":
