import torchaudio
import torchaudio.transforms as T
import torch
import matplotlib.pyplot as plt
import numpy as np

class btff_transoform:
    def __init__(self , input_audio_path , n_fft = 1024 , n_mels = 128 , fmin = 0 ,fmax = 16000):
        self.fmin = fmin
        self.fmax = fmax
        self.n_mels = n_mels
        self.n_fft = n_fft
        waveform, self.sr = torchaudio.load_with_torchcodec(input_audio_path)

        self.bin_width = self.sr/self.n_fft
        self.hop_size = round(n_fft/4)

        self.eps = 1e-6

        if waveform.shape[0] == 1:
            ValueError("the audio file that was passed in mono not stereo")
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]

        stft_transform = T.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.n_fft,
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

        self.mel_fb = T.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sr,
            f_min=self.fmin,
            f_max=self.fmax,
            n_stft=self.n_fft // 2 + 1,
            norm="slaney",
            mel_scale="slaney",

        )

    def ITD_spect(self , start_freq=0, stop_freq=1500):
        if start_freq < 0 or stop_freq > self.sr/2:
            raise ValueError("Invalid frequency range. Valid range is [0, {}]".format(self.sr/2))
        if start_freq >= stop_freq:
            raise ValueError("Invalid frequency range. Valid range is [0, {}]".format(self.sr/2))

        ITD_spectra = torch.zeros((self.numbins , self.numframes))

        startbin = int(round(start_freq/self.bin_width))
        stopbin = int(round(stop_freq/self.bin_width))

        for frame in range(self.numframes):
            for bin in range(startbin, stopbin):
                phasediff =  self.phasediffs[bin, frame]

                wrapped_phase_diff = torch.remainder(phasediff + torch.pi, 2 * torch.pi) - torch.pi

                if bin == 0:
                    bin = 1

                bindelay = wrapped_phase_diff / (2 * torch.pi * self.bin_width * bin)

                if self.intensity[bin , frame] >= self.eps:
                    ITD_spectra[bin, frame] = bindelay
        return self.mel_fb(ITD_spectra)


    def ILD_spect(self, stop_freq=5000):
        stopbin = int(round(stop_freq/self.bin_width))
        left_mag  = self.left_mag.clone()
        right_mag = self.right_mag.clone()

        left_mag[stopbin:, :] = self.eps
        right_mag[stopbin:, :] = self.eps

        ILD_spectra = (20 * torch.log10(left_mag)) - (20 * torch.log10(right_mag))

        return self.mel_fb(ILD_spectra)


    def vc_map(self , mel_left , mel_right):

        v_map_left = torch.zeros_like(mel_left)
        v_map_left[0] = mel_left[1] - mel_left[0]
        v_map_left[1:-1] = (mel_left[2:] - mel_left[:-2]) / 2
        v_map_left[-1] = mel_left[-1] - mel_left[-2]

        v_map_right = torch.zeros_like(mel_right)
        v_map_right[0] = mel_right[1] - mel_right[0]
        v_map_right[1:-1] = (mel_right[2:] - mel_right[:-2]) / 2
        v_map_right[-1] = mel_right[-1] - mel_right[-2]
        return v_map_left , v_map_right


    def mel_log_spectrogram_and_vmap(self):
        # mel_left = self.left_mag.pow(2)
        # mel_right = self.right_mag.pow(2)

        mel_left = self.mel_fb(self.left_mag)
        mel_right = self.mel_fb(self.right_mag)

        mel_left = torch.log10(mel_left)
        mel_right = torch.log10(mel_right)

        v_map_left , v_map_right = self.vc_map(mel_left  = mel_left , mel_right = mel_right)

        return mel_left , mel_right , v_map_left , v_map_right


    def sc_map(self, stop_freq=5000):

        stoptbin = int(round(stop_freq/self.bin_width))

        left_mag  = self.left_mag.clone()
        right_mag = self.right_mag.clone()

        left_mag[stoptbin:, :] = self.eps
        right_mag[stoptbin:, :] = self.eps

        # sc_map_left = left_mag.pow(2)
        # sc_map_right = right_mag.pow(2)

        sc_map_left = self.mel_fb(left_mag)
        sc_map_right = self.mel_fb(right_mag)

        sc_map_left = torch.log10(sc_map_left)
        sc_map_right = torch.log10(sc_map_right)

        return sc_map_left , sc_map_right













if __name__ == "__main__":

    path = "/home/baraa/Desktop/BAST/dataset_parallel/sample_0001.wav"

    # Initialize the BTFF transform
    btff = btff_transoform(path , fmax = 16000)

    # Generate all outputs
    itd_spectra = btff.ITD_spect(start_freq=0, stop_freq=1500)
    ild_spectra = btff.ILD_spect(stop_freq=5000)
    mel_left, mel_right, v_map_left, v_map_right = btff.mel_log_spectrogram_and_vmap()
    sc_map_left, sc_map_right = btff.sc_map(stop_freq=5000)

    # Create a figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle('BTFF Transform Outputs', fontsize=16, fontweight='bold')

    # Plot ITD Spectrogram
    im1 = axes[0, 0].imshow(itd_spectra.numpy(), aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_title('ITD Spectrogram')
    axes[0, 0].set_xlabel('Time Frames')
    axes[0, 0].set_ylabel('Frequency Bins')
    plt.colorbar(im1, ax=axes[0, 0], label='ITD (seconds)')

    # Plot ILD Spectrogram
    im2 = axes[0, 1].imshow(ild_spectra.numpy(), aspect='auto', origin='lower', cmap='RdBu_r')
    axes[0, 1].set_title('ILD Spectrogram')
    axes[0, 1].set_xlabel('Time Frames')
    axes[0, 1].set_ylabel('Frequency Bins')
    plt.colorbar(im2, ax=axes[0, 1], label='ILD (dB)')

    # Plot Mel Log Spectrogram (Left)
    im3 = axes[1, 0].imshow(mel_left.numpy(), aspect='auto', origin='lower', cmap='magma')
    axes[1, 0].set_title('Mel Log Spectrogram (Left)')
    axes[1, 0].set_xlabel('Time Frames')
    axes[1, 0].set_ylabel('Mel Bins')
    plt.colorbar(im3, ax=axes[1, 0], label='Log Magnitude')

    # Plot Mel Log Spectrogram (Right)
    im4 = axes[1, 1].imshow(mel_right.numpy(), aspect='auto', origin='lower', cmap='magma')
    axes[1, 1].set_title('Mel Log Spectrogram (Right)')
    axes[1, 1].set_xlabel('Time Frames')
    axes[1, 1].set_ylabel('Mel Bins')
    plt.colorbar(im4, ax=axes[1, 1], label='Log Magnitude')

    # Plot V-map (Left)
    im5 = axes[2, 0].imshow(v_map_left.numpy(), aspect='auto', origin='lower', cmap='seismic')
    axes[2, 0].set_title('Velocity Map (Left)')
    axes[2, 0].set_xlabel('Time Frames')
    axes[2, 0].set_ylabel('Mel Bins')
    plt.colorbar(im5, ax=axes[2, 0], label='finite difference')

    # Plot V-map (Right)
    im6 = axes[2, 1].imshow(v_map_right.numpy(), aspect='auto', origin='lower', cmap='seismic')
    axes[2, 1].set_title('Velocity Map (Right)')
    axes[2, 1].set_xlabel('Time Frames')
    axes[2, 1].set_ylabel('Mel Bins')
    plt.colorbar(im6, ax=axes[2, 1], label='finite difference')

    # Plot SC-map (Left)
    im7 = axes[3, 0].imshow(sc_map_left.numpy(), aspect='auto', origin='lower', cmap='plasma')
    axes[3, 0].set_title('SC Map (Left)')
    axes[3, 0].set_xlabel('Time Frames')
    axes[3, 0].set_ylabel('Mel Bins')
    plt.colorbar(im7, ax=axes[3, 0], label='Log Magnitude')

    # Plot SC-map (Right)
    im8 = axes[3, 1].imshow(sc_map_right.numpy(), aspect='auto', origin='lower', cmap='plasma')
    axes[3, 1].set_title('SC Map (Right)')
    axes[3, 1].set_xlabel('Time Frames')
    axes[3, 1].set_ylabel('Mel Bins')
    plt.colorbar(im8, ax=axes[3, 1], label='Log Magnitude')

    # Adjust layout and display
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    plt.show()
