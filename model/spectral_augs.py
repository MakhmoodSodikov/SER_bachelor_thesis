import torch
import numpy as np

from scipy.signal import stft, istft, sawtooth
from acclib.transforms.base import BaseTransform


class RandomSpecAugTransform(BaseTransform):
    def __init__(self,
                 aug_type="noise",
                 noise_mean=0.0,
                 noise_dev=0.01,
                 saw_period=2.0,
                 saw_amplitude=1.0,
                 saw_omega=5,
                 sine_amplitude=1.0,
                 prob=1.0):
        super().__init__(prob)
        self.aug_type = aug_type
        self.noise_mean = noise_mean
        self.noise_dev = noise_dev
        self.saw_period = saw_period
        self.saw_amplitude = saw_amplitude
        self.saw_omega = saw_omega
        self.sine_amplitude = sine_amplitude
        self.prob = prob

        self.augs_pull = {
            "noise": SpecAugNoiseTransform(noise_mean=self.noise_mean,
                                           noise_dev=self.noise_dev,
                                           prob=self.prob),
            "saw": SpecAugSawToothTransform(period=self.saw_period,
                                            amplitude=self.saw_amplitude,
                                            omega=self.saw_omega,
                                            prob=self.prob),
            "sine": SpecAugSineTransform(amplitude=self.sine_amplitude,
                                         prob=self.prob),
        }

    def _transform(self, item):
        self.augs_pull[self.aug_type]._transform(item)


class SpecAugNoiseTransform(BaseTransform):
    def __init__(self, noise_mean, noise_dev, prob=1.0):
        super().__init__(prob)
        self.mean = noise_mean
        self.sigma = noise_dev

    def _transform(self, item):
        if 'acc_s' in item:
            item['acc_s'] = self.add_spec_noise(item['acc_s'])
        if 'gyro_s' in item:
            item['gyro_s'] = self.add_spec_noise(item['gyro_s'])

    def add_spec_noise(self, signal):
        noised_signal = np.empty_like(signal)

        for axis in range(signal.shape[1]):
            sig_1d = signal[:, axis]
            fft = stft(sig_1d, nperseg=signal.shape[0])[2]
            noise = np.random.normal(self.mean, self.sigma, fft.shape)
            fft_noised = fft + noise
            noised_signal[:, axis] = istft(fft_noised)[1]

        return torch.Tensor(noised_signal)


class SpecAugSawToothTransform(BaseTransform):
    def __init__(self, period, amplitude, omega, prob=1.0):
        super().__init__(prob)
        self.period = period
        self.amplitude = amplitude
        self.omega = omega

    def _transform(self, item):
        if 'acc_s' in item:
            item['acc_s'] = self.add_spec_noise(item['acc_s'])
        if 'gyro_s' in item:
            item['gyro_s'] = self.add_spec_noise(item['gyro_s'])

    def add_spec_noise(self, signal):
        noised_signal = np.empty_like(signal)

        for axis in range(signal.shape[1]):
            sig_1d = signal[:, axis]
            fft = stft(sig_1d, nperseg=sig_1d.shape[0])[2]
            fft_noised = np.empty_like(fft)

            for i in range(fft.shape[1]):
                t = np.linspace(0, self.period, fft.shape[0])
                noise = self.amplitude * sawtooth(2 * np.pi * self.omega * t)
                fft_noised[:, i] = fft[:, i] + noise

            noised_signal[:, axis] = istft(fft_noised)[1]

        return torch.Tensor(noised_signal)


class SpecAugSineTransform(BaseTransform):
    def __init__(self, amplitude, prob=1.0):
        super().__init__(prob)
        self.amplitude = amplitude

    def _transform(self, item):
        if 'acc_s' in item:
            item['acc_s'] = self.add_spec_noise(item['acc_s'])
        if 'gyro_s' in item:
            item['gyro_s'] = self.add_spec_noise(item['gyro_s'])

    def add_spec_noise(self, signal):
        noised_signal = np.empty_like(signal)

        for axis in range(signal.shape[1]):
            sig_1d = signal[axis]
            fft = stft(sig_1d, nperseg=sig_1d.shape[0])[2]
            fft_noised = np.empty_like(fft)

            for i in range(fft.shape[1]):
                noise = self.amplitude * np.sin(np.linspace(0, 1, fft.shape[0]))
                fft_noised[:, i] = fft[:, i] + noise

            noised_signal[:, axis] = istft(fft_noised)[1]

        return torch.Tensor(noised_signal)
