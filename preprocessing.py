import torch, torchaudio
from torch.nn.functional import pad

class Preprocessor:
    """This class takes raw audio and returns MFCC features"""
    def __init__(self, sample_rate=16000, n_mfcc=40, n_mels=64, fixed_frames=None):
        self.sample_rate = sample_rate
        self.fixed_frames = fixed_frames
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=dict(n_fft=400, hop_length=160, n_mels=n_mels,
                           f_min=0.0, f_max=sample_rate/2, center=True,
                           power=2.0, window_fn=torch.hann_window))

    def __call__(self, waveform, samplingrate):
        if samplingrate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform if waveform.dim()==2 else waveform.unsqueeze(0), samplingrate, self.sample_rate)
        if waveform.dim()==1: waveform = waveform.unsqueeze(0)
        if waveform.size(0)>1: waveform = waveform.mean(0, keepdim=True)
        x = self.mfcc(waveform) 

        if self.fixed_frames is not None: # to pad zeros when shorter than fixed_frames
            T = x.shape[-1]
            if T < self.fixed_frames:
                x = pad(x, (0, self.fixed_frames - T))    
            elif T > self.fixed_frames:
                x = x[:, :self.fixed_frames]              
        return x
