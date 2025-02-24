# DisCoder: High-Fidelity Music Vocoder Using Neural Audio Codecs

[Paper](https://arxiv.org/abs/2502.12759) | [Samples](https://lucala.github.io/discoder/) | [Code](https://github.com/ETH-DISCO/discoder) | [Model](https://huggingface.co/disco-eth/discoder)  

DisCoder is a neural vocoder that leverages a generative adversarial encoder-decoder architecture informed by
a neural audio codec to reconstruct high-fidelity 44.1 kHz audio from mel spectrograms. Our approach first transforms
the mel spectrogram into a lower-dimensional representation aligned with the Descript Audio Codec (DAC) latent space
before reconstructing it to an audio signal using a fine-tuned DAC decoder.


## Installation
The codebase has been tested with Python 3.11. To get started, clone the repository and set up the environment using Conda:
```shell
git clone https://github.com/ETH-DISCO/discoder

conda create -n discoder python=3.11
conda activate discoder
python -m pip install -r requirements.txt
```

## Inference with 🤗 Hugging Face
Use the following script to perform inference with the pretrained DisCoder model from Hugging Face.
The model uses the Z prediction target and was trained using 128 mel bins.
```python
import torch
from IPython.display import Audio, display
from discoder.models import DisCoder
from discoder import meldataset, utils

device = "cuda"
audio_path = "clip.wav"

# load pretrained DisCoder model
discoder = DisCoder.from_pretrained("disco-eth/discoder").eval().to(device)

# load and pad 44.1 kHz audio file
audio_org, _ = meldataset.load_wav(full_path=audio_path, sr_target=discoder.config["sample_rate"], resample=True, normalize=True)
audio = torch.tensor(audio_org).unsqueeze(dim=0).to(device)
to_pad = discoder.config["segment_size"] - (audio.shape[-1] % discoder.config["segment_size"])
audio = torch.nn.functional.pad(audio, pad=(0, to_pad), mode="constant", value=0)

# create mel spectrogram
mel = utils.get_mel_spectrogram_from_config(audio, discoder.config)  # [B, 128, frames]

# reconstruct audio
with torch.no_grad():
    wav_recon = discoder(mel)  # [B, 1, time]
    wav_recon = wav_recon[..., 0:-to_pad]  # remove padding

display(Audio(audio_org, rate=discoder.config["sample_rate"]))
display(Audio(wav_recon.squeeze().cpu().numpy(), rate=discoder.config["sample_rate"]))
```


## Training
To calculate [ViSQOL](https://github.com/google/visqol) during validation, install the library by following the steps below:
```shell
cd discoder
git clone https://github.com/google/visqol
bazel build :visqol -c opt

cd visqol && pip install .
```

To start training, use the following command:
```shell
python -u train.py --config configs/config_z.json
```


## Inference
The inference script allows batch processing of audio files. It converts all WAV files in the specified `input_dir` to
mel spectrograms, then reconstructs them into audio files in the `output_dir`.
```shell
python -u inference.py --input_dir input_dir --output_dir output_dir --checkpoint_file model.pt --config configs/config_z.json
```
You can also pass the `normalize_volume` flag to standardize the output volume.
