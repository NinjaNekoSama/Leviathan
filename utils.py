import yaml
import matplotlib.pyplot as plt
import torchaudio

def plot_spectrogram(waveform, title="Spectrogram", save_path="spectrogram.png"):

    transform = torchaudio.transforms.Spectrogram(n_fft=1024)
    spec = transform(waveform)

    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)

    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.numpy()[0], aspect="auto", origin="lower", cmap="magma")
    plt.colorbar(label="dB")
    plt.title(title)
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency (Hz)")

    # Save before showing
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def load_config(config_path):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    return config


def plot_waveform(waveform, sample_rate, title="Waveform"):
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.show()
