import os
import sys
sys.path.append('waveglow/')
import numpy as np
import torch
import argparse
import timeit

from hparams import create_hparams
from model import Tacotron2
from layers import TacotronSTFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from utils import save_wav, plot_alignment, inv_mel_spectrogram



def synthesis_mel(model, text):
    # Prepare text input
    sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    return mel_outputs_postnet, alignments


def synthesis_audio_wavenet(waveglow, mel):
    with torch.no_grad():
        output = waveglow.infer(mel, sigma=0.666)
    audio = output[0].data.cpu().numpy()
    return audio

def tts(model, waveglow, text, hparams):
    start_time = timeit.default_timer()
    mel, alignments = synthesis_mel(model, text)
    print ('text to mel: {}'.format(timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    # audio = synthesis_audio_wavenet(waveglow, mel)
    audio = inv_mel_spectrogram(mel.data.cpu().numpy()[0], hparams)
    print ('mel to audio: {}'.format(timeit.default_timer() - start_time))

    return audio, alignments.data.cpu().numpy()[0].T
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_directory', type=str, help='directory to save results')
    parser.add_argument('-t', '--tacotron_checkpoint_path', type=str, default=None, required=True, help='tacotron checkpoint path')
    parser.add_argument('-w', '--waveglow_checkpoint_path', type=str, default=None, required=True, help='waveglow checkpoint path')
    parser.add_argument('--n_gpus', type=int, default=1, required=False, help='number of gpus')
    parser.add_argument('-l', '--text_list', type=str, default=None, required=True, help='text list for synthesis')
    parser.add_argument('--hparams', type=str, required=False, help='comma separated name=value pairs')
    args = parser.parse_args()

    # Setup hparams
    hparams = create_hparams(args.hparams)

    # Load model from checkpoint
    checkpoint_path = args.tacotron_checkpoint_path
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.eval()

    # Load WaveGlow for mel2audio synthesis
    waveglow_path = args.waveglow_checkpoint_path
    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda()

    with open(args.text_list, "rb") as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            start_time = timeit.default_timer()
            text = line.decode("utf-8")[:-1]
            audio, alignments = tts(model, waveglow, text, hparams)
            dst_wav_path = os.path.join(args.output_directory, "{}.wav".format(idx))
            end_time = timeit.default_timer()
            save_wav(audio, dst_wav_path, hparams)
            print ('synthesized {} audio, using {} seconds'.format(idx, end_time-start_time))

            dst_alignment_path = os.path.join(args.output_directory, "{}_alignment.png".format(idx))
            plot_alignment(alignments, dst_alignment_path)
