# import tensorflow as tf
import hparam_tf.hparam
from text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = hparam_tf.hparam.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=10,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=True,

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=True,
        training_files='data/ljspeech/ljs_train.txt',
        validation_files='data/ljspeech/ljs_val.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=3,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=256,
        postnet_kernel_size=3,
        postnet_n_convolutions=5,

        # converter parameters
        converter_channels=256,
        n_speakers=1,
        speaker_embed_dim=16,
        dropout=0.05,
        downsample_step=1,
        converter_kernel_size=3,

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=8,
        mask_padding=True, # set model's padded outputs to padded values

        ################################
        # Vocoder Parameters           #
        ################################
        use_lws=True, # Only used to set as True if using WaveNet, no difference in performance is observed in either cases.
        signal_normalization=True, # Whether to normalize mel spectrograms to some predefined range (following below parameters)
        allow_clipping_in_normalization = False,  # Only relevant if mel_normalization = True
        symmetric_mels=True, # Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
        max_abs_value=4., # max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, # not too small for fast convergence)
        min_level_db=-100,
        ref_level_db=20,
        # Griffin Lim
        power=1.5,  # Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
        griffin_lim_iters=60,  # Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
        preemphasize=True,  # whether to apply filter
        preemphasis=0.97,  # filter coefficient.
    )

    # if hparams_string:
    #     tf.logging.info('Parsing command line hparams: %s', hparams_string)
    #     hparams.parse(hparams_string)
    #
    # if verbose:
    #     tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
