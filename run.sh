CUDA_VISIBLE_DEVICES=0 python -m multiproc train.py --output_directory=outdir --log_directory=logdir
#python synthesis.py -o resutls/ljspeech -t outdir/ljspeech/checkpoint_97170 -w outdir/waveglow_old.pt -l test_training_list_en.txt 
#python synthesis.py -o resutls/ljspeech -t outdir/ljspeech/checkpoint_97170 -w /home/yy/TTS/waveglow/checkpoints/waveglow_242276 -l test_training_list_en.txt 
