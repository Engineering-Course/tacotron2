#CUDA_VISIBLE_DEVICES=0 python -m multiproc train.py --output_directory=outdir --log_directory=logdir
#python synthesis.py -o resutls/thchs30 -t outdir/thchs30/checkpoint_15990 -w outdir/waveglow_old.pt -l test_list_zh.txt 
CUDA_VISIBLE_DEVICES=0 python -m multiproc train.py --output_directory=outdir --log_directory=logdir
