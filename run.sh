python synthesis.py -o resutls/bznsyp -t outdir/bznsyp/checkpoint_53125 -w outdir/waveglow_old.pt -l test_list_zh.txt 
#CUDA_VISIBLE_DEVICES=0 python -m multiproc train.py --output_directory=outdir --log_directory=logdir
