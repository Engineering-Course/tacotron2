CUDA_VISIBLE_DEVICES=0 python -m multiproc train.py --output_directory=outdir --log_directory=logdir
#python synthesis.py -o resutls -t outdir/checkpoint_11480 -w outdir/waveglow_old.pt -l test_list_en.txt 
