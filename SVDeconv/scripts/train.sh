python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py with ours_re10k_svd-1 distdataparallel=True resume=True -p
# python val.py with $1 -p
# python  train.py with $1 -p