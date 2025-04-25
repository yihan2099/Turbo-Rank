PYTHONPATH=. torchrun --standalone \
        --nproc_per_node=4 \
        cli/train_nrms_ddp.py