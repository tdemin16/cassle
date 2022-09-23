python3 main_continual.py \
    --dataset officehome \
    --encoder resnet18 \
    --data_dir $DATA_DIR/officehome \
    --split_strategy domain \
    --max_epochs 200 \
    --num_tasks 4 \
    --task_idx 0 \
    --gpus 0 \
    --accelerator ddp \
    --sync_batchnorm \
    --num_workers 5 \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.4 \
    --weight_decay 1e-4 \
    --batch_size 512 \
    --brightness 0.8 \
    --contrast 0.8 \
    --saturation 0.8 \
    --hue 0.2 \
    --dali \
    --name simclr-officehome \
    --wandb \
    --save_checkpoint \
    --entity tdemin \
    --project ever-learn \
    --method simclr \
    --temperature 0.2 \
    --proj_hidden_dim 2048 \
    --check_val_every_n_epoch 9999 \
    --disable_knn_eval \
    --tiny
