# all
python3 main_linear.py \
    --dataset officehome \
    --encoder resnet18 \
    --data_dir $DATA_DIR/officehome \
    --split_strategy domain \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 1.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 2 \
    --dali \
    --name simclr-officehome_all-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity tdemin \
    --wandb \
    --save_checkpoint \
    --tiny omit4

# art
python3 main_linear.py \
    --dataset officehome \
    --encoder resnet18 \
    --data_dir $DATA_DIR/officehome \
    --split_strategy domain \
    --domain art \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 1.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 7 \
    --dali \
    --name simclr-officehome_art-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity tdemin \
    --wandb \
    --save_checkpoint \
    --tiny omit4

# clipart
python3 main_linear.py \
    --dataset officehome \
    --encoder resnet18 \
    --data_dir $DATA_DIR/officehome \
    --split_strategy domain \
    --domain clipart \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 1.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 7 \
    --dali \
    --name simclr-officehome_clipart-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity tdemin \
    --wandb \
    --save_checkpoint \
    --tiny omit4

# product
python3 main_linear.py \
    --dataset officehome \
    --encoder resnet18 \
    --data_dir $DATA_DIR/officehome \
    --split_strategy domain \
    --domain product \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 1.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 7 \
    --dali \
    --name simclr-officehome_product-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity tdemin \
    --wandb \
    --save_checkpoint \
    --tiny omit4

# reak_world
python3 main_linear.py \
    --dataset officehome \
    --encoder resnet18 \
    --data_dir $DATA_DIR/officehome \
    --split_strategy domain \
    --domain real_world \
    --max_epochs 100 \
    --gpus 0 \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 1.0 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 7 \
    --dali \
    --name simclr-officehome_realworld-linear-eval \
    --pretrained_feature_extractor $PRETRAINED_PATH \
    --project ever-learn \
    --entity tdemin \
    --wandb \
    --save_checkpoint \
    --tiny omit4
    