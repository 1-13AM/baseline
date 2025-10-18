
python -m train.train_from_scratch --config model_configs/VideoMAE_v2.yaml \
                                --num_classes 100 \
                                --num_epochs 50 \
                                --learning_rate 0.00015 \
                                --batch_size 32 \
                                --scheduler StepLR \
                                --model_path pretrained_checkpoints/VideoMAE_v2/vit_b.pth \
                                --train_data_path TRAIN_DATA_PATH \
                                --validation_data_path VALIDATION_DATA_PATH \
                                --warmup_steps 0.01 \
                                --save_ckpt_every 1 \
                                --save_ckpt_dir model_ckpts/VideoMAE-v2-base-sl \
                                --accumulation_steps 1 \
                                --class_balance true \
                                --device cuda:0 \
                                --use_wandb false

                                