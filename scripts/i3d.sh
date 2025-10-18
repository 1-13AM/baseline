# train an I3D model from scratch
python -m train.train_from_scratch --config model_configs/Inception3D.yaml \
                                    --num_epochs 30 \
                                    --learning_rate 0.0001 \
                                    --batch_size 24 \
                                    --model_path pretrained_checkpoints/i3d/i3d.pth \
                                    --train_data_path TRAIN_DATA_PATH \
                                    --validation_data_path VALIDATION_DATA_PATH \
                                    --warmup_steps 0. \
                                    --accumulation_steps 1 \
                                    --save_ckpt_every 1 \
                                    --device cuda:0 \
                                    --class_balance true \
                                    --use_wandb false



