# train an X3D model from scratch
python -m train.train_from_scratch --config model_configs/X3D_m.yaml \
                                    --num_classes 83 \
                                    --num_epochs 100 \
                                    --learning_rate 0.0001 \
                                    --batch_size 24 \
                                    --model_path pretrained_checkpoints/x3d/x3d_m.pth \
                                    --train_data_path TRAIN_DATA_PATH \
                                    --validation_data_path VALIDATION_DATA_PATH \
                                    --warmup_steps 0. \
                                    --save_ckpt_every 1 \
                                    --accumulation_steps 1 \
                                    --device cuda:0 \
                                    --use_wandb false                                                                                    