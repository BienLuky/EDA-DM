
# W8A8
# CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained \
#                                         --timesteps 100 --eta 0 --skip_type quad \
#                                         --ptq --weight_bit 8 --quant_mode qdiff --split --quant_act --act_bit 8 \
#                                         --logdir result/cifar --device cuda:0 \
#                                         --calib_num_samples 1024 --batch_samples 1024 --lamda 1.2 \
#                                         --max_images 50000 \
#                                         --recon --block_recon --lr_w 5e-2 --lr_a 1e-3 --add_loss 0.8
# W4A8
CUDA_VISIBLE_DEVICES=0 python scripts/sample_diffusion_ddim.py --config configs/cifar10.yml --use_pretrained \
                                        --timesteps 100 --eta 0 --skip_type quad \
                                        --ptq --weight_bit 4 --quant_mode qdiff --split --quant_act --act_bit 8 \
                                        --logdir result/cifar --device cuda:0 \
                                        --calib_num_samples 1024 --batch_samples 1024 --lamda 1.2 \
                                        --max_images 50000 \
                                        --recon --block_recon --lr_w 5e-1 --lr_a 5e-4 --add_loss 0.8

