# W8A8
# CUDA_VISIBLE_DEVICES=3 python ./scripts/sample_diffusion_ldm_imagenet.py --cond --no_grad_ckpt --skip_grid \
#                                                         --ckpt models/ldm/cin256/model.ckpt --config configs/latent-diffusion/cin256-v2.yaml \
#                                                         --ptq --split --weight_bit 8 --quant_act --act_bit 8 --sm_abit 8 \
#                                                         --custom_steps 20 --ddim_eta 0.0 --logdir result/imagenet --device cuda:0 \
#                                                         --n_samples 50000 --n_batch 50 \
#                                                         --calib_num_samples 1024 --batch_samples 64 --lamda 1.2 \
#                                                         --recon --lr_w 1e-4 --lr_a 1e-3 --add_loss 0.8
# W4A8
CUDA_VISIBLE_DEVICES=3 python ./scripts/sample_diffusion_ldm_imagenet.py --cond --no_grad_ckpt --skip_grid \
                                                        --ckpt models/ldm/cin256/model.ckpt --config configs/latent-diffusion/cin256-v2.yaml \
                                                        --ptq --split --weight_bit 4 --quant_act --act_bit 8 --sm_abit 8 \
                                                        --custom_steps 20 --ddim_eta 0.0 --logdir result/imagenet --device cuda:0 \
                                                        --n_samples 50000 --n_batch 50 \
                                                        --calib_num_samples 1024 --batch_samples 64 --lamda 1.2 \
                                                        --recon --lr_w 5e-1 --lr_a 1e-4 --add_loss 0.8