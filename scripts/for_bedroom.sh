# W8A8
# CUDA_VISIBLE_DEVICES=1 python ./scripts/sample_diffusion_ldm_bedroom.py --n_samples 50000 --batch_size 50 \
#                                                 --resume models/ldm/lsun_beds256/model.ckpt \
#                                                 --custom_steps 200 --eta 1.0 \
#                                                 --logdir result/bedroom --device cuda:0 \
#                                                 --ptq --split --weight_bit 8 --quant_act --act_bit 8 \
#                                                 --calib_num_samples 1024 --batch_samples 64 --lamda 1.0 \
#                                                 --recon --lr_w 5e-4 --lr_a 1e-4 --add_loss 1.0
# W4A8
CUDA_VISIBLE_DEVICES=1 python ./scripts/sample_diffusion_ldm_bedroom.py --n_samples 50000 --batch_size 50 \
                                                --resume models/ldm/lsun_beds256/model.ckpt \
                                                --custom_steps 200 --eta 1.0 \
                                                --logdir result/bedroom --device cuda:0 \
                                                --ptq --split --weight_bit 4 --quant_act --act_bit 8 \
                                                --calib_num_samples 1024 --batch_samples 64 --lamda 1.0 \
                                                --recon --lr_w 1e-2 --lr_a 5e-3 --add_loss 1.0
