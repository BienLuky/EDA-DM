# W8A8
python ./scripts/sample_diffusion_ldm_imagenet.py --cond --ptq --no_grad_ckpt --split --ddim_steps 20 --ddim_eta 0.0 --ckpt models/ldm/cin256/model.ckpt --config configs/latent-diffusion/cin256-v2.yaml --logdir result/imagenet --device cuda:0 --skip_grid --n_samples 50000 --n_batch 50 --weight_bit 8 --quant_act --act_bit 8 --sm_abit 8 --calib_t_mode normal --calib_num_samples 1024 --batch_samples 64 --calib_im_mode greedy --lamda 0.5 --recon --lr_w 1e-4 --lr_a 1e-3 --add_loss 1.3
# W4A8
python ./scripts/sample_diffusion_ldm_imagenet.py --cond --ptq --no_grad_ckpt --split --ddim_steps 20 --ddim_eta 0.0 --ckpt models/ldm/cin256/model.ckpt --config configs/latent-diffusion/cin256-v2.yaml --logdir result/imagenet --device cuda:0 --skip_grid --n_samples 50000 --n_batch 50 --weight_bit 4 --quant_act --act_bit 8 --sm_abit 8 --calib_t_mode normal --calib_num_samples 1024 --batch_samples 64 --calib_im_mode greedy --lamda 0.5 --recon --lr_w 5e-1 --lr_a 1e-4 --add_loss 1.3