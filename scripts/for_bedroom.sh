# W8A8
python ./scripts/sample_diffusion_ldm_bedroom.py -n 50000 --batch_size 50 -r models/ldm/lsun_beds256/model.ckpt -c 200 -e 1.0 --ptq --split --logdir result/bedroom --device cuda:0 --weight_bit 8 --quant_act --act_bit 8 --a_sym --calib_t_mode normal --calib_num_samples 1024 --batch_samples 64 --calib_im_mode greedy --lamda 100.0 --recon --lr_w 5e-4 --lr_a 1e-4 --add_loss 0.1
# W4A8
python ./scripts/sample_diffusion_ldm_bedroom.py -n 50000 --batch_size 50 -r models/ldm/lsun_beds256/model.ckpt -c 200 -e 1.0 --ptq --split --logdir result/bedroom --device cuda:0 --weight_bit 4 --quant_act --act_bit 8 --a_sym --calib_t_mode normal --calib_num_samples 1024 --batch_samples 64 --calib_im_mode greedy --lamda 100.0 --recon --lr_w 1e-2 --lr_a 5e-3 --add_loss 0.001
