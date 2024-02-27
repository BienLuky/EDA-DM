# W8A8
python ./scripts/sample_diffusion_ldm_church.py -n 50000 --batch_size 100 -r models/ldm/lsun_churches256/model.ckpt -c 500 -e 0.0 --ptq --split --logdir result/church --device cuda:0 --weight_bit 8 --quant_act --act_bit 8 --a_sym --calib_t_mode normal --calib_num_samples 1024 --batch_samples 64 --calib_im_mode greedy --lamda 1.0 --recon --lr_w 5e-2 --lr_a 1e-4 --add_loss 1.0
# W4A8
python ./scripts/sample_diffusion_ldm_church.py -n 50000 --batch_size 100 -r models/ldm/lsun_churches256/model.ckpt -c 500 -e 0.0 --ptq --split --logdir result/church --device cuda:0 --weight_bit 4 --quant_act --act_bit 8 --a_sym --calib_t_mode normal --calib_num_samples 1024 --batch_samples 64 --calib_im_mode greedy --lamda 1.0 --recon --lr_w 5e-2 --lr_a 1e-4 --add_loss 1.0

