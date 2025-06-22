# W8A8
# CUDA_VISIBLE_DEVICES=0 python ./scripts/sample_txt2img.py \
#                                 --prompt "a puppy wearing a hat" --from-file /coco/annotations/captions_val2014.json \
#                                 --ckpt models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --logdir result/coco --device cuda:0 \
#                                 --plms --cond --no_grad_ckpt --skip_grid \
#                                 --ptq --split --weight_bit 8 --quant_act --act_bit 8 --sm_abit 8 \
#                                 --n_samples 10000 --n_batch 4 \
#                                 --calib_num_samples 256 --batch_samples 8 --lamda 5.0 \
#                                 --recon --lr_w 5e-4 --lr_a 1e-4 --add_loss 0.8
# W4A8
CUDA_VISIBLE_DEVICES=0 python ./scripts/sample_txt2img.py \
                                --prompt "a puppy wearing a hat" --from-file /coco/annotations/captions_val2014.json \
                                --ckpt models/ldm/stable-diffusion-v1/sd-v1-4.ckpt --logdir result/coco --device cuda:0 \
                                --plms --cond --no_grad_ckpt --skip_grid \
                                --ptq --split --weight_bit 4 --quant_act --act_bit 8 --sm_abit 8 \
                                --n_samples 10000 --n_batch 4 \
                                --calib_num_samples 256 --batch_samples 8 --lamda 5.0 \
                                --recon --lr_w 3e-2 --lr_a 1e-4 --add_loss 0.8