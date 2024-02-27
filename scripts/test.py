import sys
sys.path.append('./pytorch-fid/src')
sys.path.append('./clip-score/src')
import torch_fidelity
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from clip_score.clip_score import calculate_clip_given_paths

def test_fid(input1, input2='cifar10-train', device = 'cuda:0'):
    torch.cuda.set_device(device)
    print(torch.cuda.current_device())
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=input1, 
        input2=input2, 
        cuda=True, 
        isc=True, 
        fid=True, 
        kid=False, 
        prc=False, 
        verbose=False,
    )
    print(metrics_dict)

def test_bedroom_fid(input1, input2, device = 'cuda:0'):
    torch.cuda.set_device(device)
    print(torch.cuda.current_device())
    device = "cuda"
    kwargs = dict(paths=[input1, input2], 
                    batch_size=100,
                    device=device, 
                    dims=2048,
                    num_workers=8
                    )
    fid_value = calculate_fid_given_paths(**kwargs)
    print('FID: ', fid_value)

def test_church_fid(input1, input2, device = 'cuda:0'):
    torch.cuda.set_device(device)
    print(torch.cuda.current_device())
    device = "cuda"
    kwargs = dict(paths=[input1, input2], 
                    batch_size=100,
                    device=device, 
                    dims=2048,
                    num_workers=8
                    )
    fid_value = calculate_fid_given_paths(**kwargs)
    print('FID: ', fid_value)

def test_stable_diffusion(input1, input2, txt, device = 'cuda:0'):
    torch.cuda.set_device(device)
    print(torch.cuda.current_device())
    device = "cuda"
    kwargs = dict(paths=[input1, input2], 
                    batch_size=100,
                    device=device, 
                    dims=2048,
                    num_workers=8
                    )
    fid_value = calculate_fid_given_paths(**kwargs)
    print('FID: ', fid_value)
    kwargs = dict(real_path=txt, 
                    fake_path=input1,
                    batch_size=100,
                    device=device, 
                    clip_model="ViT-L/14",
                    num_workers=8
                    )
    clip_score = calculate_clip_given_paths(**kwargs)
    print('CLIP Score: ', clip_score)

def test_conditional_fid(input1, input2, device = 'cuda:0'):
    torch.cuda.set_device(device)
    print(torch.cuda.current_device())
    device = "cuda"
    kwargs = dict(paths=[input1, input2], 
                    batch_size=100,
                    device=device, 
                    dims=2048,
                    num_workers=8
                    )
    fid_value = calculate_fid_given_paths(**kwargs)
    print('FID: ', fid_value)

if __name__ == '__main__':
    # test_fid(input1="~/Dome/q-diffusion/image_conditional", input2="/dataset/imagenet/train_new")
    # test_bedroom_fid()
    # test_church_fid()
    test_stable_diffusion()
    # test_conditional_fid()
