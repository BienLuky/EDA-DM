import torch
import torch.nn as nn
from torch.cuda import amp
from tqdm import tqdm
import matplotlib.pyplot as plt
from ddim.functions.denoising import generalized_steps, cali_generalized_steps
from qdiff.utils import AttentionMap
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddim_control import DDIMSampler_control
from ldm.models.diffusion.plms import PLMSSampler

def TDAC_cifar_calib_data_generator(model, config, lamda, calib_num_samples, num_samples, device, diffusion, class_cond=True):
    model_kwargs = {}
    if class_cond:
        cls = torch.tensor([1] * num_samples, device=device).long()  # TODO class gen
        model_kwargs["y"] = cls

    loop_fn = (cali_generalized_steps)

    seq = diffusion.seq
    all_sample=[]
    calib_data = None
    all_mask = []
    hooks = []
    feature_map = []
    shape = (num_samples, 3, config.data.image_size, config.data.image_size)
    img = torch.randn(*shape, device=device) 
    hooks.append(AttentionMap(model.mid.attn_1))
    for now_rt, sample_t in enumerate(
            loop_fn(
                model=model, seq=seq, x=img, b=diffusion.betas, eta=diffusion.args.eta, args=diffusion.args
            )
        ):
        if len(seq)==now_rt:
            all_sample = sample_t[:-1]
            feature_map.append(hooks[0].feature[0])
            break
        elif now_rt==0:
            continue
        else:
            feature_map.append(hooks[0].feature[0])

    torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()

    dense_r = 3.0
    dense_num = torch.zeros(len(feature_map), dtype=torch.int16)
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                mse = torch.mean((feature_map[i]-feature_map[j])**2)
                if mse <= dense_r:
                    dense_num[i] = dense_num[i] + 1

    dense_num_normal = (dense_num - dense_num.min())/(dense_num.max() - dense_num.min())

    calib_mask = []
    CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

    index = torch.argmax(dense_num_normal)
    calib_mask.append(index)

    Cos_dis = torch.zeros(len(feature_map))
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                Cos_dis[i] = Cos_dis[i] + torch.sum(1-CosineSimilarity(feature_map[i], feature_map[j]))

    Cos_dis_normal = (Cos_dis - Cos_dis.min())/(Cos_dis.max() - Cos_dis.min())
    l = lamda
    print("The para l is {}".format(l))
    w = dense_num_normal + l * Cos_dis_normal

    prob = w / torch.sum(w)
    t_num = torch.tensor((prob * calib_num_samples).round(), dtype=int)
    t_error = calib_num_samples - torch.sum(t_num)
    _, t_num_sort = torch.sort(t_num, descending = True)
    if t_error>=0:
        t_num_add = t_num_sort[:t_error]
        t_num[t_num_add] += 1
    else:
        a = range(len(t_num))
        for i in reversed(a):
            if t_error == 0:
                break
            if t_num[i] > 0:
                t_num[i] -= 1
                t_error = t_error + 1
            else:
                continue
    assert torch.sum(t_num)==calib_num_samples

    x = range(len(t_num))
    f = plt.figure()
    plt.plot(x, t_num.to('cpu'))
    plt.savefig('./CIFAR_TDAC_t_num.png')

    assert torch.sum(t_num)==calib_num_samples
    t = []
    for time, num in enumerate(t_num):
        tensor_t = torch.full((num,), time)
        t.append(tensor_t)
    t = torch.hstack(t).to(device)
    t_mask = torch.randperm(t.size(0))
    t = torch.tensor(t[t_mask]).to(device)

    all_calib_data = []
    all_mask=[]
    # '''
    for i in range(int(calib_num_samples/num_samples)):
        calib_data = None
        t1 = t[i*num_samples:(i+1)*num_samples]
        for now_rt, sample_t in enumerate(all_sample):
            if calib_data is None:
                calib_data = torch.zeros_like(sample_t)
            mask = t1 == now_rt
            all_mask.append(mask.float())
            if mask.any():
                calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
        all_calib_data.append(calib_data)
    calib_data = torch.cat(all_calib_data)

    '''
    for i in range(int(calib_num_samples/num_samples)):
        calib_data = None
        t1 = t[i*num_samples:(i+1)*num_samples]
        if i != 0:
            img = torch.randn(*shape, device=device)
            all_sample = generalized_steps(model=model, seq=seq, x=img, b=diffusion.betas, eta=diffusion.args.eta, args=diffusion.args)
            all_sample = all_sample[0][:-1]
        for now_rt, sample_t in enumerate(all_sample):
            if calib_data is None:
                calib_data = torch.zeros_like(sample_t)
            mask = t1 == now_rt
            calib_data = calib_data.to(device)
            sample_t = sample_t.to(device)
            all_mask.append(mask.float())
            if mask.any():
                calib_data += sample_t * mask.float().view(-1, 1, 1, 1)
        all_calib_data.append(calib_data)
    calib_data = torch.cat(all_calib_data)
    # '''

    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(seq[(len(seq)-1)-time])
    t = torch.tensor(calib_t).to(device)
    if class_cond:
        return calib_data, t, cls.to(device)
    else:
        return calib_data, t


def TDAC_bedroom_calib_data_generator(model, args, calib_num_samples, num_samples, device, num_timesteps):
    samples = []
    ts = None
    shape = [num_samples,
            model.model.diffusion_model.in_channels,
            model.model.diffusion_model.image_size,
            model.model.diffusion_model.image_size]
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    hooks = []
    hooks.append(AttentionMap(model.model.diffusion_model.model.middle_block[1]))

    with torch.no_grad():
        for i in tqdm(range(int(calib_num_samples/num_samples)), desc="Generating image samples for cali-data"):
            if i == 0:
                sample, intermediates, feature_map = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False, hooks=hooks,)
            else:
                sample, intermediates = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False,)
            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
        torch.cuda.empty_cache()

    all_samples = []
    for t_sample in range(num_timesteps):
        t_samples = torch.cat([sample[t_sample].to(device) for sample in samples])
        all_samples.append(t_samples)
    samples = None
    torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()

    print("caculate density num")
    dense_r = 0.3
    dense_num = torch.zeros(len(feature_map), dtype=torch.int16)
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                mse = torch.mean((feature_map[i]-feature_map[j])**2)
                # print(mse)
                if mse <= dense_r:
                    dense_num[i] = dense_num[i] + 1

    dense_num_normal = (dense_num - dense_num.min())/(dense_num.max() - dense_num.min())
    CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

    Cos_dis = torch.zeros(len(feature_map))
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                Cos_dis[i] = Cos_dis[i] + torch.sum(1-CosineSimilarity(feature_map[i], feature_map[j]))

    Cos_dis_normal = (Cos_dis - Cos_dis.min())/(Cos_dis.max() - Cos_dis.min())
    l = args.lamda
    print("The para l is {}".format(l))
    w = dense_num_normal + l * Cos_dis_normal

    prob = w / torch.sum(w)
    t_num = torch.tensor((prob * calib_num_samples).round(), dtype=int)
    t_error = calib_num_samples - torch.sum(t_num)
    _, t_num_sort = torch.sort(t_num, descending = True)
    if t_error>=0:
        t_num_add = t_num_sort[:t_error]
        t_num[t_num_add] += 1
    else:
        a = range(len(t_num))
        for i in reversed(a):
            if t_error == 0:
                break
            if t_num[i] > 0:
                t_num[i] -= 1
                t_error = t_error + 1
            else:
                continue
    assert torch.sum(t_num)==calib_num_samples

    x = range(len(t_num))
    f = plt.figure()
    plt.plot(x, t_num.to('cpu'))
    plt.savefig('./Bedroom_TDAC_t_num.png')

    t = []
    for time, num in enumerate(t_num):
        tensor_t = torch.full((num,), time)
        t.append(tensor_t)
    t = torch.hstack(t)
    t_mask = torch.randperm(t.size(0))
    t = torch.tensor(t[t_mask]).to(device)
    
    index = (num_timesteps-1)-t
    calib_data = None
    for now_rt, sample_t in enumerate(all_samples):
        if calib_data is None:
            calib_data = torch.zeros_like(sample_t)
        mask = t == now_rt
        sample_t.to(device)
        if mask.any():
            calib_data += sample_t * mask.float().view(-1, 1, 1, 1)

    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
    t = torch.tensor(calib_t).to(device)

    return calib_data, t, index

def TDAC_church_calib_data_generator(model, args, calib_num_samples, num_samples, device, num_timesteps):
    samples = []
    ts = None
    shape = [num_samples,
            model.model.diffusion_model.in_channels,
            model.model.diffusion_model.image_size,
            model.model.diffusion_model.image_size]

    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    hooks = []
    hooks.append(AttentionMap(model.model.diffusion_model.model.middle_block[1]))

    with torch.no_grad():
        for i in tqdm(range(int(calib_num_samples/num_samples)), desc="Generating image samples for cali-data"):
            if i == 0:
                sample, intermediates, feature_map = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False, hooks=hooks,)
            else:
                sample, intermediates = ddim.sample(args.custom_steps, batch_size=bs, shape=shape, eta=args.eta, verbose=False,)
            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
        torch.cuda.empty_cache()

    all_samples = []
    for t_sample in range(num_timesteps):
        t_samples = torch.cat([sample[t_sample].to(device) for sample in samples])
        all_samples.append(t_samples)
    samples = None
    torch.cuda.empty_cache()
    for hook in hooks:
        hook.remove()

    print("caculate density num")
    dense_r = 0.3
    dense_num = torch.zeros(len(feature_map), dtype=torch.int16)
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                mse = torch.mean((feature_map[i]-feature_map[j])**2)
                if mse <= dense_r:
                    dense_num[i] = dense_num[i] + 1

    dense_num_normal = (dense_num - dense_num.min())/(dense_num.max() - dense_num.min())
    CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

    Cos_dis = torch.zeros(len(feature_map))
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                Cos_dis[i] = Cos_dis[i] + torch.sum(1-CosineSimilarity(feature_map[i], feature_map[j]))

    Cos_dis_normal = (Cos_dis - Cos_dis.min())/(Cos_dis.max() - Cos_dis.min())
    l = args.lamda
    print("The para l is {}".format(l))
    w = dense_num_normal + l * Cos_dis_normal

    prob = w / torch.sum(w)
    t_num = torch.tensor((prob * calib_num_samples).round(), dtype=int)
    t_error = calib_num_samples - torch.sum(t_num)
    _, t_num_sort = torch.sort(t_num, descending = True)
    if t_error>=0:
        t_num_add = t_num_sort[:t_error]
        t_num[t_num_add] += 1
    else:
        a = range(len(t_num))
        for i in reversed(a):
            if t_error == 0:
                break
            if t_num[i] >= 0:
                t_num[i] -= 1
                t_error = t_error + 1
            else:
                continue
    assert torch.sum(t_num)==calib_num_samples

    x = range(len(t_num))
    f = plt.figure()
    plt.plot(x, t_num.to('cpu'))
    plt.savefig('./Church_TDAC_t_num.png')

    t = []
    for time, num in enumerate(t_num):
        tensor_t = torch.full((num,), time)
        t.append(tensor_t)
    t = torch.hstack(t)
    t_mask = torch.randperm(t.size(0))
    t = torch.tensor(t[t_mask]).to(device)
    
    index = (num_timesteps-1)-t
    calib_data = None
    for now_rt, sample_t in enumerate(all_samples):
        if calib_data is None:
            calib_data = torch.zeros_like(sample_t)
        mask = t == now_rt
        sample_t.to(device)
        if mask.any():
            calib_data += sample_t * mask.float().view(-1, 1, 1, 1)

    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
    t = torch.tensor(calib_t).to(device)

    return calib_data, t, index


def TDAC_imagenet_calib_data_generator(model, args, calib_num_samples, num_samples, device, num_timesteps):
    uc = None
    if args.scale != 1.0:                         
        uc = model.get_learned_conditioning(
            {model.cond_stage_key: torch.tensor(calib_num_samples*[1000]).to(model.device)}
            )
    xc = args.data[:calib_num_samples]
    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
    shape = [3, 64, 64]

    sampler = DDIMSampler_control(model)
    samples = []
    cond = []
    uncond = []
    ts = None
    hooks = []
    hooks.append(AttentionMap(model.model.diffusion_model.model.middle_block[1]))

    with torch.no_grad():
        for i in tqdm(range(int(calib_num_samples/num_samples)), desc="Generating image samples for cali-data"):
            if i == 0:
                _, intermediates, feature_map = sampler.sample(S=args.custom_steps,
                                                conditioning=c[i*num_samples:(i+1)*num_samples],
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc[i*num_samples:(i+1)*num_samples],
                                                eta=args.ddim_eta,
                                                hooks=hooks)
            else:
                _, intermediates = sampler.sample(S=args.custom_steps,
                                                conditioning=c[i*num_samples:(i+1)*num_samples],
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc[i*num_samples:(i+1)*num_samples],
                                                eta=args.ddim_eta)
            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
            cond.append(intermediates['cond'][0].to(device))
            uncond.append(intermediates['uncond'][0].to(device))
        torch.cuda.empty_cache()
    cond = torch.cat(cond)
    uncond = torch.cat(uncond)
    all_samples = []
    for t_sample in range(num_timesteps):
        t_samples = torch.cat([sample[t_sample].to(device) for sample in samples])
        all_samples.append(t_samples)
    samples = None
    torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()

    print("caculate density num")
    dense_r = 3.0
    dense_num = torch.zeros(len(feature_map), dtype=torch.int16)
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                mse = torch.mean((feature_map[i]-feature_map[j])**2)
                if mse <= dense_r:
                    dense_num[i] = dense_num[i] + 1

    dense_num_normal = (dense_num - dense_num.min())/(dense_num.max() - dense_num.min())
    CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

    Cos_dis = torch.zeros(len(feature_map))
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                Cos_dis[i] = Cos_dis[i] + torch.sum(1-CosineSimilarity(feature_map[i], feature_map[j]))

    Cos_dis_normal = (Cos_dis - Cos_dis.min())/(Cos_dis.max() - Cos_dis.min())
    l = args.lamda
    print("The para l is {}".format(l))
    w = dense_num_normal + l * Cos_dis_normal

    prob = w / torch.sum(w)
    t_num = torch.tensor((prob * calib_num_samples).round(), dtype=int)
    t_error = calib_num_samples - torch.sum(t_num)
    _, t_num_sort = torch.sort(t_num, descending = True)
    if t_error>=0:
        t_num_add = t_num_sort[:t_error]
        t_num[t_num_add] += 1
    else:
        a = range(len(t_num))
        for i in reversed(a):
            if t_error == 0:
                break
            if t_num[i] > 0:
                t_num[i] -= 1
                t_error = t_error + 1
            else:
                continue
    assert torch.sum(t_num)==calib_num_samples

    x = range(len(t_num))
    f = plt.figure()
    plt.plot(x, t_num.to('cpu'))
    plt.savefig('./ImageNet_TDAC_t_num.png')

    t = []
    for time, num in enumerate(t_num):
        tensor_t = torch.full((num,), time)
        t.append(tensor_t)
    t = torch.hstack(t)
    t_mask = torch.randperm(t.size(0))
    t = torch.tensor(t[t_mask]).to(device)
    
    index = (num_timesteps-1)-t
    calib_data = None
    for now_rt, sample_t in enumerate(all_samples):
        if calib_data is None:
            calib_data = torch.zeros_like(sample_t)
        mask = t == now_rt
        sample_t.to(device)
        if mask.any():
            calib_data += sample_t * mask.float().view(-1, 1, 1, 1)

    calib_data = calib_data.to(device)
    calib_t = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
    t = torch.tensor(calib_t).to(device)

    return calib_data, t, index, cond, uncond


def TDAC_coco_calib_data_generator(model, args, calib_num_samples, num_samples, device, num_timesteps):
    uc = None
    if args.scale != 1.0:
        uc = model.get_learned_conditioning(calib_num_samples * [""])
    prompts = args.list_prompts[:calib_num_samples]
    c = model.get_learned_conditioning(prompts)
    shape = [args.C, args.H // args.f, args.W // args.f]
    start_code = None
    if args.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    samples = []
    cond = []
    uncond = []
    ts = None

    hooks = []
    hooks.append(AttentionMap(model.model.diffusion_model.model.middle_block[1]))

    with torch.no_grad():
        for i in tqdm(range(int(calib_num_samples/num_samples)), desc="Generating image samples for cali-data"):
            if i == 0:
                _, intermediates, feature_map = sampler.sample(S=args.custom_steps,
                                                conditioning=c[i*num_samples:(i+1)*num_samples],
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc[i*num_samples:(i+1)*num_samples],
                                                eta=args.ddim_eta,
                                                x_T=start_code,
                                                hooks=hooks)
            else:
                _, intermediates = sampler.sample(S=args.custom_steps,
                                                conditioning=c[i*num_samples:(i+1)*num_samples],
                                                batch_size=num_samples,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=args.scale,
                                                unconditional_conditioning=uc[i*num_samples:(i+1)*num_samples],
                                                eta=args.ddim_eta,
                                                x_T=start_code)
            samples.append(intermediates['x_inter'][:-1])                                    
            ts = intermediates['ts']
            ts_next = intermediates['ts_next']
            cond.append(intermediates['cond'][0].to(device))
            uncond.append(intermediates['uncond'][0].to(device))
        torch.cuda.empty_cache()
    cond = torch.cat(cond)
    uncond = torch.cat(uncond)
    all_samples = []
    for t_sample in range(num_timesteps):
        t_samples = torch.cat([sample[t_sample].to(device) for sample in samples])
        all_samples.append(t_samples)
    samples = None
    torch.cuda.empty_cache()

    for hook in hooks:
        hook.remove()

    print("caculate density num")
    dense_r = 0.3
    dense_num = torch.zeros(len(feature_map), dtype=torch.int16)
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                mse = torch.mean((feature_map[i]-feature_map[j])**2)
                if mse <= dense_r:
                    dense_num[i] = dense_num[i] + 1

    dense_num_normal = (dense_num - dense_num.min())/(dense_num.max() - dense_num.min())
    CosineSimilarity = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()

    Cos_dis = torch.zeros(len(feature_map))
    for i in range(len(feature_map)):
        for j in range(len(feature_map)):
            if i != j:
                Cos_dis[i] = Cos_dis[i] + torch.sum(1-CosineSimilarity(feature_map[i], feature_map[j]))

    Cos_dis_normal = (Cos_dis - Cos_dis.min())/(Cos_dis.max() - Cos_dis.min())
    l = args.lamda
    print("The para l is {}".format(l))
    w = dense_num_normal + l * Cos_dis_normal

    prob = w / torch.sum(w)
    t_num = torch.tensor((prob * calib_num_samples).round(), dtype=int)
    t_error = calib_num_samples - torch.sum(t_num)
    _, t_num_sort = torch.sort(t_num, descending = True)
    if t_error>=0:
        t_num_add = t_num_sort[:t_error]
        t_num[t_num_add] += 1
    else:
        a = range(len(t_num))
        for i in reversed(a):
            if t_error == 0:
                break
            if t_num[i] > 0:
                t_num[i] -= 1
                t_error = t_error + 1
            else:
                continue
    assert torch.sum(t_num)==calib_num_samples

    x = range(len(t_num))
    f = plt.figure()
    plt.plot(x, t_num.to('cpu'))
    plt.savefig('./COCO_TDAC_t_num.png')

    t = []
    for time, num in enumerate(t_num):
        tensor_t = torch.full((num,), time)
        t.append(tensor_t)
    t = torch.hstack(t)
    t_mask = torch.randperm(t.size(0))
    t = torch.tensor(t[t_mask]).to(device)
    
    index = (num_timesteps-1)-t
    calib_data = None
    for now_rt, sample_t in enumerate(all_samples):
        if calib_data is None:
            calib_data = torch.zeros_like(sample_t)
        mask = t == now_rt
        sample_t.to(device)
        if mask.any():
            calib_data += sample_t * mask.float().view(-1, 1, 1, 1)

    calib_data = calib_data.to(device)
    calib_t = []
    calib_t_next = []
    for time in t:
        calib_t.append(ts[time][0].to(device))
        calib_t_next.append(ts_next[time][0].to(device))
    t = torch.tensor(calib_t).to(device)
    t_next = torch.tensor(calib_t_next).to(device)

    return calib_data, t, index, cond, uncond, t_next
