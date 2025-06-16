import torch
import random
import numpy as np
import wandb
from torchvision import transforms
from torch.optim import AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from einops import rearrange
from dino_decoder import VQVAE
from test_loader import SplitTrajectoryDataset
from dino_models import VideoTransformer, normalize_acs
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

transform = transforms.Compose([           
                                transforms.Resize(256),                    
                                transforms.CenterCrop(224),               
                                transforms.ToTensor(),                    
                                transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )])


transform1 = transforms.Compose([           
                                transforms.Resize(520),
                                transforms.CenterCrop(518), #should be multiple of model patch_size                 
                                transforms.ToTensor(),                    
                                transforms.Normalize(mean=0.5, std=0.2)
                                ])



DINO_transform = transforms.Compose([           
                            transforms.Resize(224),
                            #transforms.CenterCrop(224), #should be multiple of model patch_size                 
                            
                            transforms.ToTensor(),])
norm_transform = transforms.Normalize(                      
                                mean=[0.485, 0.456, 0.406],                
                                std=[0.229, 0.224, 0.225]              
                                )



def fail_loss(pred, fail_data):
    
    safe_data = torch.where(fail_data == 0.)
    unsafe_data = torch.where(fail_data == 1.)
    unsafe_data_weak = torch.where(fail_data == 2.)
    
    pos = pred[safe_data]
    neg = pred[unsafe_data]
    neg_weak = pred[unsafe_data_weak]
    
    gamma = 0.75
    lx_loss = (1/pos.size(0))*torch.sum(torch.relu(gamma - pos)) if pos.size(0) > 0 else 0. #penalizes safe for being negative
    lx_loss +=  (1/neg.size(0))*torch.sum(torch.relu(gamma + neg)) if neg.size(0) > 0 else 0. # penalizes unsafe for being positive
    lx_loss +=  (1/neg_weak.size(0))*torch.sum(torch.relu(neg_weak)) if neg_weak.size(0) > 0 else 0. # penalizes unsafe for being positive

    return lx_loss

if __name__ == "__main__":
    wandb.init(project="dino-WM",
               name="Classifier")

    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    BS = 16
    BL= 4
    EVAL_H = 16
    H = 3

    hdf5_file = '/data/ken/latent-labeled/consolidated.h5'
    hdf5_file_test = '/data/ken/latent-test-labeled/consolidated.h5'

    expert_data = SplitTrajectoryDataset(hdf5_file, BL, split='train', num_test=0)
    expert_data_eval = SplitTrajectoryDataset(hdf5_file_test, BL, split='test', num_test=5)
    expert_data_imagine = SplitTrajectoryDataset(hdf5_file_test, 32, split='test', num_test=5)

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

    device = 'cuda:0'
   
    decoder = VQVAE().to(device)
    decoder.load_state_dict(torch.load('/home/kensuke/latent-safety/scripts/checkpoints/testing_decoder.pth'))
    decoder.eval()

    transition = VideoTransformer(
        image_size=(224, 224),
        dim=384,  # DINO feature dimension
        ac_dim=10,  # Action embedding dimension
        state_dim=8,  # State dimension
        depth=6,
        heads=16,
        mlp_dim=2048,
        num_frames=BL-1,
        dropout=0.1
    ).to(device)
    transition.load_state_dict(torch.load('/home/kensuke/latent-safety/scripts/checkpoints/best_testing.pth'))

    for name, param in transition.named_parameters():
        param.requires_grad = name.startswith("failure_head")

    data = next(expert_loader)
    

    data1 = data['cam_zed_embd'].to(device)
    data2 =  data['cam_rs_embd'].to(device)
    inputs1 = data1[:, :-1]
    output1 = data1[:, 1:]

    inputs2 = data2[:, :-1]
    output2 = data2[:, 1:]

    data_state = data['state'].to(device)
    states = data_state[:, :-1]
    output_state = data_state[:, 1:]

    data_acs = data['action'].to(device)
    acs = data_acs[:, :-1]
    acs = normalize_acs(acs, device)


    # Forward pass
    optimizer = AdamW([
        {'params': transition.failure_head.parameters(), 'lr': 5e-5}, 
    ])

    best_eval = float('inf')
    best_fail= float('inf')
    iters = []
    train_iter = 10000

    for i in tqdm(range(train_iter), desc="Training", unit="iter"):
        if i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
        if i %len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        if i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))


        data = next(expert_loader)

        data1 = data['cam_zed_embd'].to(device)
        data2 =  data['cam_rs_embd'].to(device)
        inputs1 = data1[:, :-1]
        output1 = data1[:, 1:]

        inputs2 = data2[:, :-1]
        output2 = data2[:, 1:]

        data_state = data['state'].to(device)
        states = data_state[:, :-1]
        output_state = data_state[:, 1:]

        data_acs = data['action'].to(device)
        acs = data_acs[:, :-1]
        acs = normalize_acs(acs, device)
        
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
            failure_loss = fail_loss(pred_fail, data['failure'][:, 1:])
            loss = failure_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss = loss.item()
        wandb.log({'train_loss': train_loss})
        print(f"\rIter {i}, Train Loss: {train_loss:.4f}, failure Loss: {failure_loss.item():.4f}", end='', flush=True)
        
        if (i) % 500 == 0:
            iters.append(i)
            eval_data = next(expert_loader_imagine)
            transition.eval()
            with torch.no_grad():
                eval_data1 = eval_data['cam_zed_embd'].to(device)
                eval_data2 =  eval_data['cam_rs_embd'].to(device)

                inputs1 = eval_data1[[0], :H].to(device)
                inputs2 = eval_data2[[0], :H].to(device)
                all_acs = eval_data['action'][[0]].to(device)
                all_acs = normalize_acs(all_acs, device)
                acs = eval_data['action'][[0],:H].to(device)
                acs = normalize_acs(acs, device)
                states = eval_data['state'][[0],:H].to(device)
                im1s = eval_data['agentview_image'][[0], :H].squeeze().to(device)/255.
                im2s = eval_data['robot0_eye_in_hand_image'][[0], :H].squeeze().to(device)/255.
                for k in range(EVAL_H-H):
                    pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
                    pred_latent = torch.cat([pred1[:,[-1]], pred2[:,[-1]]], dim=0)#.squeeze()
                    pred_ims, _ = decoder(pred_latent)

                    pred_ims = rearrange(pred_ims, "(b t) c h w -> b t c h w", t=1)
                    pred_im1, pred_im2 = torch.split(pred_ims, [inputs1.shape[0], inputs2.shape[0]], dim=0)

                    pred_im1 = pred_im1[0].permute(0,2,3,1).detach()
                    pred_im2 = pred_im2[0].permute(0,2,3,1).detach()
                    pred_fail = pred_fail[:,-1]

                    if pred_fail < 0:
                        pred_im1[:,:,:,0] *= 2
                        pred_im2[:,:,:,0] *= 2
                    
                    im1s = torch.cat([im1s, pred_im1], dim=0)
                    im2s = torch.cat([im2s, pred_im2], dim=0)
                    
                    # getting next inputs
                    acs = torch.cat([acs[[0], 1:], all_acs[0,H+k].unsqueeze(0).unsqueeze(0)], dim=1)
                    inputs1 = torch.cat([inputs1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
                    inputs2 = torch.cat([inputs2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
                    states = torch.cat([states[[0], 1:], pred_state[:,-1].unsqueeze(1)], dim=1)

                    
                gt_im1 = eval_data['agentview_image'][[0], :EVAL_H].squeeze().to(device)
                gt_im2 = eval_data['robot0_eye_in_hand_image'][[0], :EVAL_H].squeeze().to(device)
                gt_fail = eval_data['failure'][[0], :EVAL_H].squeeze().to(device)
                
                for j in range(EVAL_H):
                    if gt_fail[j] > 0:
                        gt_im1[j,:,:,0] *= 2
                        gt_im2[j,:,:,0] *= 2
               

                gt_imgs = torch.cat([gt_im1, gt_im2], dim=-3)/255.
                pred_imgs = torch.cat([im1s, im2s], dim=-3)


                vid = torch.cat([gt_imgs, pred_imgs], dim=-2)
                vid = vid[H:]

                vid = rearrange(vid, "t h w c -> t c h w")
                vid = vid.detach().cpu().numpy()
                vid = (vid * 255).clip(0, 255).astype(np.uint8)

                wandb.log({"video": wandb.Video(vid, fps=20)})
                
                # done logging video

    
                eval_data = next(expert_loader_eval)

                data1 = eval_data['cam_zed_embd'].to(device)
                data2 =  eval_data['cam_rs_embd'].to(device)

                inputs1 = data1[:, :-1]
                output1 = data1[:, 1:]

                inputs2 = data2[:, :-1]
                output2 = data2[:, 1:]

                data_state = eval_data['state'].to(device)
                states = data_state[:, :-1]
                output_state = data_state[:, 1:]

                data_acs = eval_data['action'].to(device)
                acs = data_acs[:, :-1]
                acs = normalize_acs(acs, device)

                pred1, pred2, pred_state, pred_fail = transition(inputs1, inputs2, states, acs)
                
                failure_loss = fail_loss(pred_fail, eval_data['failure'][:, 1:])
                loss = failure_loss
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f},")

            torch.save(transition.state_dict(), f'checkpoints/classifier.pth')

            if loss < best_eval:
                best_eval = loss
                print(f"New best at iter {i}, saving model.")
                torch.save(transition.state_dict(), 'checkpoints/best_classifier.pth')

            
            transition.train()
            wandb.log({'eval_loss': loss.item(), 'failure_loss':failure_loss.item()})


    plt.legend()
    plt.savefig('training curve.png')    
      


