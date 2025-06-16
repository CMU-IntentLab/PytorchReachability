import torch
import wandb
from torch import nn
from torch.optim import AdamW
from torchvision import transforms
from torch.utils.data import DataLoader
from einops import rearrange
import matplotlib.pyplot as plt
from test_loader import SplitTrajectoryDataset
from dino_decoder import VQVAE

dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')


DINO_transform = transforms.Compose([           
                                transforms.Resize(224),                                
                                transforms.ToTensor(),])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    wandb.init(project="dino-WM",
               name="Decoder")


    hdf5_file = '/data/ken/ken_data/skittles_trajectories.h5'
    hdf5_file = '/data/ken/latent/consolidated.h5'
    H = 1
    BS = 64
    expert_data = SplitTrajectoryDataset(hdf5_file, H, split='train', num_test=100)
    expert_data_eval = SplitTrajectoryDataset(hdf5_file, H, split='test', num_test=100)

    expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
    device = 'cuda:0'
    
    decoder = VQVAE().to(device)
    print('decoder with parameters', count_parameters(decoder))
    
    optimizer = AdamW([
        {'params': decoder.parameters(), 'lr': 3e-4}
    ])

    best_eval = float('inf')
    iters = []
    train_losses = []
    eval_losses = []
    train_iter = 5000
    for i in range(train_iter):
        if i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=BS, shuffle=True))
        if i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=BS, shuffle=True))
        data = next(expert_loader)

        inputs1 = data['cam_zed_embd'].to(device)
        inputs2 = data['cam_rs_embd'].to(device)
        output1 = data['agentview_image'].squeeze().to(device)/255.
        output2 = data['robot0_eye_in_hand_image'].squeeze().to(device)/255.


        inputs = torch.cat([inputs1, inputs2], dim=0)#.squeeze()

        pred, _ = decoder(inputs)
        pred = rearrange(pred, "(b t) c h w -> b t c h w", t=1)
        
        pred1, pred2 = torch.split(pred, [inputs1.shape[0], inputs2.shape[0]], dim=0)
        pred1 = pred1.squeeze().permute(0, 2, 3, 1)
        pred2 = pred2.squeeze().permute(0, 2, 3, 1)
        loss = nn.MSELoss()(pred1, output1.squeeze())
        loss += nn.MSELoss()(pred2, output2.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({'train_loss': loss.item()})
        print(f"\rIter {i}, Train Loss: {loss.item():.4f}", end='', flush=True)
        
        if i % 100 == 0:
            train_losses.append(loss.item())
            iters.append(i)
            eval_data = next(expert_loader_eval)
            decoder.eval()
            with torch.no_grad():
                inputs1 = eval_data['cam_zed_embd'].to(device)
                inputs2 = eval_data['cam_rs_embd'].to(device)
                output1 = eval_data['agentview_image'].squeeze().to(device)/255.
                output2 = eval_data['robot0_eye_in_hand_image'].squeeze().to(device)/255.


                inputs = torch.cat([inputs1, inputs2], dim=0)
                pred, _ = decoder(inputs)
                pred = rearrange(pred, "(b t) c h w -> b t c h w", t=1)
                pred1, pred2 = torch.split(pred, [inputs1.shape[0], inputs2.shape[0]], dim=0)
                pred1 = pred1.squeeze().permute(0, 2, 3, 1)
                pred2 = pred2.squeeze().permute(0, 2, 3, 1)
                
                loss = nn.MSELoss()(pred1, output1)
                loss += nn.MSELoss()(pred2, output2)

            print()
            print(f"\rIter {i}, Eval Loss: {loss.item():.4f}")
            if loss < best_eval:
                best_eval = loss
                torch.save(decoder.state_dict(), 'checkpoints/testing_decoder.pth')
            decoder.train()
            
            out_log = (output1[0].detach().detach().cpu().numpy())
            pred_log = (pred1[0].detach().detach().cpu().numpy())
            out_log2 = (output2[0].detach().detach().cpu().numpy())
            pred_log2 = (pred2[0].detach().detach().cpu().numpy())

            wandb.log({'eval_loss': loss.item(), 'ground_truth_front': wandb.Image(out_log), 'pred_front': wandb.Image(pred_log), 'ground_truth_wrist': wandb.Image(out_log2), 'pred_wrist': wandb.Image(pred_log2)})
            eval_losses.append(loss.item())


    plt.plot(iters, train_losses, label='train')
    plt.plot(iters, eval_losses, label='eval')
    plt.legend()
    plt.savefig('training curve.png')    