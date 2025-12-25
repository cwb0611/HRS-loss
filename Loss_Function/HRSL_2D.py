
import torch
from torch import nn
import torch.nn.functional as F

def unfold_split_patch_2d(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    N, C, H, W = image.size()
    patchs = F.unfold(image, kernel_size=(patch_size, patch_size), stride=int(patch_size // 2))
    patchs = patchs.reshape(N, C, patch_size, patch_size, -1)
    patchs = patchs.permute(0, 1, 4, 2, 3)
    return patchs
    

class HardRegionSensitiveLoss_2D(nn.Module):

    def __init__(self, patch_size: int = None):
        super().__init__()
        self.patch_size = patch_size
        self.selected_patch_size_list = [2, 4, 8, 16, 32]

    def forward(self, preds, labels):
        if self.patch_size is None:
            patch_size = self.Automated_Patch_Splitting_Strategy(preds, labels)
        else:
            patch_size = self.patch_size
        labels_onehot = self.one_hot_encode(labels, preds.shape)
        return self.Calculate_Loss_Value(preds, labels_onehot, patch_size)

    def one_hot_encode(self, encoded_tensor, encode_shape):
        tensor_onehot = torch.zeros(encode_shape, device=encoded_tensor.device)
        return tensor_onehot.scatter_(1, encoded_tensor.unsqueeze(1).long(), 1)

    def Automated_Patch_Splitting_Strategy(self, preds, labels):
        score_map = torch.zeros_like(labels)
        for class_index in range(preds.shape[1]):
            score_map = torch.where(labels==class_index,preds[:,class_index,:,:],score_map)

        def erode(image):
            bin_image= F.pad(image, pad=[1,1,1,1], mode= "reflect")
            out = F.max_pool2d(bin_image,3,1,0)
            return out

        bound_map = torch.zeros_like(labels)
        labels_onehot = self.one_hot_encode(labels, preds.shape)
        for class_index in range(labels_onehot.shape[1])[1:]:
            target_label = labels_onehot[:,class_index,:,:].unsqueeze(1)
            target_label_dilate = erode(target_label)
            target_label_erode = 1 - erode(1 - target_label)
            target_label_edges = target_label_dilate - target_label_erode
            bound_map = bound_map + target_label_edges.squeeze(1)

        def get_hard_region(bound_map, ps):
            hr1 = F.max_pool2d(bound_map,ps,ps,0)
            hr1 = F.interpolate(hr1,scale_factor=[ps,ps])
            
            hr2 = F.pad(bound_map,[0,int(ps/2),0,0])[:,:,:,int(ps/2):]
            hr2 = F.max_pool2d(hr2,ps,ps,0)
            hr2 = F.interpolate(hr2,scale_factor=[ps,ps])
            hr2 = F.pad(hr2,[int(ps/2),0,0,0])[:,:,:,:-int(ps/2)]
                                
            hr3 = F.pad(bound_map,[0,0,0,int(ps/2)])[:,:,int(ps/2):,:]
            hr3 = F.max_pool2d(hr3,ps,ps,0)
            hr3 = F.interpolate(hr3,scale_factor=[ps,ps])
            hr3 = F.pad(hr3,[0,0,int(ps/2),0])[:,:,:-int(ps/2),:]
                                
            hr4 = F.pad(bound_map,[0,int(ps/2),0,int(ps/2)])[:,:,int(ps/2):,int(ps/2):]
            hr4 = F.max_pool2d(hr4,ps,ps,0)
            hr4 = F.interpolate(hr4,scale_factor=[ps,ps])
            hr4 = F.pad(hr4,[int(ps/2),0,int(ps/2),0])[:,:,:-int(ps/2),:-int(ps/2)]

            hr = hr1 + hr2 + hr3 + hr4
            hr = (hr > 0).float()
            return hr

        best_patch_size = None
        best_hard_avg_score = 100000        
        for patch_size in self.selected_patch_size_list:
            exit_bound_region_map = get_hard_region(bound_map.unsqueeze(1), patch_size)
            hard_avg_score = (exit_bound_region_map * score_map).sum()/exit_bound_region_map.sum()
            if hard_avg_score < best_hard_avg_score:
                best_hard_avg_score = hard_avg_score
                best_patch_size = patch_size
        print(best_patch_size)
        return best_patch_size


    def Calculate_Loss_Value(self, preds, labels, patch_size):
        pred_patches = unfold_split_patch_2d(preds, patch_size)  # (N, C, P, p, p, p)
        label_patches = unfold_split_patch_2d(labels, patch_size)
        
        N, C, P = pred_patches.shape[:3]
        pred_patches = pred_patches.view(N, C, P, -1)  # (N, C, P, p^3)
        label_patches = label_patches.view(N, C, P, -1)

        masked_preds = torch.where(label_patches != 0, pred_patches, torch.inf)
        patch_minp = torch.min(masked_preds, dim=3)[0]
        patch_minp = torch.nan_to_num(patch_minp, nan=0.0, posinf=0.0)

        pos_counts = label_patches.sum(dim=3)  # (N, C, P)
        pos_loss = (label_patches * torch.log(pred_patches + 1e-8)).sum(3)  # (N, C, P)
        
        weights = (1 - patch_minp)  # (N, C, P)
        weighted_loss = weights * pos_loss / (pos_counts + 1e-8)
        return -weighted_loss.mean()

