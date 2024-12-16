import cv2
import numpy as np
import torch
from models.warplayer import warp as backwarp
from models.model_gmfss_union.GMFSS import Model as GMFSS
from models.rife_426_heavy.IFNet_HDv3 import IFNet
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }


ifnet = IFNet().to(device).eval()
state_dict = convert(torch.load(r'weights/train_log_rife_426_heavy/flownet.pkl', map_location='cpu', weights_only=True))
ifnet.load_state_dict(state_dict, strict=False)
gmfss = GMFSS()
gmfss.load_model(r'weights\train_log_gmfss_union', -1)
gmfss.device()
gmfss.eval()

HAS_CUDA = True
try:
    import cupy

    if cupy.cuda.get_cuda_path() == None:
        HAS_CUDA = False
except Exception:
    HAS_CUDA = False

if HAS_CUDA:
    from models.softsplat.softsplat import softsplat as forwarp
else:
    print("System does not have CUDA installed, falling back to PyTorch")
    from models.softsplat.softsplat_torch import softsplat as forwarp


# f01, f10, f12, f21 -> f02, f20
def flow_merge(fxy, fyx, fyz, fzy):
    # align fyz to startpoint x via softspalt
    fwd_fxz = forwarp(fyz, fyx, None, 'avg')
    fwd_fzx = forwarp(fyx, fyz, None, 'avg')

    # identify holes
    ones_mask = torch.ones_like(fwd_fxz)
    warped_ones_mask0 = forwarp(ones_mask, fyx, None, 'avg')
    warped_ones_mask1 = forwarp(ones_mask, fyz, None, 'avg')
    holes0 = warped_ones_mask0 < 0.999
    holes1 = warped_ones_mask1 < 0.999

    # align fyz to startpoint x via backwarp
    bwd_fxz = backwarp(fyz, fxy)
    bwd_fzx = backwarp(fyx, fzy)

    warped_fxz, warped_fzx = fwd_fxz, fwd_fzx

    # fill holes
    warped_fxz[holes0] = bwd_fxz[holes0]
    warped_fzx[holes1] = bwd_fzx[holes1]

    # merge flow
    merged_fxz = fxy + warped_fxz
    merged_fzx = fzy + warped_fzx

    return merged_fxz, merged_fzx


img0 = cv2.imread('input/0.png')
img1 = cv2.imread('input/1.png')
img2 = cv2.imread('input/2.png')
img3 = cv2.imread('input/3.png')
img0 = cv2.resize(img0, (1920, 1152))
img1 = cv2.resize(img1, (1920, 1152))
img2 = cv2.resize(img2, (1920, 1152))
img3 = cv2.resize(img3, (1920, 1152))
I0 = torch.from_numpy(img0.transpose(2, 0, 1)).unsqueeze(0).cuda().float() / 255.
I1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).cuda().float() / 255.
I2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).cuda().float() / 255.
I3 = torch.from_numpy(img3.transpose(2, 0, 1)).unsqueeze(0).cuda().float() / 255.
I0f = F.interpolate(I0, scale_factor=0.5, mode='bilinear', align_corners=False)
I1f = F.interpolate(I1, scale_factor=0.5, mode='bilinear', align_corners=False)
I2f = F.interpolate(I2, scale_factor=0.5, mode='bilinear', align_corners=False)
I3f = F.interpolate(I3, scale_factor=0.5, mode='bilinear', align_corners=False)

with torch.no_grad():
    flow01 = gmfss.flownet(I0f, I1f)
    flow10 = gmfss.flownet(I1f, I0f)
    flow12 = gmfss.flownet(I1f, I2f)
    flow21 = gmfss.flownet(I2f, I1f)
    flow23 = gmfss.flownet(I2f, I3f)
    flow32 = gmfss.flownet(I3f, I2f)

    flow02, flow20 = flow_merge(flow01, flow10, flow12, flow21)
    flow03, flow30 = flow_merge(flow02, flow20, flow23, flow32)

    metric0, metric1 = gmfss.metricnet(I0f, I3f, flow03, flow30)
    feat_ext0 = gmfss.feat_ext(I0)
    feat_ext1 = gmfss.feat_ext(I3)

    outputs = [I0]
    for t in (1 / 3, 2 / 3):
        rife = ifnet(torch.cat((I0f, I3f), 1), timestep=t, scale_list=[16, 8, 4, 2, 1])[0]
        out = gmfss.inference(I0, I3, (flow03, flow30, metric0, metric1, feat_ext0, feat_ext1), t, rife)
        outputs.append(out)
    outputs.append(I3)

    for i in range(len(outputs)):
        outputs[i] = (outputs[i][0].cpu().float().numpy().transpose(1, 2, 0) * 255.).astype(np.uint8)
        outputs[i] = cv2.resize(outputs[i], (1920, 1080))

    cv2.imwrite('output/0.png', outputs[0])
    cv2.imwrite('output/1.png', outputs[1])
    cv2.imwrite('output/2.png', outputs[2])
    cv2.imwrite('output/3.png', outputs[3])
