# for real-time playback(+TensorRT)
import math
from queue import Queue
import cv2
import _thread
from tqdm import tqdm
import subprocess
import argparse
import torch
import numpy as np
import time
from models.pytorch_msssim import ssim_matlab
from torch.nn import functional as F
from models.warplayer import warp as backwarp
from models.model_gmfss_union.GMFSS import Model as GMFSS
from models.rife_426_heavy.IFNet_HDv3 import IFNet
import warnings

warnings.filterwarnings("ignore")

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Interpolation a video with AFI-ForwardDeduplicate')
parser.add_argument('-i', '--input', dest='input', type=str, default='input.mp4', help='absolute path of input video')
parser.add_argument('-o', '--output', dest='output', type=str, default='output.mp4',
                    help='absolute path of output video')
parser.add_argument('-fps', '--dst_fps', dest='dst_fps', type=float, default=60, help='interpolate to ? fps')
parser.add_argument('-s', '--enable_scdet', dest='enable_scdet', action='store_true', default=True,
                    help='enable scene change detection')
parser.add_argument('-st', '--scdet_threshold', dest='scdet_threshold', type=float, default=0.3,
                    help='ssim scene detection threshold')
parser.add_argument('-hw', '--hwaccel', dest='hwaccel', action='store_true', default=True,
                    help='enable hardware acceleration encode(require nvidia graph card)')
parser.add_argument('-scale', '--scale', dest='scale', type=float, default=1.0,
                    help='flow scale, generally use 1.0 with 1080P and 0.5 with 4K resolution')
parser.add_argument('-w', '--window_size', dest='window_size', type=int, default=3,
                    help='merge frame window size')
args = parser.parse_args()

input = args.input  # input video path
output = args.output  # output video path
scale = args.scale  # flow scale
dst_fps = args.dst_fps  # Must be an integer multiple
enable_scdet = args.enable_scdet  # enable scene change detection
scdet_threshold = args.scdet_threshold  # scene change detection threshold
hwaccel = args.hwaccel  # Use hardware acceleration video encoder
window_size = args.window_size

if window_size <= 1:
    raise Exception("window_size must be greater than 1")

def check_scene(x1, x2):
    if not enable_scdet:
        return False
    x1 = F.interpolate(x1, (32, 32), mode='bilinear', align_corners=False)
    x2 = F.interpolate(x2, (32, 32), mode='bilinear', align_corners=False)
    return ssim_matlab(x1, x2) < scdet_threshold


class TMapper:
    def __init__(self, src=-1., dst=0., times=None):
        self.times = dst / src if times is None else times
        self.now_step = -1

    def get_range_timestamps(self, _min: float, _max: float, lclose=True, rclose=False, normalize=True) -> list:
        _min_step = math.ceil(_min * self.times)
        _max_step = math.ceil(_max * self.times)
        _start = _min_step if lclose else _min_step + 1
        _end = _max_step if not rclose else _max_step + 1
        if _start >= _end:
            return []
        if normalize:
            return [((_i / self.times) - _min) / (_max - _min) for _i in range(_start, _end)]
        return [_i / self.times for _i in range(_start, _end)]


video_capture = cv2.VideoCapture(input)
width, height = map(int, map(video_capture.get, [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT]))


def generate_frame_renderer(input_path, output_path):
    encoder = 'libx264'
    preset = 'medium'
    if hwaccel:
        encoder = 'h264_nvenc'
        preset = 'p7'
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-r', f'{dst_fps}',
        '-s', f'{width}x{height}',
        '-i', 'pipe:0', '-i', input_path,
        '-map', '0:v', '-map', '1:a',
        '-c:v', encoder, "-movflags", "+faststart", "-pix_fmt", "yuv420p", "-qp", "16", '-preset', preset,
        '-c:a', 'aac', '-b:a', '320k', f'{output_path}'
    ]

    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


ffmpeg_writer = generate_frame_renderer(input, output)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def convert(param):
    return {
        k.replace("module.", ""): v
        for k, v in param.items()
        if "module." in k
    }


ifnet = IFNet().to(device).eval()
ifnet.load_state_dict(convert(torch.load(r'weights/train_log_rife_426_heavy/flownet.pkl', map_location='cpu')), False)
gmfss = GMFSS()
gmfss.load_model(r'weights\train_log_gmfss_union', -1)
gmfss.device()
gmfss.eval()


def to_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.


def to_numpy(tensor):
    return (tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy() * 255.).astype(np.uint8)


def load_image(img, _scale):
    h, w, _ = img.shape
    while h * _scale % 128 != 0:
        h += 1
    while w * _scale % 128 != 0:
        w += 1
    img = cv2.resize(img, (w, h))
    img = to_tensor(img)
    return img


def put(things):
    write_buffer.put(things)


def get():
    return read_buffer.get()


def build_read_buffer(r_buffer, v):
    ret, __x = v.read()
    while ret is True:
        r_buffer.put(__x)
        ret, __x = v.read()
    r_buffer.put(None)


def clear_write_buffer(w_buffer):
    global ffmpeg_writer
    while True:
        item = w_buffer.get()
        if item is None:
            break
        result = cv2.resize(item, (width, height))
        ffmpeg_writer.stdin.write(np.ascontiguousarray(result[:, :, ::-1]))
    ffmpeg_writer.stdin.close()
    ffmpeg_writer.wait()


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


def calc_flow(x, y, _scale):
    if _scale != 1:
        x = F.interpolate(x, scale_factor=_scale, mode='bilinear', align_corners=False)
        y = F.interpolate(y, scale_factor=_scale, mode='bilinear', align_corners=False)
    flow = gmfss.flownet(x, y)
    if _scale != 1:
        flow = F.interpolate(flow, scale_factor=1 / _scale, mode='bilinear', align_corners=False) * (1 / _scale)
    return flow


@torch.inference_mode()
@torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu")
def merge_interp(_frame_list, _ts, _scale):
    _frame_list = _frame_list.copy()
    if len(_frame_list) < 2:
        return []
    head = _frame_list[0]
    tail = _frame_list[-1]
    for i in range(len(_frame_list)):
        _frame_list[i] = F.interpolate(_frame_list[i], scale_factor=0.5, mode='bilinear', align_corners=False)
    head_f = F.interpolate(head, scale_factor=0.5, mode='bilinear', align_corners=False)
    tail_f = F.interpolate(tail, scale_factor=0.5, mode='bilinear', align_corners=False)
    merged_flow_head_tail = calc_flow(_frame_list[0], _frame_list[1], _scale)
    merged_flow_tail_head = calc_flow(_frame_list[1], _frame_list[0], _scale)
    for i in range(1, len(_frame_list) - 1):
        flow0 = calc_flow(_frame_list[i], _frame_list[i + 1], _scale)
        flow1 = calc_flow(_frame_list[i + 1], _frame_list[i], _scale)
        merged_flow_head_tail, merged_flow_tail_head = flow_merge(merged_flow_head_tail, merged_flow_tail_head, flow0,
                                                                  flow1)

    metric0, metric1 = gmfss.metricnet(head_f, tail_f, merged_flow_head_tail, merged_flow_tail_head)
    feat_ext0 = gmfss.feat_ext(head)
    feat_ext1 = gmfss.feat_ext(tail)

    outputs = []
    for t in _ts:
        if t == 0:
            outputs.append(head)
        elif t == 1:
            outputs.append(tail)
        else:
            rife = ifnet(torch.cat((head_f, tail_f), 1), timestep=t, scale_list=[16, 8, 4, 2, 1])[0]
            out = gmfss.inference(head, tail, (
                merged_flow_head_tail, merged_flow_tail_head, metric0, metric1, feat_ext0, feat_ext1), t, rife)
            outputs.append(out)

    for i in range(len(outputs)):
        outputs[i] = to_numpy(outputs[i][0])

    return outputs


src_fps = video_capture.get(cv2.CAP_PROP_FPS)
assert dst_fps > src_fps, 'dst fps should be greater than src fps'
total_frames_count = video_capture.get(7)
pbar = tqdm(total=total_frames_count)
read_buffer = Queue(maxsize=100)
write_buffer = Queue(maxsize=-1)
_thread.start_new_thread(build_read_buffer, (read_buffer, video_capture))
_thread.start_new_thread(clear_write_buffer, (write_buffer,))

# start inference
i0 = get()

if i0 is None:
    raise Exception("src does not contain any frames")

I0 = load_image(i0, scale)
frame_list = [I0]

t_mapper = TMapper(src_fps, dst_fps)
idx = 0


flag_end = False
while True:
    scene_change = False
    for _ in range(window_size):
        frame = get()
        if frame is None:
            flag_end = True
            break
        frame = load_image(frame, scale)
        frame_list.append(frame)
        if len(frame_list) >= 2:
            if check_scene(frame_list[-2], frame_list[-1]):
                scene_change = True
                break

    idx_end = idx + len(frame_list) - 1

    if idx_end <= idx:
        break

    if scene_change:
        idx_end -= 1

    ts = t_mapper.get_range_timestamps(idx, idx_end, lclose=True, rclose=flag_end, normalize=True)

    outputs = []
    if scene_change:
        outputs = merge_interp(frame_list[:-1], ts, scale)
    else:
        outputs = merge_interp(frame_list, ts, scale)

    if scene_change:
        idx_end += 1

        frame_scene_start = frame_list[-2]
        frame_scene_end = frame_list[-1]

        ts = t_mapper.get_range_timestamps(idx_end - 1, idx_end, lclose=True, rclose=flag_end, normalize=True)
        outputs.extend([to_numpy(frame_scene_start) for _ in ts])

    for out in outputs:
        put(out)

    idx += len(frame_list) - 1
    pbar.update(len(frame_list) - 1)
    frame_list = [frame_list[-1]]

    if flag_end:
        break

# wait for output
while not write_buffer.empty():
    time.sleep(1)

pbar.update(1)
pbar.close()
