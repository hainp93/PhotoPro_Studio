import torch
from pathlib import Path
import sys

sys.path.insert(0, r"e:\Tool\PhotoPro_Studio\CodeFormer_repo")

from basicsr.utils.registry import ARCH_REGISTRY

cf_cls = ARCH_REGISTRY.get("CodeFormer") or ARCH_REGISTRY.get("CodeFormer_basicsr")
net = cf_cls(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]).cuda()
net.eval()

face_t = torch.randn(1, 3, 512, 512).cuda()

print("Testing float32...")
with torch.no_grad():
    out = net(face_t, w=0.5, adain=True)
print("Float32 success!")

print("Testing float16...")
net = net.half()
face_t = face_t.half()
try:
    with torch.no_grad():
        out = net(face_t, w=0.5, adain=True)
    print("Float16 success!")
except Exception as e:
    print(f"Float16 failed: {e}")
