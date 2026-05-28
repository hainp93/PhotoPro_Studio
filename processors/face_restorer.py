"""
Face Restorer — CodeFormer / GFPGAN face restoration.
TẮT mặc định — chỉ bật khi ảnh hỏng mặt nặng.
Fidelity weight điều chỉnh mức độ AI tác động vào khuôn mặt.
"""
import cv2
import numpy as np
import logging
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_DIR = Path(__file__).parent.parent / "weights"

MODEL_URLS = {
    "codeformer": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "filename": "codeformer.pth",
        "dir": "CodeFormer",
    },
    "gfpgan": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
        "filename": "GFPGANv1.3.pth",
        "dir": "GFPGAN",
    },
}


class FaceRestorer:
    """
    Phục hồi khuôn mặt AI.
    - CodeFormer: tốt nhất, có fidelity weight
    - GFPGAN: nhanh hơn, nhẹ hơn
    """

    def __init__(self):
        self._net = None
        self._face_helper = None
        self._loaded_model = ""

    def _load_codeformer(self):
        from basicsr.utils.download_util import load_file_from_url
        from basicsr.utils.registry import ARCH_REGISTRY
        from core.gpu_detector import get_gpu_info

        gpu = get_gpu_info()
        device = gpu.get_torch_device()
        cfg = MODEL_URLS["codeformer"]
        weight_dir = WEIGHTS_DIR / cfg["dir"]
        weight_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = load_file_from_url(
            cfg["url"], model_dir=str(weight_dir), progress=True
        )

        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(checkpoint["params_ema"])
        net.eval()

        self._net = net
        logger.info("CodeFormer loaded.")

    def _get_face_helper(self, upscale: int = 1):
        from facelib.utils.face_restoration_helper import FaceRestoreHelper
        from core.gpu_detector import get_device
        device = get_device()
        return FaceRestoreHelper(
            upscale_factor=upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
        )

    def process(
        self,
        image: np.ndarray,
        fidelity: float = 0.5,
        model_name: str = "codeformer",
        upsample: bool = True,
    ) -> np.ndarray:
        """
        image   : BGR uint8
        fidelity: 0.0 (AI hoàn toàn) → 1.0 (giữ nguyên nhất có thể)
        """
        from core.gpu_detector import get_device
        from torchvision.transforms.functional import normalize
        from basicsr.utils import img2tensor, tensor2img
        from facelib.utils.misc import is_gray

        device = get_device()

        if self._loaded_model != model_name or self._net is None:
            self._load_codeformer()
            self._loaded_model = model_name

        face_helper = self._get_face_helper(upscale=2 if upsample else 1)
        face_helper.clean_all()
        face_helper.read_image(image)

        num_faces = face_helper.get_face_landmarks_5(
            only_center_face=False, resize=640, eye_dist_threshold=5
        )
        logger.debug(f"FaceRestorer: detected {num_faces} faces")

        if num_faces == 0:
            logger.info("No faces detected, skipping face restoration.")
            return image

        face_helper.align_warp_face()

        # Restore mỗi mặt
        for cropped_face in face_helper.cropped_faces:
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = self._net(face_t, w=fidelity, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"CodeFormer inference failed: {e}")
                restored_face = tensor2img(face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face, cropped_face)

        # Paste back
        face_helper.get_inverse_affine(None)
        result = face_helper.paste_faces_to_input_image()
        return result

    def unload(self):
        self._net = None
        self._face_helper = None
        self._loaded_model = ""
        torch.cuda.empty_cache()
        logger.info("FaceRestorer unloaded.")
