"""
Face Restorer — CodeFormer face restoration.
Pipeline giống hệt Wedding Beauty Studio V9.5:
  1. Detect faces (RetinaFace resnet50)
  2. Crop & align mỗi mặt → 512×512
  3. CodeFormer restore với fidelity weight
  4. Optional: RealESRGAN x2 upscale background
  5. Paste restored faces vào background

Cần model weights (copy từ Wedding app hoặc download):
  weights/CodeFormer/codeformer.pth
  weights/realesrgan/RealESRGAN_x2plus.pth
  weights/facelib/detection_Resnet50_Final.pth
  weights/facelib/parsing_parsenet.pth
"""
import cv2
import numpy as np
import logging
import sys
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = ROOT / "weights"

# Fallback: lấy weights từ Wedding Beauty Studio nếu có trên máy dev
WEDDING_WEIGHTS = Path(
    r"f:\Setup\Wedding Beauty Studio V9.5\Wedding Beauty Studio V9.5\VGA\weights"
)

# Thêm CodeFormer repo vào sys.path (cài bằng git clone, không pip)
# CodeFormer_repo/facelib cũng available qua path này — không cần pip install facelib
_CF_REPO = ROOT / "CodeFormer_repo"
if _CF_REPO.exists():
    for _p in [str(_CF_REPO)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    logger.info(f"sys.path += CodeFormer_repo: {_CF_REPO}")


def _resolve_weight(subdir: str, filename: str) -> str | None:
    """Tìm file weight: PhotoPro weights/ → Wedding app → None."""
    local = WEIGHTS_DIR / subdir / filename
    if local.exists():
        return str(local)
    wedding = WEDDING_WEIGHTS / subdir / filename
    if wedding.exists():
        logger.info(f"Dùng weights từ Wedding app: {wedding}")
        return str(wedding)
    return None


class FaceRestorer:
    """
    Phục hồi khuôn mặt bằng CodeFormer + RealESRGAN background.
    Pipeline giống Wedding Beauty Studio.
    """

    def __init__(self):
        self._net = None
        self._bg_upsampler = None
        self._loaded = False

    def _load(self, use_bg_upscale: bool = True):
        """Load CodeFormer + (optional) RealESRGAN background upsampler."""
        from basicsr.utils.registry import ARCH_REGISTRY

        # Registry name: "CodeFormer" hoặc "CodeFormer_basicsr" tùy phiên bản
        cf_cls = ARCH_REGISTRY.get("CodeFormer") or ARCH_REGISTRY.get("CodeFormer_basicsr")
        if cf_cls is None:
            raise ImportError(
                "CodeFormer không tìm thấy trong ARCH_REGISTRY. "
                "Chạy: git clone https://github.com/sczhou/CodeFormer.git CodeFormer_repo "
                "&& cd CodeFormer_repo && pip install -e . --no-deps"
            )

        # ── Device ────────────────────────────────────────────────────
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_half = (
            torch.cuda.is_available()
            and not any(
                tag in torch.cuda.get_device_name(0)
                for tag in ["1650", "1660"]
            )
        )
        logger.info(f"FaceRestorer: device={device} fp16={use_half}")

        # ── CodeFormer ────────────────────────────────────────────────
        cf_path = _resolve_weight("CodeFormer", "codeformer.pth")
        if not cf_path:
            raise FileNotFoundError(
                "codeformer.pth không tìm thấy. "
                "Chạy: python scripts/setup_models.py"
            )

        net = cf_cls(
            dim_embd=512, codebook_size=1024,
            n_head=8, n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(device)

        checkpoint = torch.load(cf_path, map_location=device)
        net.load_state_dict(checkpoint["params_ema"])
        net.eval()
        if use_half and device.type == "cuda":
            net = net.half()
        self._net = net
        logger.info("✅ CodeFormer loaded")

        # ── RealESRGAN background upsampler ───────────────────────────
        if use_bg_upscale:
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from basicsr.utils.realesrgan_utils import RealESRGANer

                esrgan_path = _resolve_weight("realesrgan", "RealESRGAN_x2plus.pth")
                if not esrgan_path:
                    raise FileNotFoundError("RealESRGAN_x2plus.pth không tìm thấy")

                bg_model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2,
                )
                self._bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path=esrgan_path,
                    model=bg_model,
                    tile=0,           # 0 = no tile, dùng toàn VRAM (OK với RTX 5090 16GB)
                    tile_pad=40,
                    pre_pad=0,
                    half=use_half,
                    device=device,
                )
                logger.info("✅ RealESRGAN x2 background upsampler loaded")
            except Exception as e:
                logger.warning(f"RealESRGAN không load được: {e}. Bỏ qua background upscale.")
                self._bg_upsampler = None

        self._loaded = True

    def _get_face_helper(self, upscale: int, device):
        """Tạo FaceRestoreHelper với detection + parsing models."""
        from facelib.utils.face_restoration_helper import FaceRestoreHelper

        # Override model path để dùng local weights thay vì download
        det_path = _resolve_weight("facelib", "detection_Resnet50_Final.pth")
        parse_path = _resolve_weight("facelib", "parsing_parsenet.pth")

        helper = FaceRestoreHelper(
            upscale_factor=upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=device,
        )
        # Override model paths nếu có local weights
        if det_path:
            helper.face_det.model_path = det_path
        if parse_path and hasattr(helper, 'face_parse') and helper.face_parse is not None:
            pass  # FaceRestoreHelper tự quản lý path parsing

        return helper

    def setup_image(self, image: np.ndarray, upsample: bool = True, bg_upscale: bool = True):
        """Khởi tạo FaceRestoreHelper và load ảnh gốc."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self._loaded:
            self._load(use_bg_upscale=bg_upscale)
            
        self._current_image = image
        self._current_upsample = upsample
        self._current_bg_upscale = bg_upscale
        
        self.face_helper = self._get_face_helper(upscale=2 if upsample else 1, device=device)
        self.face_helper.clean_all()
        self.face_helper.read_image(image)

    def detect_faces(self, high_res: bool = True) -> list:
        """Quét ảnh và trả về danh sách bounding boxes của AI [x1, y1, x2, y2]."""
        resize_dim = None if high_res else 640
        eye_dist = 3 if high_res else 5
        
        self.face_helper.get_face_landmarks_5(
            only_center_face=False, resize=resize_dim, eye_dist_threshold=eye_dist
        )
        logger.info(f"AI phát hiện {len(self.face_helper.det_faces)} khuôn mặt (high_res={high_res})")
        
        bboxes = []
        for face in self.face_helper.det_faces:
            bboxes.append(face[:4].tolist()) # [x1, y1, x2, y2]
        return bboxes

    def add_manual_face(self, bbox: list) -> bool:
        """
        Xử lý khung thủ công do người dùng vẽ:
        Cắt vùng ảnh đó ra, chạy lại RetinaFace cục bộ để tìm 5 điểm neo (landmarks).
        Nếu tìm thấy, cộng dồn tọa độ gốc và lưu vào face_helper.
        bbox: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = map(int, bbox)
        # Nới rộng bbox một chút để RetinaFace dễ bắt
        ih, iw = self._current_image.shape[:2]
        w, h = x2 - x1, y2 - y1
        px, py = int(w * 0.2), int(h * 0.2)
        
        cx1 = max(0, x1 - px)
        cy1 = max(0, y1 - py)
        cx2 = min(iw, x2 + px)
        cy2 = min(ih, y2 + py)
        
        crop = self._current_image[cy1:cy2, cx1:cx2]
        
        # Chạy detect trên ảnh crop (Dùng một helper tạm thời)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        temp_helper = self._get_face_helper(upscale=1, device=device)
        temp_helper.clean_all()
        temp_helper.read_image(crop)
        
        num = temp_helper.get_face_landmarks_5(only_center_face=True, resize=None)
        if num == 0:
            logger.warning("Không tìm thấy 5 điểm neo (landmarks) trong khung thủ công.")
            return False
            
        # Ánh xạ tọa độ từ crop về ảnh gốc
        local_det = temp_helper.det_faces[0]
        local_det[0] += cx1; local_det[1] += cy1
        local_det[2] += cx1; local_det[3] += cy1
        
        local_lmk = temp_helper.all_landmarks_5[0]
        local_lmk[:, 0] += cx1
        local_lmk[:, 1] += cy1
        
        # Thêm vào helper chính
        self.face_helper.det_faces.append(local_det)
        self.face_helper.all_landmarks_5.append(local_lmk)
        logger.info(f"Đã thêm khuôn mặt thủ công tại {x1},{y1}-{x2},{y2}")
        return True

    def restore_faces(self, fidelity: float = 0.5) -> np.ndarray:
        """
        Tiến hành align, restore và paste các khuôn mặt (cả AI và thủ công)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_faces = len(self.face_helper.det_faces)
        
        if num_faces == 0:
            logger.info("Không có mặt → bỏ qua face restore")
            if self._bg_upsampler and self._current_bg_upscale:
                rgb = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2RGB)
                bg, _ = self._bg_upsampler.enhance(rgb, outscale=2)
                return cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
            return self._current_image

        self.face_helper.align_warp_face()

        # ── Restore mỗi khuôn mặt ────────────────────────────────────
        for cropped_face in self.face_helper.cropped_faces:
            from torchvision.transforms.functional import normalize as norm_fn
            from basicsr.utils import img2tensor, tensor2img
            
            face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            norm_fn(face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            face_t = face_t.unsqueeze(0).to(device)

            if next(self._net.parameters()).dtype == torch.float16:
                face_t = face_t.half()

            try:
                with torch.no_grad():
                    output = self._net(face_t, w=fidelity, adain=True)[0]
                    restored = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"CodeFormer inference lỗi: {e}")
                restored = tensor2img(face_t.float(), rgb2bgr=True, min_max=(-1, 1))

            self.face_helper.add_restored_face(restored.astype("uint8"), cropped_face)

        # ── Upscale background + paste faces ─────────────────────────
        self.face_helper.get_inverse_affine(None)

        if self._bg_upsampler and self._current_bg_upscale:
            logger.info("RealESRGAN: upscaling background 2x...")
            rgb = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2RGB)
            bg_rgb, _ = self._bg_upsampler.enhance(rgb, outscale=2)
            bg_img = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)
        else:
            bg_img = None

        # Paste restored faces vào background
        if self._current_upsample and self._bg_upsampler:
            result = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img,
                draw_box=False,
                face_upsampler=self._bg_upsampler,
            )
        else:
            result = self.face_helper.paste_faces_to_input_image(
                upsample_img=bg_img,
                draw_box=False,
            )

        logger.info("✅ Face restore hoàn thành")
        return result


    def unload(self):
        self._net = None
        self._bg_upsampler = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("FaceRestorer unloaded.")
