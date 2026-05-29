"""
Settings Panel — Thanh cài đặt bên phải với các slider và toggle cho pipeline.
"""
import customtkinter as ctk
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)

# ─── Design tokens ────────────────────────────────────────────────────────────
CARD_BG        = "#161628"
CARD_BORDER    = "#252545"
HEADER_BG      = "#1c1c36"
TEXT_PRIMARY   = "#dde6ff"
TEXT_SECONDARY = "#7a94c0"
TEXT_DIM       = "#445577"


class SectionFrame(ctk.CTkFrame):
    """Nhóm cài đặt có tiêu đề collapsible — thiết kế card hiện đại."""

    def __init__(self, master, title: str, accent: str = "#4f8ef7",
                 icon: str = "", **kwargs):
        super().__init__(
            master,
            fg_color=CARD_BG,
            corner_radius=10,
            border_width=1,
            border_color=CARD_BORDER,
            **kwargs,
        )
        self._collapsed = False
        self._accent = accent

        # Header bar
        hdr = ctk.CTkFrame(self, fg_color=HEADER_BG, corner_radius=6)
        hdr.pack(fill="x", padx=2, pady=(2, 0))

        # Accent line bên trái
        accent_bar = ctk.CTkFrame(hdr, fg_color=accent, width=3, corner_radius=2)
        accent_bar.pack(side="left", fill="y", padx=(5, 0), pady=5)

        # Icon + Title
        ctk.CTkLabel(
            hdr, text=f"{icon}  {title}" if icon else title,
            font=("Inter", 11),
            text_color=accent,
            anchor="w",
        ).pack(side="left", padx=6, pady=5, fill="x", expand=True)

        # Toggle indicator
        self._indicator = ctk.CTkLabel(
            hdr, text="▾", font=("Inter", 12),
            text_color=TEXT_SECONDARY, width=20,
        )
        self._indicator.pack(side="right", padx=6)

        # Bind toggle
        for w in (hdr, self._indicator):
            w.bind("<Button-1>", self._toggle)

        # Content frame
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.pack(fill="x", padx=10, pady=(4, 10))

    def _toggle(self, event=None):
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._content.pack_forget()
            self._indicator.configure(text="▸")
        else:
            self._content.pack(fill="x", padx=10, pady=(4, 10))
            self._indicator.configure(text="▾")

    @property
    def content(self) -> ctk.CTkFrame:
        return self._content


def _labeled_slider(
    parent, label: str, from_: float, to: float,
    default: float, steps: int = 100,
    fmt: str = ".1f",
    accent: str = "#4f8ef7",
) -> tuple[ctk.CTkSlider, ctk.CTkLabel]:
    """Slider có label + giá trị hiển thị, thiết kế hiện đại."""
    container = ctk.CTkFrame(parent, fg_color="transparent")
    container.pack(fill="x", pady=(4, 0))

    # Label row
    label_row = ctk.CTkFrame(container, fg_color="transparent")
    label_row.pack(fill="x")
    ctk.CTkLabel(
        label_row, text=label,
        font=("Inter", 11), text_color=TEXT_SECONDARY, anchor="w",
    ).pack(side="left")
    val_label = ctk.CTkLabel(
        label_row,
        text=f"{default:{fmt}}",
        font=("Inter", 11, "bold"),
        text_color=accent,
        width=42,
        anchor="e",
    )
    val_label.pack(side="right")

    # Slider
    slider = ctk.CTkSlider(
        container,
        from_=from_, to=to,
        number_of_steps=steps,
        height=16,
        button_color=accent,
        button_hover_color=accent,
        progress_color=accent,
        fg_color="#252545",
    )
    slider.set(default)
    slider.pack(fill="x", pady=(2, 0))

    def _update(val):
        val_label.configure(text=f"{float(val):{fmt}}")
    slider.configure(command=_update)
    return slider, val_label


class SettingsPanel(ctk.CTkScrollableFrame):
    """
    Panel bên phải chứa tất cả cài đặt xử lý.
    Thiết kế card hiện đại với accent color per-section.
    """

    def __init__(self, master, on_change=None, gpu=None, **kwargs):
        super().__init__(
            master,
            fg_color="#0b0b18",
            corner_radius=0,
            scrollbar_button_color="#1e2040",
            scrollbar_button_hover_color="#2a2a55",
            **kwargs,
        )
        self._on_change = on_change or (lambda: None)
        self._gpu = gpu  # GPUInfo hoặc None
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # Title
        title_frame = ctk.CTkFrame(self, fg_color="#1a1a35", corner_radius=8)
        title_frame.pack(fill="x", padx=8, pady=(10, 6))
        ctk.CTkLabel(
            title_frame,
            text="⚙  Cài đặt xử lý",
            font=("Inter", 13, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x", padx=10, pady=8)

        # ── Denoise ───────────────────────────────────────────────────
        ACCENT_DENOISE = "#3ecf8e"
        self._sec_denoise = SectionFrame(
            self, title="Khử Noise", accent=ACCENT_DENOISE, icon="🔇"
        )
        self._sec_denoise.pack(fill="x", **pad)
        c = self._sec_denoise.content

        self._denoise_on = ctk.CTkSwitch(
            c, text="Bật khử noise",
            font=("Inter", 11), text_color=TEXT_PRIMARY,
            button_color=ACCENT_DENOISE, button_hover_color=ACCENT_DENOISE,
            progress_color=ACCENT_DENOISE,
        )
        # Mặc định TẮt — chỉ bật khi cần, vì nó làm mờ nhẹ và triệt tiêu hiệu ứng sharpen
        self._denoise_on.pack(anchor="w", pady=(2, 4))

        self._denoise_lum, _ = _labeled_slider(
            c, "Luminance", 0, 20, 5.0, fmt=".1f", accent=ACCENT_DENOISE
        )
        self._denoise_color, _ = _labeled_slider(
            c, "Color", 0, 20, 5.0, fmt=".1f", accent=ACCENT_DENOISE
        )

        # ── Upscale ───────────────────────────────────────────────────
        ACCENT_UPSCALE = "#4f8ef7"
        self._sec_upscale = SectionFrame(
            self, title="Upscale (Real-ESRGAN)", accent=ACCENT_UPSCALE, icon="🔍"
        )
        self._sec_upscale.pack(fill="x", **pad)
        c = self._sec_upscale.content

        self._upscale_on = ctk.CTkSwitch(
            c, text="Bật upscale",
            font=("Inter", 11), text_color=TEXT_PRIMARY,
            button_color=ACCENT_UPSCALE, button_hover_color=ACCENT_UPSCALE,
            progress_color=ACCENT_UPSCALE,
        )
        # Tự bật nếu có GPU đủ mạnh (high/mid tier với CUDA)
        gpu_has_power = (
            self._gpu is not None
            and self._gpu.has_cuda
            and self._gpu.gpu_tier in ("high", "mid")
        )
        if gpu_has_power:
            self._upscale_on.select()
        self._upscale_on.pack(anchor="w", pady=(2, 4))

        # Scale factor
        ctk.CTkLabel(
            c, text="Scale factor",
            font=("Inter", 11), text_color=TEXT_SECONDARY, anchor="w",
        ).pack(fill="x", pady=(2, 1))
        self._scale_var = ctk.StringVar(value="2x")
        scale_row = ctk.CTkFrame(c, fg_color="#1e1e3a", corner_radius=6)
        scale_row.pack(fill="x", pady=(0, 4))
        for s in ["2x", "4x"]:
            ctk.CTkRadioButton(
                scale_row, text=s, variable=self._scale_var, value=s,
                font=("Inter", 11), text_color=TEXT_PRIMARY,
                radiobutton_width=16, radiobutton_height=16,
                fg_color=ACCENT_UPSCALE, hover_color=ACCENT_UPSCALE,
            ).pack(side="left", padx=12, pady=6)

        # Model
        ctk.CTkLabel(
            c, text="Model",
            font=("Inter", 11), text_color=TEXT_SECONDARY, anchor="w",
        ).pack(fill="x", pady=(2, 1))
        self._model_var = ctk.StringVar(value="realesrgan-x4plus")
        self._model_menu = ctk.CTkOptionMenu(
            c, variable=self._model_var,
            values=["realesrgan-x4plus", "realesrgan-x2plus", "realesrgan-x4plus-anime"],
            font=("Inter", 11),
            fg_color="#1e1e3a", button_color=ACCENT_UPSCALE,
            button_hover_color="#3a7aed",
            dropdown_fg_color="#1a1a35",
            dropdown_text_color=TEXT_PRIMARY,
        )
        self._model_menu.pack(fill="x", pady=(0, 2))

        # ── Sharpen ───────────────────────────────────────────────────
        ACCENT_SHARPEN = "#f5a623"
        self._sec_sharpen = SectionFrame(
            self, title="Làm Nét", accent=ACCENT_SHARPEN, icon="✨"
        )
        self._sec_sharpen.pack(fill="x", **pad)
        c = self._sec_sharpen.content

        self._sharpen_on = ctk.CTkSwitch(
            c, text="Bật làm nét",
            font=("Inter", 11), text_color=TEXT_PRIMARY,
            button_color=ACCENT_SHARPEN, button_hover_color=ACCENT_SHARPEN,
            progress_color=ACCENT_SHARPEN,
        )
        self._sharpen_on.select()
        self._sharpen_on.pack(anchor="w", pady=(2, 4))

        self._sharpen_amount, _ = _labeled_slider(
            c, "Mức độ", 0, 3, 1.5, fmt=".1f", accent=ACCENT_SHARPEN
        )
        self._sharpen_radius, _ = _labeled_slider(
            c, "Radius (px)", 0.1, 5, 1.0, fmt=".1f", accent=ACCENT_SHARPEN
        )
        self._sharpen_thresh, _ = _labeled_slider(
            c, "Threshold", 0, 15, 3, steps=15, fmt=".0f", accent=ACCENT_SHARPEN
        )

        # ── Face Restore ─────────────────────────────────────────────
        ACCENT_FACE = "#e668a7"
        self._sec_face = SectionFrame(
            self, title="Face Restore (AI)", accent=ACCENT_FACE, icon="👤"
        )
        self._sec_face.pack(fill="x", **pad)
        c = self._sec_face.content

        warn_frame = ctk.CTkFrame(c, fg_color="#2a1a28", corner_radius=6)
        warn_frame.pack(fill="x", pady=(0, 6))
        ctk.CTkLabel(
            warn_frame,
            text="⚠  Tắt để giữ nguyên đường nét khuôn mặt",
            font=("Inter", 10), text_color="#ff8faf",
            wraplength=220, justify="left",
        ).pack(fill="x", padx=8, pady=6)

        self._face_on = ctk.CTkSwitch(
            c, text="Bật Face Restore",
            font=("Inter", 11), text_color=TEXT_PRIMARY,
            button_color=ACCENT_FACE, button_hover_color=ACCENT_FACE,
            progress_color=ACCENT_FACE,
        )
        self._face_on.pack(anchor="w", pady=(2, 4))

        self._face_fidelity, _ = _labeled_slider(
            c, "Fidelity (0=AI, 1=Original)", 0, 1, 0.5,
            fmt=".2f", accent=ACCENT_FACE,
        )

        ctk.CTkLabel(
            c, text="Model",
            font=("Inter", 11), text_color=TEXT_SECONDARY, anchor="w",
        ).pack(fill="x", pady=(4, 1))
        self._face_model_var = ctk.StringVar(value="codeformer")
        ctk.CTkOptionMenu(
            c, variable=self._face_model_var,
            values=["codeformer", "gfpgan"],
            font=("Inter", 11),
            fg_color="#1e1e3a", button_color=ACCENT_FACE,
            button_hover_color="#d45090",
            dropdown_fg_color="#1a1a35",
            dropdown_text_color=TEXT_PRIMARY,
        ).pack(fill="x", pady=(0, 2))

        # ── Export ────────────────────────────────────────────────────
        ACCENT_EXPORT = "#a78bfa"
        self._sec_export = SectionFrame(
            self, title="Export", accent=ACCENT_EXPORT, icon="💾"
        )
        self._sec_export.pack(fill="x", **pad)
        c = self._sec_export.content

        ctk.CTkLabel(
            c, text="Định dạng",
            font=("Inter", 11), text_color=TEXT_SECONDARY, anchor="w",
        ).pack(fill="x")
        self._fmt_var = ctk.StringVar(value="PNG")
        ctk.CTkOptionMenu(
            c, variable=self._fmt_var,
            values=["PNG", "JPEG", "WEBP", "TIFF"],
            font=("Inter", 11),
            fg_color="#1e1e3a", button_color=ACCENT_EXPORT,
            button_hover_color="#8b6ef0",
            dropdown_fg_color="#1a1a35",
            dropdown_text_color=TEXT_PRIMARY,
            command=self._on_fmt_change,
        ).pack(fill="x", pady=(2, 6))

        # Quality slider (ẩn khi PNG)
        self._quality_frame = ctk.CTkFrame(c, fg_color="transparent")
        self._quality_slider, _ = _labeled_slider(
            self._quality_frame, "Quality", 60, 100, 95,
            steps=40, fmt=".0f", accent=ACCENT_EXPORT,
        )
        # mặc định ẩn (PNG)
        # self._quality_frame.pack(fill="x")  # sẽ show khi chọn JPEG/WEBP

        ctk.CTkLabel(
            c, text="Suffix",
            font=("Inter", 11), text_color=TEXT_SECONDARY, anchor="w",
        ).pack(fill="x", pady=(6, 2))
        self._suffix_entry = ctk.CTkEntry(
            c, font=("Inter", 11),
            placeholder_text="_enhanced",
            fg_color="#1e1e3a",
            border_color=ACCENT_EXPORT,
            border_width=1,
            text_color=TEXT_PRIMARY,
        )
        self._suffix_entry.insert(0, "_enhanced")
        self._suffix_entry.pack(fill="x", pady=(0, 2))

        # Spacer ở cuối
        ctk.CTkFrame(self, fg_color="transparent", height=16).pack()

    def _on_fmt_change(self, val):
        if val in ("JPEG", "WEBP"):
            self._quality_frame.pack(fill="x")
        else:
            self._quality_frame.pack_forget()

    def get_pipeline_settings(self):
        """Trả về PipelineSettings từ trạng thái UI."""
        from core.pipeline import PipelineSettings
        scale_str = self._scale_var.get()
        scale = int(scale_str.replace("x", ""))
        return PipelineSettings(
            denoise_enabled=bool(self._denoise_on.get()),
            denoise_strength=float(self._denoise_lum.get()),
            denoise_color_strength=float(self._denoise_color.get()),
            upscale_enabled=bool(self._upscale_on.get()),
            upscale_factor=scale,
            upscale_model=self._model_var.get(),
            sharpen_enabled=bool(self._sharpen_on.get()),
            sharpen_amount=float(self._sharpen_amount.get()),
            sharpen_radius=float(self._sharpen_radius.get()),
            sharpen_threshold=int(self._sharpen_thresh.get()),
            face_restore_enabled=bool(self._face_on.get()),
            face_restore_fidelity=float(self._face_fidelity.get()),
            face_restore_model=self._face_model_var.get(),
            export_format=self._fmt_var.get(),
            export_quality=int(self._quality_slider.get()),
            export_suffix=self._suffix_entry.get() or "_enhanced",
        )

    def apply_settings(self, s):
        """Load PipelineSettings vào UI."""
        self._denoise_on.select() if s.denoise_enabled else self._denoise_on.deselect()
        self._denoise_lum.set(s.denoise_strength)
        self._denoise_color.set(s.denoise_color_strength)
        self._upscale_on.select() if s.upscale_enabled else self._upscale_on.deselect()
        self._scale_var.set(f"{s.upscale_factor}x")
        self._model_var.set(s.upscale_model)
        self._sharpen_on.select() if s.sharpen_enabled else self._sharpen_on.deselect()
        self._sharpen_amount.set(s.sharpen_amount)
        self._sharpen_radius.set(s.sharpen_radius)
        self._sharpen_thresh.set(s.sharpen_threshold)
        self._face_on.select() if s.face_restore_enabled else self._face_on.deselect()
        self._face_fidelity.set(s.face_restore_fidelity)
        self._face_model_var.set(s.face_restore_model)
        self._fmt_var.set(s.export_format)
        self._quality_slider.set(s.export_quality)
        self._suffix_entry.delete(0, "end")
        self._suffix_entry.insert(0, s.export_suffix)
        self._on_fmt_change(s.export_format)
