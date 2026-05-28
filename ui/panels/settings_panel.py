"""
Settings Panel — Thanh cài đặt bên phải với các slider và toggle cho pipeline.
"""
import customtkinter as ctk
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)


class SectionFrame(ctk.CTkFrame):
    """Nhóm cài đặt có tiêu đề collapsible."""
    def __init__(self, master, title: str, accent: str = "#2196F3", **kwargs):
        super().__init__(master, fg_color="#1e1e3a", corner_radius=10, **kwargs)
        self._collapsed = False
        self._content = None

        # Header
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=8, pady=(8, 4))

        self._indicator = ctk.CTkLabel(hdr, text="▼", font=("Inter", 10), text_color=accent, width=16)
        self._indicator.pack(side="left")
        ctk.CTkLabel(hdr, text=title, font=("Inter Bold", 12, "bold"), text_color=accent).pack(side="left", padx=4)
        hdr.bind("<Button-1>", self._toggle)
        self._indicator.bind("<Button-1>", self._toggle)

        # Content frame
        self._content = ctk.CTkFrame(self, fg_color="transparent")
        self._content.pack(fill="x", padx=8, pady=(0, 8))

    def _toggle(self, event=None):
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._content.pack_forget()
            self._indicator.configure(text="▶")
        else:
            self._content.pack(fill="x", padx=8, pady=(0, 8))
            self._indicator.configure(text="▼")

    @property
    def content(self) -> ctk.CTkFrame:
        return self._content


def _labeled_slider(parent, label: str, from_: float, to: float,
                    default: float, steps: int = 100,
                    fmt: str = ".1f") -> tuple[ctk.CTkSlider, ctk.CTkLabel]:
    row = ctk.CTkFrame(parent, fg_color="transparent")
    row.pack(fill="x", pady=2)
    ctk.CTkLabel(row, text=label, font=("Inter", 11), anchor="w", width=120).pack(side="left")
    val_label = ctk.CTkLabel(row, text=f"{default:{fmt}}", font=("Inter", 11), width=36, text_color="#8899cc")
    val_label.pack(side="right")
    slider = ctk.CTkSlider(row, from_=from_, to=to, number_of_steps=steps)
    slider.set(default)
    slider.pack(side="left", fill="x", expand=True, padx=(4, 4))

    def _update(val):
        val_label.configure(text=f"{float(val):{fmt}}")
    slider.configure(command=_update)
    return slider, val_label


class SettingsPanel(ctk.CTkScrollableFrame):
    """
    Panel bên phải chứa tất cả cài đặt xử lý.
    """

    def __init__(self, master, on_change=None, **kwargs):
        super().__init__(master, fg_color="#13132a", corner_radius=0, **kwargs)
        self._on_change = on_change or (lambda: None)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        title = ctk.CTkLabel(self, text="⚙  Cài đặt xử lý",
                              font=("Inter Bold", 14, "bold"),
                              text_color="#e0e8ff")
        title.pack(fill="x", padx=12, pady=(12, 8))

        # ── Denoise ───────────────────────────────────────────────────
        self._sec_denoise = SectionFrame(self, "🔇  Khử Noise", accent="#4CAF50")
        self._sec_denoise.pack(fill="x", **pad)
        c = self._sec_denoise.content

        self._denoise_on = ctk.CTkSwitch(c, text="Bật khử noise", font=("Inter", 12))
        self._denoise_on.select()
        self._denoise_on.pack(anchor="w", pady=2)

        self._denoise_lum, _ = _labeled_slider(c, "Luminance", 0, 20, 5.0, fmt=".1f")
        self._denoise_color, _ = _labeled_slider(c, "Color", 0, 20, 5.0, fmt=".1f")

        # ── Upscale ───────────────────────────────────────────────────
        self._sec_upscale = SectionFrame(self, "🔍  Upscale (Real-ESRGAN)", accent="#2196F3")
        self._sec_upscale.pack(fill="x", **pad)
        c = self._sec_upscale.content

        self._upscale_on = ctk.CTkSwitch(c, text="Bật upscale", font=("Inter", 12))
        self._upscale_on.select()
        self._upscale_on.pack(anchor="w", pady=2)

        ctk.CTkLabel(c, text="Scale factor", font=("Inter", 11), anchor="w").pack(fill="x", pady=(4, 0))
        self._scale_var = ctk.StringVar(value="2x")
        scale_row = ctk.CTkFrame(c, fg_color="transparent")
        scale_row.pack(fill="x", pady=2)
        for s in ["2x", "4x"]:
            ctk.CTkRadioButton(scale_row, text=s, variable=self._scale_var, value=s,
                               font=("Inter", 11)).pack(side="left", padx=8)

        ctk.CTkLabel(c, text="Model", font=("Inter", 11), anchor="w").pack(fill="x", pady=(4, 0))
        self._model_var = ctk.StringVar(value="realesrgan-x4plus")
        self._model_menu = ctk.CTkOptionMenu(
            c, variable=self._model_var,
            values=["realesrgan-x4plus", "realesrgan-x2plus", "realesrgan-x4plus-anime"],
            font=("Inter", 11),
        )
        self._model_menu.pack(fill="x", pady=2)

        # ── Sharpen ───────────────────────────────────────────────────
        self._sec_sharpen = SectionFrame(self, "✨  Làm Nét", accent="#FF9800")
        self._sec_sharpen.pack(fill="x", **pad)
        c = self._sec_sharpen.content

        self._sharpen_on = ctk.CTkSwitch(c, text="Bật làm nét", font=("Inter", 12))
        self._sharpen_on.select()
        self._sharpen_on.pack(anchor="w", pady=2)

        self._sharpen_amount, _ = _labeled_slider(c, "Mức độ", 0, 3, 1.0, fmt=".1f")
        self._sharpen_radius, _ = _labeled_slider(c, "Radius (px)", 0.1, 5, 1.0, fmt=".1f")
        self._sharpen_thresh, _ = _labeled_slider(c, "Threshold", 0, 15, 3, steps=15, fmt=".0f")

        # ── Face Restore ─────────────────────────────────────────────
        self._sec_face = SectionFrame(self, "👤  Face Restore (AI)", accent="#E91E63")
        self._sec_face.pack(fill="x", **pad)
        c = self._sec_face.content

        warn = ctk.CTkLabel(c, text="⚠ Tắt để giữ nguyên đường nét khuôn mặt",
                            font=("Inter", 10), text_color="#ff6b6b",
                            wraplength=200, justify="left")
        warn.pack(fill="x", pady=(0, 4))

        self._face_on = ctk.CTkSwitch(c, text="Bật Face Restore", font=("Inter", 12))
        # mặc định TẮT
        self._face_on.pack(anchor="w", pady=2)

        self._face_fidelity, _ = _labeled_slider(c, "Fidelity (0=AI, 1=Original)", 0, 1, 0.5, fmt=".2f")

        ctk.CTkLabel(c, text="Model", font=("Inter", 11), anchor="w").pack(fill="x", pady=(4, 0))
        self._face_model_var = ctk.StringVar(value="codeformer")
        ctk.CTkOptionMenu(c, variable=self._face_model_var,
                          values=["codeformer", "gfpgan"],
                          font=("Inter", 11)).pack(fill="x", pady=2)

        # ── Export ────────────────────────────────────────────────────
        self._sec_export = SectionFrame(self, "💾  Export", accent="#9C27B0")
        self._sec_export.pack(fill="x", **pad)
        c = self._sec_export.content

        ctk.CTkLabel(c, text="Định dạng", font=("Inter", 11), anchor="w").pack(fill="x")
        self._fmt_var = ctk.StringVar(value="PNG")
        ctk.CTkOptionMenu(c, variable=self._fmt_var,
                          values=["PNG", "JPEG", "WEBP", "TIFF"],
                          font=("Inter", 11),
                          command=self._on_fmt_change).pack(fill="x", pady=2)

        self._quality_row = ctk.CTkFrame(c, fg_color="transparent")
        self._quality_row.pack(fill="x")
        ctk.CTkLabel(self._quality_row, text="Quality", font=("Inter", 11)).pack(side="left")
        self._quality_slider, _ = _labeled_slider(c, "Quality", 60, 100, 95, steps=40, fmt=".0f")
        self._quality_row.pack_forget()  # ẩn khi PNG

        ctk.CTkLabel(c, text="Suffix", font=("Inter", 11), anchor="w").pack(fill="x", pady=(4, 0))
        self._suffix_entry = ctk.CTkEntry(c, font=("Inter", 11), placeholder_text="_enhanced")
        self._suffix_entry.insert(0, "_enhanced")
        self._suffix_entry.pack(fill="x", pady=2)

    def _on_fmt_change(self, val):
        if val in ("JPEG", "WEBP"):
            self._quality_row.pack(fill="x")
        else:
            self._quality_row.pack_forget()

    def get_pipeline_settings(self):
        """Trả về dict settings để tạo PipelineSettings."""
        from core.pipeline import PipelineSettings
        scale_str = self._scale_var.get()
        scale = int(scale_str.replace("x", ""))
        s = PipelineSettings(
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
        return s

    def apply_settings(self, s):
        """Load PipelineSettings vào UI."""
        if s.denoise_enabled:
            self._denoise_on.select()
        else:
            self._denoise_on.deselect()
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
