"""
Settings Panel — Thanh cài đặt bên phải với các slider và toggle cho pipeline.
"""
import customtkinter as ctk
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)

# ─── Design tokens ────────────────────────────────────────────────────────────
CARD_BG        = "#181818"
CARD_BORDER    = "#2a2a2a"
HEADER_BG      = "#1e1e1e"
TEXT_PRIMARY   = "#ffffff"
TEXT_SECONDARY = "#a0a0a0"
TEXT_DIM       = "#666666"
ACCENT         = "#2979ff"



class SectionFrame(ctk.CTkFrame):
    """Nhóm cài đặt có tiêu đề collapsible — thiết kế card hiện đại."""

    def __init__(self, master, title: str, accent: str = ACCENT,
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
        hdr = ctk.CTkFrame(self, fg_color=HEADER_BG, corner_radius=6, height=30)
        hdr.pack(fill="x", padx=2, pady=(2, 0))
        hdr.pack_propagate(False)

        # Accent line bên trái
        accent_bar = ctk.CTkFrame(hdr, fg_color=accent, width=3, corner_radius=2)
        accent_bar.pack(side="left", fill="y", padx=(5, 0), pady=4)

        # Icon + Title
        ctk.CTkLabel(
            hdr, text=f"{icon}  {title}" if icon else title,
            font=("Inter", 11),
            text_color=accent,
            anchor="w",
        ).pack(side="left", padx=6, fill="x", expand=True)

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
        self._content.pack(fill="x", padx=8, pady=(2, 6))

    def _toggle(self, event=None):
        self._collapsed = not self._collapsed
        if self._collapsed:
            self._content.pack_forget()
            self._indicator.configure(text="▸")
        else:
            self._content.pack(fill="x", padx=8, pady=(2, 6))
            self._indicator.configure(text="▾")

    @property
    def content(self) -> ctk.CTkFrame:
        return self._content


def _labeled_slider(
    parent, label: str, from_: float, to: float,
    default: float, steps: int = 100,
    fmt: str = ".1f",
    accent: str = ACCENT,
) -> tuple[ctk.CTkSlider, ctk.CTkLabel]:
    """Slider có label + giá trị hiển thị, thiết kế hiện đại."""
    container = ctk.CTkFrame(parent, fg_color="transparent")
    container.pack(fill="x", pady=(2, 0))

    # Label row
    label_row = ctk.CTkFrame(container, fg_color="transparent")
    label_row.pack(fill="x")
    ctk.CTkLabel(
        label_row, text=label,
        font=("Inter", 12), text_color=TEXT_SECONDARY, anchor="w",
    ).pack(side="left")
    val_label = ctk.CTkLabel(
        label_row,
        text=f"{default:{fmt}}",
        font=("Inter", 12, "bold"),
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

    # ── Preset values (mức mặc định tự nhiên, không quá lộ) ──────────────
    PRESET_SLIM      = 40   # 40/100 → nén 4% chiều ngang
    PRESET_STRETCH   = 45   # 45/100 → kéo 5.4% chiều cao
    PRESET_SKIN_TONE = 45   # da trắng hồng 45%
    PRESET_SKIN_SMOOTH = 55 # mịn da 55%
    PRESET_BRIGHT    = True  # bật tăng sáng auto (CLAHE)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 3}

        # ── ⚡ Quick Presets ─────────────────────────────────────────────
        PRESET_ACCENT = "#f5a623"
        self._sec_presets = SectionFrame(
            self, title="Tuỳ Chỉnh Nhanh", accent=PRESET_ACCENT, icon="⚡"
        )
        self._sec_presets.pack(fill="x", **pad)
        cp = self._sec_presets.content

        ctk.CTkLabel(
            cp, text="Tick để áp ngay — mức tự nhiên, không quá lộ",
            font=("Inter", 10), text_color=TEXT_DIM, anchor="w",
        ).pack(fill="x", pady=(0, 6))

        def _make_preset_cb(text, icon, on_toggle):
            row = ctk.CTkFrame(cp, fg_color="#1a1a2e", corner_radius=8)
            row.pack(fill="x", pady=2)
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                row, text=f"{icon}  {text}", variable=var,
                font=("Inter", 12), text_color=TEXT_PRIMARY,
                checkbox_width=18, checkbox_height=18,
                fg_color=PRESET_ACCENT, hover_color=PRESET_ACCENT,
                border_color="#444",
                command=lambda: on_toggle(var.get()),
            )
            cb.pack(anchor="w", padx=10, pady=6)
            return var

        def _on_slim(checked):
            self._beauty_on.select()
            self._body_slim.set(self.PRESET_SLIM if checked else 0)

        def _on_stretch(checked):
            self._beauty_on.select()
            self._leg_stretch.set(self.PRESET_STRETCH if checked else 0)

        def _on_skin(checked):
            self._beauty_on.select()
            self._skin_tone.set(self.PRESET_SKIN_TONE if checked else 0)
            self._skin_smooth.set(self.PRESET_SKIN_SMOOTH if checked else 0)

        def _on_bright(checked):
            if checked:
                self._auto_bright_on.select()
            else:
                self._auto_bright_on.deselect()

        self._preset_slim_var    = _make_preset_cb("Thon gọn",       "👗", _on_slim)
        self._preset_stretch_var = _make_preset_cb("Dài chân",       "📏", _on_stretch)
        self._preset_skin_var    = _make_preset_cb("Da trắng hồng",  "✨", _on_skin)
        self._preset_bright_var  = _make_preset_cb("Tăng sáng auto", "☀️", _on_bright)

        # ── ☀️ Auto Brighten (CLAHE) ──────────────────────────────────
        BRIGHT_ACCENT = "#ffd600"
        self._sec_bright = SectionFrame(
            self, title="Tăng Sáng", accent=BRIGHT_ACCENT, icon="☀️"
        )
        self._sec_bright.pack(fill="x", **pad)
        cb_bright = self._sec_bright.content

        self._auto_bright_on = ctk.CTkSwitch(
            cb_bright, text="Tăng sáng tự động (CLAHE)",
            font=("Inter", 12), text_color=TEXT_PRIMARY,
            button_color=BRIGHT_ACCENT, button_hover_color=BRIGHT_ACCENT,
            progress_color=BRIGHT_ACCENT,
        )
        self._auto_bright_on.pack(anchor="w", pady=(2, 4))
        self._bright_strength, _ = _labeled_slider(
            cb_bright, "Cường độ", 0.5, 4.0, 2.0, steps=35, fmt=".1f",
            accent=BRIGHT_ACCENT
        )

        # ── Beauty & Body ─────────────────────────────────────────────
        self._sec_beauty = SectionFrame(
            self, title="Làm Đẹp (Body & Skin)", accent="#ff4081", icon="✨"
        )
        self._sec_beauty.pack(fill="x", **pad)
        c = self._sec_beauty.content
        
        self._beauty_on = ctk.CTkSwitch(
            c, text="Bật làm đẹp & nắn dáng",
            font=("Inter", 12, "bold"), text_color="#ff4081",
            button_color="#ff4081", button_hover_color="#ff4081",
            progress_color="#ff4081",
        )
        self._beauty_on.pack(anchor="w", pady=(2, 6))

        self._skin_smooth, _ = _labeled_slider(c, "Mịn da", 0, 100, 0, fmt=".0f", accent="#ff4081")
        self._skin_tone, _ = _labeled_slider(c, "Trắng hồng", 0, 100, 0, fmt=".0f", accent="#ff4081")
        self._body_slim, _ = _labeled_slider(c, "Thon gọn", 0, 100, 0, fmt=".0f", accent="#ff4081")
        self._leg_stretch, _ = _labeled_slider(c, "Kéo dài chân", 0, 100, 0, fmt=".0f", accent="#ff4081")


        # ── Denoise ───────────────────────────────────────────────────
        self._sec_denoise = SectionFrame(
            self, title="Khử Noise", accent=ACCENT, icon="🔇"
        )
        self._sec_denoise.pack(fill="x", **pad)
        c = self._sec_denoise.content

        self._denoise_on = ctk.CTkSwitch(
            c, text="Bật khử noise",
            font=("Inter", 12), text_color=TEXT_PRIMARY,
            button_color=ACCENT, button_hover_color=ACCENT,
            progress_color=ACCENT,
        )
        self._denoise_on.pack(anchor="w", pady=(2, 2))

        self._denoise_lum, _ = _labeled_slider(
            c, "Luminance", 0, 20, 5.0, fmt=".1f", accent=ACCENT
        )
        self._denoise_color, _ = _labeled_slider(
            c, "Color", 0, 20, 5.0, fmt=".1f", accent=ACCENT
        )

        # ── Upscale ───────────────────────────────────────────────────
        self._sec_upscale = SectionFrame(
            self, title="Upscale (Real-ESRGAN)", accent=ACCENT, icon="🔍"
        )
        self._sec_upscale.pack(fill="x", **pad)
        c = self._sec_upscale.content

        self._upscale_on = ctk.CTkSwitch(
            c, text="Bật upscale",
            font=("Inter", 11), text_color=TEXT_PRIMARY,
            button_color=ACCENT, button_hover_color=ACCENT,
            progress_color=ACCENT,
        )
        # Tắt tự động bật upscale (theo ý kiến người dùng)
        # gpu_has_power = (...)
        # if gpu_has_power:
        #     self._upscale_on.select()
        self._upscale_on.pack(anchor="w", pady=(2, 4))

        # Scale factor / Target size
        ctk.CTkLabel(
            c, text="Scale factor",
            font=("Inter", 11), text_color=TEXT_SECONDARY, anchor="w",
        ).pack(fill="x", pady=(2, 1))
        self._scale_var = ctk.StringVar(value="2x")
        scale_row = ctk.CTkFrame(c, fg_color="#1e1e3a", corner_radius=6)
        scale_row.pack(fill="x", pady=(0, 2))
        for s in ["2x", "4x"]:
            ctk.CTkRadioButton(
                scale_row, text=s, variable=self._scale_var, value=s,
                font=("Inter", 11), text_color=TEXT_PRIMARY,
                radiobutton_width=16, radiobutton_height=16,
                fg_color=ACCENT, hover_color=ACCENT,
            ).pack(side="left", padx=12, pady=6)

        # Target long-side (custom pixel output)
        target_row = ctk.CTkFrame(c, fg_color="transparent")
        target_row.pack(fill="x", pady=(0, 4))
        self._upscale_target_on = ctk.CTkSwitch(
            target_row, text="Giới hạn cạnh dài (px)",
            font=("Inter", 11), text_color=TEXT_SECONDARY,
            button_color=ACCENT, button_hover_color=ACCENT, progress_color=ACCENT,
            onvalue=True, offvalue=False,
        )
        self._upscale_target_on.pack(side="left", padx=(0, 8))
        self._upscale_target_px = ctk.CTkEntry(
            target_row, width=70, height=26,
            font=("Inter", 11), placeholder_text="6000",
            fg_color="#1e1e3a", border_color="#353570", text_color=TEXT_PRIMARY,
        )
        self._upscale_target_px.insert(0, "6000")
        self._upscale_target_px.pack(side="left")
        ctk.CTkLabel(
            target_row, text="px",
            font=("Inter", 11), text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(4, 0))

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
            fg_color="#1e1e3a", button_color=ACCENT,
            button_hover_color="#3a7aed",
            dropdown_fg_color="#1a1a35",
            dropdown_text_color=TEXT_PRIMARY,
        )
        self._model_menu.pack(fill="x", pady=(0, 2))

        # ── Sharpen ───────────────────────────────────────────────────
        self._sec_sharpen = SectionFrame(
            self, title="Làm Nét", accent=ACCENT, icon="✨"
        )
        self._sec_sharpen.pack(fill="x", **pad)
        c = self._sec_sharpen.content

        self._sharpen_on = ctk.CTkSwitch(
            c, text="Bật làm nét",
            font=("Inter", 12), text_color=TEXT_PRIMARY,
            button_color=ACCENT, button_hover_color=ACCENT,
            progress_color=ACCENT,
        )
        self._sharpen_on.select()
        self._sharpen_on.pack(anchor="w", pady=(2, 4))
        
        self._sharpen_person_only = ctk.CTkSwitch(
            c, text="🧍 Chỉ làm nét cơ thể (AI Person Mask)",
            font=("Inter", 11), text_color="#2979ff",
            button_color=ACCENT, button_hover_color=ACCENT,
            progress_color=ACCENT,
        )
        self._sharpen_person_only.select()
        self._sharpen_person_only.pack(anchor="w", pady=(0, 6))

        # AI mode toggle
        ai_row = ctk.CTkFrame(c, fg_color="#1e1a10", corner_radius=6)
        ai_row.pack(fill="x", pady=(0, 6))
        self._sharpen_ai_on = ctk.CTkSwitch(
            ai_row, text="⚡ AI Làm Nét (Real-ESRGAN → resize gốc)",
            font=("Inter", 11, "bold"), text_color="#f5a623",
            button_color=ACCENT, button_hover_color=ACCENT,
            progress_color=ACCENT,
            command=self._on_sharpen_mode_change,
        )
        self._sharpen_ai_on.pack(anchor="w", padx=8, pady=6)

        # Classical controls
        self._sharpen_classical = ctk.CTkFrame(c, fg_color="transparent")
        self._sharpen_classical.pack(fill="x")

        # Method selector
        method_row = ctk.CTkFrame(self._sharpen_classical, fg_color="transparent")
        method_row.pack(fill="x", pady=(2, 4))
        ctk.CTkLabel(
            method_row, text="Phương pháp",
            font=("Inter", 12), text_color=TEXT_SECONDARY, anchor="w",
        ).pack(side="left")
        self._sharpen_method_var = ctk.StringVar(value="Bilateral")
        ctk.CTkSegmentedButton(
            method_row,
            values=["Bilateral", "USM"],
            variable=self._sharpen_method_var,
            font=("Inter", 11),
            fg_color="#1e1e3a",
            selected_color=ACCENT,
            selected_hover_color=ACCENT,
            unselected_color="#1e1e3a",
            unselected_hover_color="#252540",
            text_color=TEXT_PRIMARY,
            width=140,
        ).pack(side="right")

        self._sharpen_amount, _ = _labeled_slider(
            self._sharpen_classical, "Mức độ", 0, 3, 1.5, fmt=".1f", accent=ACCENT
        )
        self._sharpen_radius, _ = _labeled_slider(
            self._sharpen_classical, "Radius (px)", 0.1, 5, 1.0, fmt=".1f", accent=ACCENT
        )
        self._sharpen_thresh, _ = _labeled_slider(
            self._sharpen_classical, "Threshold", 0, 15, 3, steps=15, fmt=".0f", accent=ACCENT
        )

        # AI controls (ẩn mặc định)
        self._sharpen_ai_frame = ctk.CTkFrame(c, fg_color="transparent")
        self._sharpen_ai_strength, _ = _labeled_slider(
            self._sharpen_ai_frame, "Blend Strength", 0.0, 1.0, 0.85,
            steps=100, fmt=".2f", accent=ACCENT,
        )
        ctk.CTkLabel(
            self._sharpen_ai_frame, text="Model AI",
            font=("Inter", 12), text_color=TEXT_SECONDARY, anchor="w",
        ).pack(fill="x", pady=(4, 1))
        self._sharpen_ai_model_var = ctk.StringVar(value="realesrgan-x4plus")
        ctk.CTkOptionMenu(
            self._sharpen_ai_frame, variable=self._sharpen_ai_model_var,
            values=["realesrgan-x4plus", "realesrgan-x4plus-anime", "realesrgan-x2plus"],
            font=("Inter", 11),
            fg_color="#1e1e3a", button_color=ACCENT,
            button_hover_color="#e09010",
            dropdown_fg_color="#1a1a35",
            dropdown_text_color=TEXT_PRIMARY,
        ).pack(fill="x")

        # ── Face Restore ─────────────────────────────────────────────
        self._sec_face = SectionFrame(
            self, title="Face Restore (AI)", accent=ACCENT, icon="👤"
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
            button_color=ACCENT, button_hover_color=ACCENT,
            progress_color=ACCENT,
        )
        self._face_on.select()
        self._face_on.pack(anchor="w", pady=(2, 4))

        self._face_high_res = ctk.CTkSwitch(
            c, text="Quét toàn bộ ảnh (tìm mặt ở xa)",
            font=("Inter", 11), text_color="#d45090",
            button_color=ACCENT, button_hover_color=ACCENT,
            progress_color=ACCENT,
        )
        self._face_high_res.pack(anchor="w", pady=(0, 6))

        self._face_fidelity, _ = _labeled_slider(
            c, "Fidelity (0=AI, 1=Original)", 0, 1, 0.8,
            fmt=".2f", accent=ACCENT,
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
            fg_color="#1e1e3a", button_color=ACCENT,
            button_hover_color="#d45090",
            dropdown_fg_color="#1a1a35",
            dropdown_text_color=TEXT_PRIMARY,
        ).pack(fill="x", pady=(0, 2))

        # ── Export ────────────────────────────────────────────────────
        self._sec_export = SectionFrame(
            self, title="Export", accent=ACCENT, icon="💾"
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
            fg_color="#1e1e3a", button_color=ACCENT,
            button_hover_color="#8b6ef0",
            dropdown_fg_color="#1a1a35",
            dropdown_text_color=TEXT_PRIMARY,
            command=self._on_fmt_change,
        ).pack(fill="x", pady=(2, 6))

        # Quality slider (ẩn khi PNG)
        self._quality_frame = ctk.CTkFrame(c, fg_color="transparent")
        self._quality_slider, _ = _labeled_slider(
            self._quality_frame, "Quality", 60, 100, 95,
            steps=40, fmt=".0f", accent=ACCENT,
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
            border_color=ACCENT,
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

    def _on_sharpen_mode_change(self):
        """Hiện/ẩn classical vs AI sharpen controls."""
        if self._sharpen_ai_on.get():
            self._sharpen_classical.pack_forget()
            self._sharpen_ai_frame.pack(fill="x")
        else:
            self._sharpen_ai_frame.pack_forget()
            self._sharpen_classical.pack(fill="x")

    def get_pipeline_settings(self):
        """Trả về PipelineSettings từ trạng thái UI."""
        from core.pipeline import PipelineSettings
        scale_str = self._scale_var.get()
        scale = int(scale_str.replace("x", ""))
        return PipelineSettings(
            beauty_enabled=bool(self._beauty_on.get()),
            skin_smooth=float(self._skin_smooth.get()),
            skin_tone=float(self._skin_tone.get()),
            body_slim=float(self._body_slim.get()),
            leg_stretch=float(self._leg_stretch.get()),
            denoise_enabled=bool(self._denoise_on.get()),
            denoise_strength=float(self._denoise_lum.get()),
            denoise_color_strength=float(self._denoise_color.get()),
            upscale_enabled=bool(self._upscale_on.get()),
            upscale_factor=scale,
            upscale_model=self._model_var.get(),
            upscale_max_long_side=(
                int(self._upscale_target_px.get() or 6000)
                if self._upscale_target_on.get() else 0
            ),
            sharpen_enabled=bool(self._sharpen_on.get()),
            sharpen_ai_enabled=bool(self._sharpen_ai_on.get()),
            sharpen_ai_strength=float(self._sharpen_ai_strength.get()),
            sharpen_ai_model=self._sharpen_ai_model_var.get(),
            sharpen_method=self._sharpen_method_var.get(),
            sharpen_amount=float(self._sharpen_amount.get()),
            sharpen_radius=float(self._sharpen_radius.get()),
            sharpen_threshold=int(self._sharpen_thresh.get()),
            sharpen_person_only=bool(self._sharpen_person_only.get()),
            face_restore_enabled=bool(self._face_on.get()),
            face_restore_fidelity=float(self._face_fidelity.get()),
            face_restore_model=self._face_model_var.get(),
            face_restore_high_res=bool(self._face_high_res.get()),
            auto_brighten=bool(self._auto_bright_on.get()),
            auto_brighten_strength=float(self._bright_strength.get()),
            export_format=self._fmt_var.get(),
            export_quality=int(self._quality_slider.get()),
            export_suffix=self._suffix_entry.get() or "_enhanced",
        )

    def apply_settings(self, s):
        """Load PipelineSettings vào UI."""
        self._beauty_on.select() if s.beauty_enabled else self._beauty_on.deselect()
        self._skin_smooth.set(s.skin_smooth)
        self._skin_tone.set(s.skin_tone)
        self._body_slim.set(s.body_slim)
        self._leg_stretch.set(s.leg_stretch)
        
        self._denoise_on.select() if s.denoise_enabled else self._denoise_on.deselect()
        self._denoise_lum.set(s.denoise_strength)
        self._denoise_color.set(s.denoise_color_strength)
        self._upscale_on.select() if s.upscale_enabled else self._upscale_on.deselect()
        self._scale_var.set(f"{s.upscale_factor}x")
        self._model_var.set(s.upscale_model)
        # Restore target size
        if getattr(s, "upscale_max_long_side", 0) > 0:
            self._upscale_target_on.select()
            self._upscale_target_px.delete(0, "end")
            self._upscale_target_px.insert(0, str(s.upscale_max_long_side))
        else:
            self._upscale_target_on.deselect()
        self._sharpen_on.select() if s.sharpen_enabled else self._sharpen_on.deselect()
        self._sharpen_person_only.select() if s.sharpen_person_only else self._sharpen_person_only.deselect()
        self._sharpen_amount.set(s.sharpen_amount)
        self._sharpen_radius.set(s.sharpen_radius)
        self._sharpen_thresh.set(s.sharpen_threshold)
        self._face_on.select() if s.face_restore_enabled else self._face_on.deselect()
        self._face_high_res.select() if getattr(s, "face_restore_high_res", True) else self._face_high_res.deselect()
        self._face_fidelity.set(s.face_restore_fidelity)
        self._face_model_var.set(s.face_restore_model)
        self._fmt_var.set(s.export_format)
        self._quality_slider.set(s.export_quality)
        self._suffix_entry.delete(0, "end")
        self._suffix_entry.insert(0, s.export_suffix)
        self._on_fmt_change(s.export_format)
