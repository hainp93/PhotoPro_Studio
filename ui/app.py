"""
PhotoPro Studio — Main Application Window
Layout: Topbar + Center Viewer + Right Settings Panel
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Theme setup ────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── Design tokens ──────────────────────────────────────────────────────────
BG_APP      = "#121212"
BG_TOPBAR   = "#1e1e1e"
BG_CENTER   = "#0a0a0a"
BG_SETTINGS = "#181818"
ACCENT      = "#2979ff"
ACCENT_G    = "#00e676"
TEXT_HI     = "#ffffff"
TEXT_DIM    = "#a0a0a0"
BORDER      = "#333333"


class PhotoProApp(ctk.CTk):
    """
    Main window:
    ┌────────────────────────────────────────────────────────────┐
    │  Topbar: logo | Mở | Xử Lý | Lưu | Reset | Hủy | GPU    │
    ├─────────────────────────────────────┬──────────────────────┤
    │                                     │                      │
    │       Image Viewer (center)         │   Settings Panel     │
    │                                     │       (320px)        │
    ├─────────────────────────────────────┴──────────────────────┤
    │  Status  |  progress  |  image size  |  GPU info           │
    └────────────────────────────────────────────────────────────┘
    """

    APP_NAME = "PhotoPro Studio"
    VERSION  = "1.0.0"

    def __init__(self):
        super().__init__()
        from utils.config import get_config
        from core.gpu_detector import get_gpu_info

        self._cfg = get_config()
        self._gpu = get_gpu_info()

        # State
        self._input_path: str | None = None
        self._original_image: np.ndarray | None = None
        self._result_image: np.ndarray | None = None
        self._worker = None
        self._batch_worker = None
        self._current_tab = "single"

        self._setup_window()
        self._build_ui()
        self._apply_theme()

        logger.info(f"App started | {self._gpu.summary()}")

    # ── Window ───────────────────────────────────────────────────────────
    def _setup_window(self):
        self.title(f"{self.APP_NAME} v{self.VERSION}")
        self.geometry("1400x900")
        self.minsize(900, 600)
        self.after(0, lambda: self.state("zoomed"))
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        icon_path = Path(__file__).parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.iconbitmap(str(icon_path))

    def _apply_theme(self):
        self.configure(fg_color=BG_APP)

    # ── UI Build ─────────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_menu()
        self._build_topbar()

        # Body: center (expands) + right panel (fixed 320px)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=320)
        body.rowconfigure(0, weight=1)

        self._build_center(body)
        self._build_right_panel(body)
        self._build_statusbar()

    # ── Menu ─────────────────────────────────────────────────────────────
    def _build_menu(self):
        mb = tk.Menu(self, bg="#10101f", fg="#c8d8f0",
                     activebackground="#2a4a7a", activeforeground="white",
                     relief="flat", borderwidth=0)
        self.configure(menu=mb)

        fm = tk.Menu(mb, tearoff=0, bg="#10101f", fg="#c8d8f0",
                     activebackground="#2a4a7a")
        mb.add_cascade(label="File", menu=fm)
        fm.add_command(label="Mở ảnh...          Ctrl+O", command=self._open_file)
        fm.add_command(label="Lưu kết quả...     Ctrl+S", command=self._save_result)
        fm.add_separator()
        fm.add_command(label="Thoát              Alt+F4", command=self._on_close)

        tm = tk.Menu(mb, tearoff=0, bg="#10101f", fg="#c8d8f0",
                     activebackground="#2a4a7a")
        mb.add_cascade(label="Công cụ", menu=tm)
        tm.add_command(label="Giải phóng VRAM", command=self._unload_models)
        tm.add_command(label="GPU Info",         command=self._show_gpu_info)

        hm = tk.Menu(mb, tearoff=0, bg="#10101f", fg="#c8d8f0",
                     activebackground="#2a4a7a")
        mb.add_cascade(label="Trợ giúp", menu=hm)
        hm.add_command(label=f"Phiên bản {self.VERSION}", state="disabled")

        self.bind("<Control-o>", lambda e: self._open_file())
        self.bind("<Control-s>", lambda e: self._save_result())

    # ── Topbar ───────────────────────────────────────────────────────────
    def _build_topbar(self):
        bar = ctk.CTkFrame(self, height=52, fg_color=BG_TOPBAR,
                           corner_radius=0, border_width=1, border_color=BORDER)
        bar.pack(side="top", fill="x")
        bar.pack_propagate(False)

        # Left group: logo + tab buttons
        left = ctk.CTkFrame(bar, fg_color="transparent")
        left.pack(side="left", padx=(10, 0), pady=6)

        # Logo
        ctk.CTkLabel(left, text="📸", font=("Segoe UI Emoji", 20)).pack(side="left")
        ctk.CTkLabel(left, text="PhotoPro",
                     font=("Inter", 13, "bold"), text_color=ACCENT).pack(side="left", padx=(4, 14))

        # Separator
        ctk.CTkFrame(left, width=1, height=28, fg_color=BORDER).pack(side="left", padx=4)

        # Tab buttons: Đơn lẻ / Batch
        tab_kw = {"height": 32, "corner_radius": 8, "font": ("Inter", 11, "bold"),
                  "border_width": 0}
        self._btn_single = ctk.CTkButton(
            left, text="Đơn lẻ",
            command=lambda: self._switch_tab("single"),
            fg_color=ACCENT, hover_color="#3a7aed", text_color="white",
            width=70, **tab_kw)
        self._btn_single.pack(side="left", padx=(8, 2))

        self._btn_batch = ctk.CTkButton(
            left, text="Batch",
            command=lambda: self._switch_tab("batch"),
            fg_color="transparent", hover_color="#1a1a35", text_color=TEXT_DIM,
            width=60, **tab_kw)
        self._btn_batch.pack(side="left")

        # Separator
        ctk.CTkFrame(left, width=1, height=28, fg_color=BORDER).pack(side="left", padx=12)

        # Action buttons
        act_kw = {"height": 32, "corner_radius": 8, "font": ("Inter", 11, "bold"),
                  "border_width": 0}

        ctk.CTkButton(
            left, text="📂  Mở Ảnh",
            command=self._open_file,
            fg_color="transparent", hover_color="#222222", text_color=TEXT_HI,
            border_width=1, border_color=BORDER,
            width=100, **act_kw,
        ).pack(side="left", padx=(0, 6))

        self._btn_process = ctk.CTkButton(
            left, text="⚡  Xử Lý",
            command=self._process_single,
            fg_color=ACCENT, hover_color="#1565c0", text_color="white",
            width=90, state="disabled", **act_kw,
        )
        self._btn_process.pack(side="left", padx=(0, 6))

        self._btn_save = ctk.CTkButton(
            left, text="💾  Lưu",
            command=self._save_result,
            fg_color="transparent", hover_color="#1a2e1e", text_color=ACCENT_G,
            border_width=1, border_color=BORDER,
            width=72, state="disabled", **act_kw,
        )
        self._btn_save.pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            left, text="↺  Reset",
            command=self._reset,
            fg_color="transparent", hover_color="#2a2a2a", text_color=TEXT_DIM,
            border_width=1, border_color=BORDER,
            width=78, **act_kw,
        ).pack(side="left", padx=(0, 6))

        # Hủy button
        self._btn_cancel = ctk.CTkButton(
            left, text="■  Hủy",
            command=self._cancel_process,
            fg_color="transparent", hover_color="#3a1c1c", text_color="#f76f6f",
            border_width=1, border_color=BORDER,
            width=72, state="disabled", **act_kw,
        )
        self._btn_cancel.pack(side="left")

        # Right group: progress + GPU info
        right = ctk.CTkFrame(bar, fg_color="transparent")
        right.pack(side="right", padx=12, pady=6)

        # GPU badge
        gpu_color = {
            "high": ACCENT_G, "mid": "#f5a623",
            "low": "#f76f6f", "cpu": "#5a7090",
        }.get(self._gpu.gpu_tier, "#5a7090")
        gpu_short = self._gpu.device_name
        if not self._gpu.has_cuda:
            gpu_short = "CPU Mode"
        elif len(gpu_short) > 22:
            gpu_short = gpu_short[:22] + "…"

        gpu_badge = ctk.CTkFrame(right, fg_color="#0d1525", corner_radius=8)
        gpu_badge.pack(side="right")
        ctk.CTkLabel(gpu_badge, text=f"🎮 {gpu_short}",
                     font=("Inter", 10), text_color=gpu_color).pack(padx=10, pady=5)

        # Progress area (in topbar, compact)
        prog_frame = ctk.CTkFrame(right, fg_color="transparent")
        prog_frame.pack(side="right", padx=(0, 12))

        self._progress_bar = ctk.CTkProgressBar(
            prog_frame, width=140, height=5,
            progress_color=ACCENT, fg_color="#1a1f35", corner_radius=3,
        )
        self._progress_bar.pack()
        self._progress_bar.set(0)

        self._lbl_step = ctk.CTkLabel(
            prog_frame, text="",
            font=("Inter", 9), text_color=TEXT_DIM, width=140,
        )
        self._lbl_step.pack()

    # ── Center viewer ────────────────────────────────────────────────────
    def _build_center(self, body):
        center = ctk.CTkFrame(body, fg_color=BG_CENTER, corner_radius=0)
        center.grid(row=0, column=0, sticky="nsew")

        # Drop hint
        self._drop_hint = ctk.CTkLabel(
            center,
            text="📂\n\nKéo thả ảnh vào đây\nhoặc nhấn  Ctrl+O  để mở",
            font=("Inter", 18), text_color="#1c2540",
            justify="center",
        )
        self._drop_hint.place(relx=0.5, rely=0.5, anchor="center")

        from ui.widgets.image_viewer import ImageViewer
        self._viewer = ImageViewer(center)
        self._viewer.pack(fill="both", expand=True)

        from ui.panels.batch_panel import BatchPanel
        self._batch_panel = BatchPanel(
            center,
            on_run=self._run_batch,
            on_cancel=self._cancel_batch,
        )

    # ── Right settings panel ─────────────────────────────────────────────
    def _build_right_panel(self, body):
        right = ctk.CTkFrame(body, width=320, fg_color=BG_SETTINGS,
                             corner_radius=0, border_width=1, border_color=BORDER)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_propagate(False)

        from ui.panels.settings_panel import SettingsPanel
        self._settings_panel = SettingsPanel(
            right, on_change=None, gpu=self._gpu
        )
        self._settings_panel.pack(fill="both", expand=True)

    # ── Statusbar ────────────────────────────────────────────────────────
    def _build_statusbar(self):
        bar = ctk.CTkFrame(self, height=26, fg_color="#080810",
                           corner_radius=0, border_width=1, border_color=BORDER)
        bar.pack(side="bottom", fill="x")
        bar.pack_propagate(False)

        self._lbl_status = ctk.CTkLabel(
            bar, text="● Sẵn sàng",
            font=("Inter", 10), text_color=TEXT_DIM, anchor="w",
        )
        self._lbl_status.pack(side="left", padx=10)

        gpu_sum = self._gpu.summary()
        ctk.CTkLabel(bar, text=gpu_sum,
                     font=("Inter", 9), text_color="#283040", anchor="e",
                     ).pack(side="right", padx=10)

    # ── Tab Switching ─────────────────────────────────────────────────────
    def _switch_tab(self, tab: str):
        self._current_tab = tab
        if tab == "single":
            self._btn_single.configure(fg_color=ACCENT, text_color="white")
            self._btn_batch.configure(fg_color="transparent", text_color=TEXT_DIM)
            self._batch_panel.pack_forget()
            self._viewer.pack(fill="both", expand=True)
        else:
            self._btn_single.configure(fg_color="transparent", text_color=TEXT_DIM)
            self._btn_batch.configure(fg_color=ACCENT, text_color="white")
            self._viewer.pack_forget()
            self._batch_panel.pack(fill="both", expand=True)

    # ── File Open / Save ──────────────────────────────────────────────────
    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh",
            initialdir=self._cfg.config.last_input_dir or str(Path.home()),
            filetypes=[
                ("Ảnh & RAW", "*.jpg *.jpeg *.png *.webp *.tif *.tiff *.bmp "
                              "*.nef *.cr2 *.cr3 *.arw *.dng *.raf *.rw2 *.orf"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("RAW", "*.nef *.cr2 *.cr3 *.arw *.dng *.raf *.rw2 *.orf"),
                ("Tất cả", "*.*"),
            ],
        )
        if not path:
            return
        self._cfg.config.last_input_dir = str(Path(path).parent)
        self._load_image(path)

    def _load_image(self, path: str):
        from utils.image_io import read_image
        try:
            self._set_status(f"Đang đọc: {Path(path).name}...")
            img = read_image(path)
            self._input_path = path
            self._original_image = img
            self._result_image = None
            self._viewer.set_before(img)
            self._viewer.set_after(None)
            self._viewer.fit_to_window()
            self._drop_hint.place_forget()
            self._btn_process.configure(state="normal")
            self._btn_save.configure(state="disabled")
            h, w = img.shape[:2]
            self._set_status(f"✅ Đã mở: {Path(path).name}  ({w}×{h}px)")
        except Exception as e:
            messagebox.showerror("Lỗi mở ảnh", str(e))
            self._set_status("❌ Lỗi mở ảnh")

    def _save_result(self):
        if self._result_image is None:
            messagebox.showwarning("Chưa có kết quả", "Hãy xử lý ảnh trước khi lưu.")
            return
        s = self._settings_panel.get_pipeline_settings()
        ext_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "TIFF": ".tiff"}
        ext = ext_map.get(s.export_format, ".png")
        init_name = (
            Path(self._input_path).stem + s.export_suffix + ext
            if self._input_path else f"result{ext}"
        )
        path = filedialog.asksaveasfilename(
            title="Lưu kết quả", initialfile=init_name,
            defaultextension=ext,
            filetypes=[(s.export_format, f"*{ext}"), ("Tất cả", "*.*")],
        )
        if not path:
            return
        from utils.image_io import save_image
        try:
            save_image(self._result_image, path,
                       fmt=s.export_format, quality=s.export_quality)
            self._set_status(f"✅ Đã lưu: {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Lỗi lưu ảnh", str(e))

    # ── Single Processing ─────────────────────────────────────────────────
    def _process_single(self):
        if self._original_image is None:
            return
        s = self._settings_panel.get_pipeline_settings()
        from core.pipeline import get_pipeline
        from core.worker import ProcessingWorker

        self._set_processing_state(True)
        self._set_status("⚡ Đang xử lý...")

        self._worker = ProcessingWorker(
            fn=get_pipeline().process,
            args=(self._original_image,),
            kwargs={"settings": s},
            on_progress=self._on_progress,
            on_done=self._on_process_done,
            on_error=self._on_process_error,
        )
        self._worker.start()

    def _on_progress(self, pct: float, msg: str):
        self.after(0, lambda: self._update_progress(pct, msg))

    def _update_progress(self, pct: float, msg: str):
        self._progress_bar.set(pct / 100)
        self._lbl_step.configure(text=msg)
        self._set_status(f"⚡ {msg}")

    def _on_process_done(self, result: np.ndarray):
        self._result_image = result
        self.after(0, self._show_result)

    def _show_result(self):
        self._viewer.set_after(self._result_image)
        self._set_processing_state(False)
        self._btn_save.configure(state="normal")
        h, w = self._result_image.shape[:2]
        self._progress_bar.set(1.0)
        lbl = self._lbl_step.cget("text")
        if "cảnh báo" in lbl.lower():
            self._set_status(f"⚠ Xong — Đang xem ở 100% zoom | Kéo thanh split để so sánh | {w}×{h}px")
        else:
            self._lbl_step.configure(text="Hoàn thành!")
            self._set_status(f"✅ Xong — Đang xem ở 100% zoom | Kéo thanh split để so sánh | {w}×{h}px")

    def _on_process_error(self, msg: str):
        self.after(0, lambda: self._handle_error(msg))

    def _handle_error(self, msg: str):
        self._set_processing_state(False)
        self._set_status(f"❌ Lỗi: {msg}")
        messagebox.showerror("Lỗi xử lý", msg)

    def _cancel_process(self):
        if self._worker and self._worker.is_running:
            self._worker.cancel()
        self._set_processing_state(False)
        self._set_status("⚪ Đã hủy")

    def _set_processing_state(self, processing: bool):
        self._btn_process.configure(state="disabled" if processing else "normal")
        self._btn_cancel.configure(state="normal" if processing else "disabled")

    # ── Batch ─────────────────────────────────────────────────────────────
    def _run_batch(self, files: list[str], out_dir: str | None):
        from core.pipeline import get_pipeline
        from core.worker import BatchWorker
        from utils.image_io import read_image, save_image, build_output_path

        s = self._settings_panel.get_pipeline_settings()
        pipeline = get_pipeline()

        def process_one(file_path: str, cancel_flag=None) -> str | None:
            img = read_image(file_path)
            result = pipeline.process(img, settings=s, cancel_flag=cancel_flag)
            dest_dir = out_dir or str(Path(file_path).parent / "enhanced")
            dest = build_output_path(file_path, dest_dir, s.export_suffix, s.export_format)
            save_image(result, dest, fmt=s.export_format, quality=s.export_quality)
            return dest

        self._batch_worker = BatchWorker(
            items=files,
            process_fn=process_one,
            on_progress=lambda pct, msg: self.after(
                0, lambda: self._batch_panel.update_progress(pct, msg)
            ),
            on_all_done=lambda results: self.after(0, self._batch_panel.on_batch_done),
        )
        self._batch_worker.start()

    def _cancel_batch(self):
        if self._batch_worker:
            self._batch_worker.cancel()

    # ── Utilities ──────────────────────────────────────────────────────────
    def _reset(self):
        self._result_image = None
        if self._original_image is not None:
            self._viewer.set_after(None)
        self._progress_bar.set(0)
        self._lbl_step.configure(text="")
        self._btn_save.configure(state="disabled")
        self._set_status("↺ Đã reset")

    def _unload_models(self):
        from core.pipeline import get_pipeline
        get_pipeline().unload_models()
        self._set_status("🗑 Đã giải phóng VRAM")

    def _show_gpu_info(self):
        messagebox.showinfo("GPU Info", self._gpu.summary())

    def _load_preset(self, name: str):
        from utils.config import get_config
        data = get_config().load_preset(name)
        if data:
            from core.pipeline import PipelineSettings
            s = PipelineSettings(**{k: v for k, v in data.items()
                                    if hasattr(PipelineSettings, k)})
            self._settings_panel.apply_settings(s)
            self._set_status(f"✅ Đã load preset: {name}")

    def _save_preset(self, name: str):
        from utils.config import get_config
        from dataclasses import asdict
        s = self._settings_panel.get_pipeline_settings()
        get_config().save_preset(name, asdict(s))
        self._set_status(f"✅ Đã lưu preset: {name}")

    def _set_status(self, text: str):
        self._lbl_status.configure(text=text)

    def _on_close(self):
        self._cfg.save()
        self.destroy()
