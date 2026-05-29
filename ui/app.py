"""
PhotoPro Studio — Main Application Window
"""
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import logging
import os
from pathlib import Path
from dataclasses import asdict

logger = logging.getLogger(__name__)

# ─── Theme setup ────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── Design tokens ──────────────────────────────────────────────────────────
BG_APP        = "#0e0e1c"
BG_SIDEBAR    = "#111128"
BG_LOGO       = "#1a1b38"
BG_CENTER     = "#0e0e1c"
BG_SETTINGS   = "#0e0e1c"
ACCENT_BLUE   = "#4f8ef7"
ACCENT_CYAN   = "#3ecf8e"
TEXT_PRIMARY  = "#dde6ff"
TEXT_DIM      = "#5c7aaa"
BORDER        = "#252545"

class PhotoProApp(ctk.CTk):
    """
    Main window với layout:
    ┌──────────────┬──────────────────────────────┬──────────────┐
    │              │                              │              │
    │   Sidebar    │     Image Viewer (Center)    │   Settings   │
    │  (controls)  │     Before / After Split     │   Panel      │
    │              │                              │              │
    └──────────────┴──────────────────────────────┴──────────────┘
    │                    Status Bar                               │
    └─────────────────────────────────────────────────────────────┘
    """

    APP_NAME = "PhotoPro Studio"
    VERSION = "1.0.0"

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
        self._current_tab = "single"   # "single" | "batch"

        self._setup_window()
        self._build_ui()
        self._apply_theme()

        logger.info(f"App started | {self._gpu.summary()}")

    # ── Window Setup ─────────────────────────────────────────────────────
    def _setup_window(self):
        self.title(f"{self.APP_NAME} v{self.VERSION}")
        w = self._cfg.config.window_width
        h = self._cfg.config.window_height
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.minsize(1000, 650)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Icon nếu có
        icon_path = Path(__file__).parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.iconbitmap(str(icon_path))

    def _apply_theme(self):
        self.configure(fg_color=BG_APP)

    # ── UI Build ─────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top menubar
        self._build_menu()

        # Main layout: 3 columns
        self._main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._main_frame.pack(fill="both", expand=True)
        # Column 0 = sidebar (fixed), col 1 = center (expands), col 2 = settings (fixed)
        self._main_frame.columnconfigure(0, weight=0, minsize=170)
        self._main_frame.columnconfigure(1, weight=1)
        self._main_frame.columnconfigure(2, weight=0, minsize=320)
        self._main_frame.rowconfigure(0, weight=1)

        # Left sidebar
        self._build_sidebar()
        # Center viewer
        self._build_center()
        # Right settings
        self._build_right_panel()
        # Bottom status bar
        self._build_statusbar()

    def _build_menu(self):
        menubar = tk.Menu(self, bg="#1a1a40", fg="#e0e8ff",
                          activebackground="#2d5a8e", activeforeground="white",
                          relief="flat", borderwidth=0)
        self.configure(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0, bg="#1a1a40", fg="#e0e8ff",
                            activebackground="#2d5a8e")
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Mở ảnh...          Ctrl+O", command=self._open_file)
        file_menu.add_command(label="Lưu kết quả...     Ctrl+S", command=self._save_result)
        file_menu.add_separator()
        file_menu.add_command(label="Thoát              Alt+F4", command=self._on_close)

        tools_menu = tk.Menu(menubar, tearoff=0, bg="#1a1a40", fg="#e0e8ff",
                             activebackground="#2d5a8e")
        menubar.add_cascade(label="Công cụ", menu=tools_menu)
        tools_menu.add_command(label="Giải phóng VRAM", command=self._unload_models)
        tools_menu.add_command(label="GPU Info", command=self._show_gpu_info)

        help_menu = tk.Menu(menubar, tearoff=0, bg="#1a1a40", fg="#e0e8ff",
                            activebackground="#2d5a8e")
        menubar.add_cascade(label="Trợ giúp", menu=help_menu)
        help_menu.add_command(label=f"Phiên bản {self.VERSION}", state="disabled")

        # Keyboard shortcuts
        self.bind("<Control-o>", lambda e: self._open_file())
        self.bind("<Control-s>", lambda e: self._save_result())

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(
            self._main_frame, width=170,
            fg_color=BG_SIDEBAR, corner_radius=0,
            border_width=1, border_color=BORDER,
        )
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        # Logo / title
        logo_frame = ctk.CTkFrame(sidebar, fg_color=BG_LOGO, corner_radius=0, height=64)
        logo_frame.pack(fill="x")
        logo_frame.pack_propagate(False)
        ctk.CTkLabel(logo_frame, text="📸", font=("Segoe UI Emoji", 22)).pack(pady=(8, 0))
        ctk.CTkLabel(
            logo_frame, text="PhotoPro Studio",
            font=("Inter Bold", 11, "bold"),
            text_color=ACCENT_BLUE,
        ).pack()

        # GPU badge
        gpu_color = {
            "high": ACCENT_CYAN, "mid": "#f5a623",
            "low": "#f76f6f", "cpu": "#7a94c0",
        }.get(self._gpu.gpu_tier, "#7a94c0")
        gpu_text = (
            self._gpu.device_name[:20] + "…"
            if len(self._gpu.device_name) > 20
            else self._gpu.device_name
        )
        gpu_frame = ctk.CTkFrame(sidebar, fg_color="#181830", corner_radius=6)
        gpu_frame.pack(fill="x", padx=8, pady=(6, 2))
        ctk.CTkLabel(
            gpu_frame, text=f"🎮  {gpu_text}",
            font=("Inter", 10), text_color=gpu_color,
            wraplength=175, anchor="w",
        ).pack(fill="x", padx=8, pady=4)

        # Tab buttons
        tab_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        tab_frame.pack(fill="x", padx=8, pady=(8, 4))

        tab_style = {"height": 36, "corner_radius": 8, "font": ("Inter", 12, "bold")}
        self._btn_single = ctk.CTkButton(
            tab_frame, text="🖼  Đơn lẻ",
            command=lambda: self._switch_tab("single"),
            fg_color=ACCENT_BLUE, hover_color="#3a7aed",
            text_color="white", **tab_style,
        )
        self._btn_single.pack(fill="x", pady=(0, 3))

        self._btn_batch = ctk.CTkButton(
            tab_frame, text="📦  Batch",
            command=lambda: self._switch_tab("batch"),
            fg_color="#1e1e3a", hover_color="#252550",
            text_color=TEXT_DIM, **tab_style,
        )
        self._btn_batch.pack(fill="x")

        # Action buttons (chỉ single mode)
        self._sidebar_actions = ctk.CTkFrame(sidebar, fg_color="transparent")
        self._sidebar_actions.pack(fill="x", padx=8)

        sep = ctk.CTkFrame(self._sidebar_actions, fg_color=BORDER, height=1)
        sep.pack(fill="x", pady=6)

        action_style = {"height": 38, "corner_radius": 8, "font": ("Inter", 12, "bold")}

        ctk.CTkButton(
            self._sidebar_actions, text="📂  Mở Ảnh",
            command=self._open_file,
            fg_color="#1e1e3a", hover_color="#2a2a55",
            text_color=TEXT_PRIMARY, **action_style,
        ).pack(fill="x", pady=(0, 3))

        self._btn_process = ctk.CTkButton(
            self._sidebar_actions, text="⚡  Xử Lý",
            command=self._process_single,
            fg_color=ACCENT_BLUE, hover_color="#3a7aed",
            text_color="white", **action_style,
            state="disabled",
        )
        self._btn_process.pack(fill="x", pady=(0, 3))

        self._btn_save = ctk.CTkButton(
            self._sidebar_actions, text="💾  Lưu Kết Quả",
            command=self._save_result,
            fg_color="#1f4d2a", hover_color="#2a6638",
            text_color="#3ecf8e", **action_style,
            state="disabled",
        )
        self._btn_save.pack(fill="x", pady=(0, 3))

        ctk.CTkButton(
            self._sidebar_actions, text="↺  Reset",
            command=self._reset,
            fg_color="#2a1a1a", hover_color="#3d2020",
            text_color="#c07070", **action_style,
        ).pack(fill="x")

        # Progress
        sep2 = ctk.CTkFrame(sidebar, fg_color=BORDER, height=1)
        sep2.pack(fill="x", padx=8, pady=6)

        self._sidebar_progress = ctk.CTkFrame(sidebar, fg_color="transparent")
        self._sidebar_progress.pack(fill="x", padx=8)

        self._progress_bar = ctk.CTkProgressBar(
            self._sidebar_progress, height=6,
            progress_color=ACCENT_BLUE, fg_color="#1e1e3a",
            corner_radius=3,
        )
        self._progress_bar.pack(fill="x", pady=(0, 3))
        self._progress_bar.set(0)

        self._lbl_step = ctk.CTkLabel(
            self._sidebar_progress, text="",
            font=("Inter", 10), text_color=TEXT_DIM,
            wraplength=180,
        )
        self._lbl_step.pack()

        self._btn_cancel = ctk.CTkButton(
            sidebar, text="■  Hủy",
            command=self._cancel_process,
            fg_color="#3d1a1a", hover_color="#5a2020",
            text_color="#f76f6f",
            height=34, corner_radius=8, font=("Inter", 11, "bold"),
            state="disabled",
        )
        self._btn_cancel.pack(fill="x", padx=8, pady=4)

        # Bottom: Preset manager
        from ui.widgets.preset_manager import PresetManagerWidget
        self._preset_mgr = PresetManagerWidget(
            sidebar,
            on_load=self._load_preset,
            on_save=self._save_preset,
        )
        self._preset_mgr.pack(fill="x", padx=10, side="bottom", pady=10)

    def _build_center(self):
        center = ctk.CTkFrame(self._main_frame, fg_color=BG_CENTER, corner_radius=0)
        center.grid(row=0, column=1, sticky="nsew")

        # Drop zone hint
        self._drop_hint = ctk.CTkLabel(
            center,
            text="📂\n\nKéo thả ảnh vào đây\nhoặc nhấn  Ctrl+O  để mở",
            font=("Inter", 17), text_color="#2a3860",
            justify="center",
        )
        self._drop_hint.place(relx=0.5, rely=0.5, anchor="center")

        # Image viewer
        from ui.widgets.image_viewer import ImageViewer
        self._viewer = ImageViewer(center)
        self._viewer.pack(fill="both", expand=True)

        # Batch panel (hidden initially)
        from ui.panels.batch_panel import BatchPanel
        self._batch_panel = BatchPanel(
            center,
            on_run=self._run_batch,
            on_cancel=self._cancel_batch,
        )

    def _build_right_panel(self):
        right = ctk.CTkFrame(
            self._main_frame, width=320,
            fg_color=BG_SETTINGS, corner_radius=0,
            border_width=1, border_color=BORDER,
        )
        right.grid(row=0, column=2, sticky="nsew")
        right.grid_propagate(False)

        from ui.panels.settings_panel import SettingsPanel
        self._settings_panel = SettingsPanel(right, on_change=None)
        self._settings_panel.pack(fill="both", expand=True)

    def _build_statusbar(self):
        self._statusbar = ctk.CTkFrame(
            self, height=30,
            fg_color="#0a0a1e", corner_radius=0,
            border_width=1, border_color=BORDER,
        )
        self._statusbar.pack(side="bottom", fill="x")
        self._statusbar.pack_propagate(False)

        self._lbl_status = ctk.CTkLabel(
            self._statusbar, text="Sẵn sàng",
            font=("Inter", 10), text_color=TEXT_DIM, anchor="w",
        )
        self._lbl_status.pack(side="left", padx=10)

        gpu_summary = self._gpu.summary()
        self._lbl_gpu = ctk.CTkLabel(
            self._statusbar, text=gpu_summary,
            font=("Inter", 9), text_color="#364560", anchor="e",
        )
        self._lbl_gpu.pack(side="right", padx=10)

    # ── Tab Switching ─────────────────────────────────────────────────────
    def _switch_tab(self, tab: str):
        self._current_tab = tab
        if tab == "single":
            self._btn_single.configure(fg_color=ACCENT_BLUE, text_color="white")
            self._btn_batch.configure(fg_color="#1e1e3a", text_color=TEXT_DIM)
            self._batch_panel.pack_forget()
            self._viewer.pack(fill="both", expand=True)
            self._sidebar_actions.pack(fill="x", padx=10)
        else:
            self._btn_single.configure(fg_color="#1e1e3a", text_color=TEXT_DIM)
            self._btn_batch.configure(fg_color=ACCENT_BLUE, text_color="white")
            self._viewer.pack_forget()
            self._batch_panel.pack(fill="both", expand=True)
            self._sidebar_actions.pack_forget()

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
            self._set_status(f"✅ Đã mở: {Path(path).name} ({w}×{h}px)")
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
        init_name = (Path(self._input_path).stem + s.export_suffix + ext) if self._input_path else f"result{ext}"

        path = filedialog.asksaveasfilename(
            title="Lưu kết quả",
            initialfile=init_name,
            defaultextension=ext,
            filetypes=[(s.export_format, f"*{ext}"), ("Tất cả", "*.*")],
        )
        if not path:
            return

        from utils.image_io import save_image
        try:
            save_image(self._result_image, path, fmt=s.export_format, quality=s.export_quality)
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

        pipeline = get_pipeline()

        self._set_processing_state(True)
        self._set_status("⚡ Đang xử lý...")

        self._worker = ProcessingWorker(
            fn=pipeline.process,
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
        self._set_status(f"✅ Xử lý xong! Kết quả: {w}×{h}px")
        self._progress_bar.set(1.0)
        self._lbl_step.configure(text="Hoàn thành!")

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
        state = "disabled" if processing else "normal"
        self._btn_process.configure(state=state)
        self._btn_cancel.configure(state="normal" if processing else "disabled")

    # ── Batch Processing ──────────────────────────────────────────────────
    def _run_batch(self, files: list[str], out_dir: str | None):
        from core.pipeline import get_pipeline, PipelineSettings
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
            on_progress=lambda pct, msg: self.after(0, lambda: self._batch_panel.update_progress(pct, msg)),
            on_all_done=lambda results: self.after(0, self._batch_panel.on_batch_done),
        )
        self._batch_worker.start()

    def _cancel_batch(self):
        if self._batch_worker:
            self._batch_worker.cancel()

    # ── Utilities ─────────────────────────────────────────────────────────
    def _reset(self):
        self._result_image = None
        if self._original_image is not None:
            self._viewer.set_after(None)
        self._progress_bar.set(0)
        self._lbl_step.configure(text="")
        self._btn_save.configure(state="disabled")
        self._set_status("↺ Đã reset")

    def _load_preset(self, name: str):
        from utils.config import get_config
        from core.pipeline import PipelineSettings
        data = get_config().load_preset(name)
        if data:
            s = PipelineSettings(**{k: v for k, v in data.items() if hasattr(PipelineSettings, k)})
            self._settings_panel.apply_settings(s)
            self._set_status(f"✅ Đã load preset: {name}")

    def _save_preset(self, name: str):
        from utils.config import get_config
        from dataclasses import asdict
        s = self._settings_panel.get_pipeline_settings()
        get_config().save_preset(name, asdict(s))
        self._set_status(f"✅ Đã lưu preset: {name}")

    def _unload_models(self):
        from core.pipeline import get_pipeline
        get_pipeline().unload_models()
        self._set_status("🗑 Đã giải phóng VRAM")

    def _show_gpu_info(self):
        messagebox.showinfo("GPU Info", self._gpu.summary())

    def _set_status(self, text: str):
        self._lbl_status.configure(text=text)

    def _on_close(self):
        self._cfg.config.window_width = self.winfo_width()
        self._cfg.config.window_height = self.winfo_height()
        self._cfg.save()
        self.destroy()
