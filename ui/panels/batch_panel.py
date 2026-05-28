"""
Batch Panel — Xử lý hàng loạt ảnh với queue và progress.
"""
import tkinter as tk
import customtkinter as ctk
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BatchPanel(ctk.CTkFrame):
    """
    Panel batch processing:
    - Queue file với thumbnail
    - Progress per-file và tổng thể
    - Pause / Cancel / Clear
    """

    def __init__(self, master, on_run=None, on_cancel=None, **kwargs):
        super().__init__(master, fg_color="#13132a", corner_radius=0, **kwargs)
        self._on_run = on_run or (lambda files, out_dir: None)
        self._on_cancel = on_cancel or (lambda: None)
        self._files: list[str] = []
        self._build_ui()

    def _build_ui(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color="#1a1a40", corner_radius=0)
        hdr.pack(fill="x")
        ctk.CTkLabel(hdr, text="📦  Batch Processing",
                     font=("Inter Bold", 14, "bold"),
                     text_color="#e0e8ff").pack(side="left", padx=12, pady=10)

        # Buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=6)

        btn_style = {"height": 32, "corner_radius": 8, "font": ("Inter", 11)}
        ctk.CTkButton(btn_row, text="+ Thêm File",
                      command=self._add_files, **btn_style).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="+ Thêm Folder",
                      command=self._add_folder, **btn_style).pack(side="left", padx=2)
        ctk.CTkButton(btn_row, text="Xóa Tất Cả",
                      command=self._clear,
                      fg_color="#8B1A1A", **btn_style).pack(side="right", padx=2)

        # File list
        list_frame = ctk.CTkFrame(self, fg_color="#0f0f2a", corner_radius=8)
        list_frame.pack(fill="both", expand=True, padx=8, pady=4)

        self._listbox = tk.Listbox(
            list_frame,
            bg="#0f0f2a", fg="#aabbcc",
            selectbackground="#2d5a8e",
            selectforeground="white",
            font=("Consolas", 10),
            relief="flat",
            activestyle="none",
            highlightthickness=0,
        )
        self._listbox.pack(side="left", fill="both", expand=True, padx=4, pady=4)

        scrollbar = ctk.CTkScrollbar(list_frame, command=self._listbox.yview)
        scrollbar.pack(side="right", fill="y", pady=4)
        self._listbox.configure(yscrollcommand=scrollbar.set)

        # Bind delete key
        self._listbox.bind("<Delete>", self._remove_selected)

        # Count label
        self._lbl_count = ctk.CTkLabel(self, text="0 file",
                                        font=("Inter", 11), text_color="#6688aa")
        self._lbl_count.pack(pady=2)

        # Output dir
        out_frame = ctk.CTkFrame(self, fg_color="#1a1a40", corner_radius=8)
        out_frame.pack(fill="x", padx=8, pady=4)
        ctk.CTkLabel(out_frame, text="Thư mục output:", font=("Inter", 11)).pack(side="left", padx=8)
        self._out_entry = ctk.CTkEntry(out_frame, font=("Inter", 11), placeholder_text="(cùng thư mục ảnh)")
        self._out_entry.pack(side="left", fill="x", expand=True, padx=4)
        ctk.CTkButton(out_frame, text="...", width=40, command=self._pick_output,
                      font=("Inter", 11)).pack(side="right", padx=4)

        # Progress
        prog_frame = ctk.CTkFrame(self, fg_color="transparent")
        prog_frame.pack(fill="x", padx=8, pady=4)

        self._progress_bar = ctk.CTkProgressBar(prog_frame, height=8,
                                                  progress_color="#00d4ff",
                                                  fg_color="#1a1a40")
        self._progress_bar.pack(fill="x", pady=(0, 4))
        self._progress_bar.set(0)

        self._lbl_progress = ctk.CTkLabel(prog_frame, text="",
                                           font=("Inter", 10), text_color="#8899aa")
        self._lbl_progress.pack()

        # Run / Cancel buttons
        act_row = ctk.CTkFrame(self, fg_color="transparent")
        act_row.pack(fill="x", padx=8, pady=(4, 8))

        self._btn_run = ctk.CTkButton(
            act_row, text="▶  Bắt Đầu Xử Lý",
            command=self._run,
            font=("Inter Bold", 13, "bold"),
            height=38, corner_radius=10,
            fg_color="#1565C0", hover_color="#1976D2",
        )
        self._btn_run.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self._btn_cancel = ctk.CTkButton(
            act_row, text="■  Dừng",
            command=self._cancel,
            font=("Inter", 12), height=38, corner_radius=10,
            fg_color="#8B1A1A", hover_color="#C62828",
            state="disabled",
        )
        self._btn_cancel.pack(side="right", padx=(4, 0))

    def _add_files(self):
        from tkinter import filedialog
        paths = filedialog.askopenfilenames(
            title="Chọn ảnh",
            filetypes=[
                ("Ảnh", "*.jpg *.jpeg *.png *.webp *.tif *.tiff *.bmp"),
                ("RAW", "*.nef *.cr2 *.cr3 *.arw *.dng *.raf *.rw2 *.orf"),
                ("Tất cả", "*.*"),
            ],
        )
        for p in paths:
            if p not in self._files:
                self._files.append(p)
                self._listbox.insert("end", Path(p).name)
        self._update_count()

    def _add_folder(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Chọn thư mục ảnh")
        if not folder:
            return
        exts = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff",
                ".bmp", ".nef", ".cr2", ".cr3", ".arw", ".dng", ".raf"}
        for p in sorted(Path(folder).iterdir()):
            if p.suffix.lower() in exts and str(p) not in self._files:
                self._files.append(str(p))
                self._listbox.insert("end", p.name)
        self._update_count()

    def _remove_selected(self, event=None):
        sel = self._listbox.curselection()
        for i in reversed(sel):
            self._listbox.delete(i)
            self._files.pop(i)
        self._update_count()

    def _clear(self):
        self._listbox.delete(0, "end")
        self._files.clear()
        self._update_count()
        self._progress_bar.set(0)
        self._lbl_progress.configure(text="")

    def _pick_output(self):
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Chọn thư mục output")
        if folder:
            self._out_entry.delete(0, "end")
            self._out_entry.insert(0, folder)

    def _update_count(self):
        n = len(self._files)
        self._lbl_count.configure(text=f"{n} file{'s' if n != 1 else ''}")

    def _run(self):
        if not self._files:
            return
        out_dir = self._out_entry.get().strip() or None
        self._btn_run.configure(state="disabled")
        self._btn_cancel.configure(state="normal")
        self._on_run(list(self._files), out_dir)

    def _cancel(self):
        self._on_cancel()
        self._btn_run.configure(state="normal")
        self._btn_cancel.configure(state="disabled")

    def update_progress(self, pct: float, message: str):
        self._progress_bar.set(pct / 100)
        self._lbl_progress.configure(text=message)

    def on_batch_done(self):
        self._btn_run.configure(state="normal")
        self._btn_cancel.configure(state="disabled")
        self._progress_bar.set(1.0)
        self._lbl_progress.configure(text="✅ Hoàn thành!")
