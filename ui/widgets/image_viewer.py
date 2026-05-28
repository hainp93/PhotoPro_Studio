"""
ImageViewer — Zoomable canvas với before/after split view.
"""
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


class ImageViewer(ctk.CTkFrame):
    """
    Canvas hiển thị ảnh với:
    - Zoom in/out bằng scroll wheel
    - Pan (kéo) bằng click giữ
    - Split before/after với divider kéo được
    - Fit-to-window tự động
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color="#1a1a2e")

        self._before_img: np.ndarray | None = None
        self._after_img: np.ndarray | None = None
        self._before_tk: ImageTk.PhotoImage | None = None
        self._after_tk: ImageTk.PhotoImage | None = None

        self._zoom = 1.0
        self._min_zoom = 0.05
        self._max_zoom = 8.0
        self._pan_x = 0
        self._pan_y = 0
        self._drag_start = None

        # Split divider (0.0 - 1.0)
        self._split_pos = 0.5
        self._dragging_split = False
        self._split_mode = True   # True = split view, False = single view

        # Canvas
        self._canvas = tk.Canvas(
            self,
            bg="#1a1a2e",
            highlightthickness=0,
            cursor="crosshair",
        )
        self._canvas.pack(fill="both", expand=True)

        # Toolbar overlay
        self._build_toolbar()

        # Bindings
        self._canvas.bind("<MouseWheel>", self._on_scroll)
        self._canvas.bind("<Button-4>", self._on_scroll)   # Linux
        self._canvas.bind("<Button-5>", self._on_scroll)
        self._canvas.bind("<ButtonPress-1>", self._on_press)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Configure>", self._on_resize)
        self._canvas.bind("<Double-Button-1>", self._fit_to_window)

        # Hover info
        self._canvas.bind("<Motion>", self._on_mouse_move)

    def _build_toolbar(self):
        self._toolbar = ctk.CTkFrame(self, fg_color="#0d0d1a", height=36, corner_radius=0)
        self._toolbar.pack(side="bottom", fill="x")
        self._toolbar.pack_propagate(False)

        btn_style = {"width": 80, "height": 28, "corner_radius": 6, "font": ("Inter", 11)}

        self._btn_fit = ctk.CTkButton(
            self._toolbar, text="Fit", command=self._fit_to_window, **btn_style
        )
        self._btn_fit.pack(side="left", padx=(6, 2), pady=4)

        self._btn_100 = ctk.CTkButton(
            self._toolbar, text="100%", command=self._zoom_100, **btn_style
        )
        self._btn_100.pack(side="left", padx=2, pady=4)

        self._btn_split = ctk.CTkButton(
            self._toolbar, text="Split ◧",
            command=self._toggle_split, **btn_style,
            fg_color="#2d5a8e",
        )
        self._btn_split.pack(side="left", padx=2, pady=4)

        self._lbl_zoom = ctk.CTkLabel(
            self._toolbar, text="100%",
            font=("Inter", 11), text_color="#8899aa",
        )
        self._lbl_zoom.pack(side="right", padx=8)

        self._lbl_info = ctk.CTkLabel(
            self._toolbar, text="Chưa có ảnh",
            font=("Inter", 11), text_color="#6688aa",
        )
        self._lbl_info.pack(side="right", padx=8)

    # ── Public API ───────────────────────────────────────────────────
    def set_before(self, image: np.ndarray | None):
        self._before_img = image
        self._redraw()

    def set_after(self, image: np.ndarray | None):
        self._after_img = image
        self._redraw()

    def clear(self):
        self._before_img = None
        self._after_img = None
        self._canvas.delete("all")
        self._lbl_info.configure(text="Chưa có ảnh")

    def fit_to_window(self):
        self._fit_to_window()

    # ── Internal rendering ────────────────────────────────────────────
    def _fit_to_window(self, event=None):
        img = self._before_img if self._before_img is not None else self._after_img
        if img is None:
            return
        cw = self._canvas.winfo_width() or 800
        ch = self._canvas.winfo_height() or 600
        ih, iw = img.shape[:2]
        self._zoom = min(cw / iw, ch / ih) * 0.95
        self._pan_x = 0
        self._pan_y = 0
        self._redraw()

    def _zoom_100(self):
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._redraw()

    def _toggle_split(self):
        self._split_mode = not self._split_mode
        label = "Split ◧" if self._split_mode else "After ◨"
        self._btn_split.configure(text=label)
        self._redraw()

    def _redraw(self):
        self._canvas.delete("all")
        cw = self._canvas.winfo_width() or 800
        ch = self._canvas.winfo_height() or 600

        # Chọn ảnh để hiển thị
        if self._split_mode and self._before_img is not None and self._after_img is not None:
            self._draw_split(cw, ch)
        elif self._after_img is not None:
            self._draw_single(self._after_img, cw, ch)
        elif self._before_img is not None:
            self._draw_single(self._before_img, cw, ch)

        self._lbl_zoom.configure(text=f"{self._zoom*100:.0f}%")

    def _np_to_pil(self, img: np.ndarray, w: int, h: int) -> Image.Image:
        """Resize numpy BGR → PIL RGB."""
        iw = int(img.shape[1] * self._zoom)
        ih = int(img.shape[0] * self._zoom)
        if iw <= 0 or ih <= 0:
            iw, ih = 1, 1
        resized = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_LANCZOS4)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _draw_single(self, img: np.ndarray, cw: int, ch: int):
        pil = self._np_to_pil(img, cw, ch)
        iw, ih = pil.size
        x = cw // 2 - iw // 2 + self._pan_x
        y = ch // 2 - ih // 2 + self._pan_y
        self._after_tk = ImageTk.PhotoImage(pil)
        self._canvas.create_image(x, y, anchor="nw", image=self._after_tk)
        self._lbl_info.configure(text=f"{img.shape[1]}×{img.shape[0]}px")

    def _draw_split(self, cw: int, ch: int):
        split_x = int(cw * self._split_pos)

        before_pil = self._np_to_pil(self._before_img, cw, ch)
        after_pil = self._np_to_pil(self._after_img, cw, ch)

        iw, ih = before_pil.size
        ox = cw // 2 - iw // 2 + self._pan_x
        oy = ch // 2 - ih // 2 + self._pan_y

        # Composite: before bên trái split, after bên phải
        combined = Image.new("RGB", (cw, ch), (26, 26, 46))
        # Before
        before_crop_w = max(0, split_x - ox)
        if before_crop_w > 0 and iw > 0:
            before_cropped = before_pil.crop((0, 0, min(before_crop_w, iw), ih))
            combined.paste(before_cropped, (ox, oy))
        # After
        after_start = max(0, split_x - ox)
        if after_start < iw:
            after_cropped = after_pil.crop((after_start, 0, iw, ih))
            combined.paste(after_cropped, (ox + after_start, oy))

        # Divider line
        draw = ImageDraw.Draw(combined)
        draw.line([(split_x, 0), (split_x, ch)], fill="#00d4ff", width=2)

        # Labels
        draw.rectangle([split_x - 60, 8, split_x - 4, 28], fill=(0, 0, 0, 180))
        draw.text((split_x - 56, 11), "BEFORE", fill="#aaccee")
        draw.rectangle([split_x + 4, 8, split_x + 56, 28], fill=(0, 0, 0, 180))
        draw.text((split_x + 8, 11), "AFTER", fill="#00ffcc")

        self._before_tk = ImageTk.PhotoImage(combined)
        self._canvas.create_image(0, 0, anchor="nw", image=self._before_tk)

        h, w = self._before_img.shape[:2]
        self._lbl_info.configure(text=f"{w}×{h}px")

    # ── Events ────────────────────────────────────────────────────────
    def _on_scroll(self, event):
        if event.num == 4 or event.delta > 0:
            self._zoom = min(self._zoom * 1.15, self._max_zoom)
        else:
            self._zoom = max(self._zoom / 1.15, self._min_zoom)
        self._redraw()

    def _on_press(self, event):
        # Check nếu click gần divider
        cw = self._canvas.winfo_width()
        split_x = int(cw * self._split_pos)
        if self._split_mode and abs(event.x - split_x) < 12:
            self._dragging_split = True
        else:
            self._drag_start = (event.x - self._pan_x, event.y - self._pan_y)
            self._dragging_split = False

    def _on_drag(self, event):
        if self._dragging_split:
            cw = self._canvas.winfo_width()
            self._split_pos = max(0.05, min(0.95, event.x / cw))
            self._redraw()
        elif self._drag_start:
            self._pan_x = event.x - self._drag_start[0]
            self._pan_y = event.y - self._drag_start[1]
            self._redraw()

    def _on_release(self, event):
        self._drag_start = None
        self._dragging_split = False

    def _on_resize(self, event):
        self._redraw()

    def _on_mouse_move(self, event):
        cw = self._canvas.winfo_width()
        split_x = int(cw * self._split_pos)
        if self._split_mode and abs(event.x - split_x) < 12:
            self._canvas.configure(cursor="sb_h_double_arrow")
        else:
            self._canvas.configure(cursor="fleur" if self._drag_start else "crosshair")
