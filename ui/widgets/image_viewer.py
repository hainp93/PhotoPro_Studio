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
        self.configure(fg_color="#0e0e1c")

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
        self._space_down = False  # Giữ Space để kéo ảnh (giống Photoshop)
        self._space_pan_start = None  # Lưu điểm bắt đầu pan khi giữ Space

        # Split divider (0.0 - 1.0)
        self._split_pos = 0.5
        self._dragging_split = False
        self._split_mode = True   # True = split view, False = single view

        # Face Detection Interactive Mode
        self._interactive_face_mode = False
        self._ai_bboxes = []       # [(x1, y1, x2, y2), ...]
        self._manual_bboxes = []   # [(x1, y1, x2, y2), ...]
        
        # Body Detection Interactive Mode
        self._interactive_body_mode = False
        self._ai_body_bboxes = []  # [(x1, y1, x2, y2), ...]
        self._excluded_body_indices = set()  # Chỉ số các bbox bị bỏ qua
        
        self._drawing_box_start = None  # (canvas_x, canvas_y)
        self._temp_box_id = None
        self._on_manual_box_cb = None

        # Canvas
        self._canvas = tk.Canvas(
            self,
            bg="#0e0e1c",
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
        
        # Space key pan (Photoshop style) — bind lên toàn bộ cửa sổ (không cần canvas focus)
        # Dùng bind_all để bắt mọi nơi trong app
        self.bind_all("<KeyPress-space>", self._on_space_press)
        self.bind_all("<KeyRelease-space>", self._on_space_release)

    def _build_toolbar(self):
        self._toolbar = ctk.CTkFrame(
            self, fg_color="#111128", height=40, corner_radius=0,
            border_width=1, border_color="#252545",
        )
        self._toolbar.pack(side="bottom", fill="x")
        self._toolbar.pack_propagate(False)

        btn_style = {
            "width": 70, "height": 30, "corner_radius": 8,
            "font": ("Inter", 11),
            "fg_color": "#1e1e3a", "hover_color": "#2d2d55",
            "text_color": "#aabbdd",
        }

        self._btn_fit = ctk.CTkButton(
            self._toolbar, text="Fit", command=self._fit_to_window, **btn_style
        )
        self._btn_fit.pack(side="left", padx=(8, 3), pady=5)

        self._btn_100 = ctk.CTkButton(
            self._toolbar, text="100%", command=self._zoom_100, **btn_style
        )
        self._btn_100.pack(side="left", padx=3, pady=5)

        self._btn_split = ctk.CTkButton(
            self._toolbar, text="Split ◧",
            command=self._toggle_split,
            width=80, height=30, corner_radius=8, font=("Inter", 11),
            fg_color="#23355a", hover_color="#2d4a7a",
            text_color="#7ab3f7",
        )
        self._btn_split.pack(side="left", padx=3, pady=5)

        # Separator
        ctk.CTkFrame(
            self._toolbar, width=1, fg_color="#252545",
        ).pack(side="left", fill="y", padx=6, pady=8)

        self._lbl_zoom = ctk.CTkLabel(
            self._toolbar, text="100%",
            font=("Inter", 11, "bold"), text_color="#4f8ef7",
        )
        self._lbl_zoom.pack(side="right", padx=10)

        self._lbl_info = ctk.CTkLabel(
            self._toolbar, text="Chưa có ảnh",
            font=("Inter", 11), text_color="#5c7aaa",
        )
        self._lbl_info.pack(side="right", padx=6)

    # ── Public API ───────────────────────────────────────────────────
    def set_before(self, image: np.ndarray | None):
        self._before_img = image
        # Reset về split mode mỗi khi load ảnh mới
        self._split_mode = True
        self._btn_split.configure(text="Split ◧", fg_color="#23355a", text_color="#7ab3f7")
        self._redraw()

    def set_after(self, image: np.ndarray | None):
        self._after_img = image
        if image is not None:
            # Tắt chế độ vẽ box khi có ảnh after
            self._interactive_face_mode = False
            # Chuyển sang Split mode để so sánh trước/sau
            self._split_mode = True
            self._split_pos = 0.5
            self._btn_split.configure(text="Split ▧", fg_color="#23355a", text_color="#7ab3f7")
            # Zoom 100% — chỉ cách duy nhất thấy rõ hiệu ứng
            self._zoom = 1.0
            self._pan_x = 0
            self._pan_y = 0
        else:
            self._split_mode = True
            self._btn_split.configure(text="Split ▧", fg_color="#23355a", text_color="#7ab3f7")
        self._redraw()

    def set_face_interactive_mode(self, active: bool, ai_bboxes: list = None, callback=None):
        """Bật/tắt chế độ cho phép vẽ box khuôn mặt."""
        self._interactive_face_mode = active
        if active:
            # Force single view on before_img
            self._split_mode = False
            self._ai_bboxes = ai_bboxes or []
            self._manual_bboxes = []
            self._on_manual_box_cb = callback
            self._canvas.configure(cursor="crosshair")
        else:
            self._ai_bboxes = []
            self._manual_bboxes = []
            self._on_manual_box_cb = None
            self._canvas.configure(cursor="crosshair")
        self._redraw()

    def set_body_interactive_mode(self, active: bool, bboxes: list = None):
        """Bật/tắt chế độ xem khung cơ thể người."""
        self._interactive_body_mode = active
        if active:
            # Force single view on before_img
            self._split_mode = False
            self._ai_body_bboxes = bboxes or []
            self._excluded_body_indices = set()  # Reset khi quét lại
            self._canvas.configure(cursor="hand2")  # Con trỏ bàn tay để chỉ rõ có thể click
        else:
            self._ai_body_bboxes = []
            self._excluded_body_indices = set()
            self._canvas.configure(cursor="crosshair")
        self._redraw()

    def get_active_body_bboxes(self) -> list:
        """Trả về danh sách bboxes cơ thể chưa bị loại bỏ."""
        return [
            bbox for i, bbox in enumerate(self._ai_body_bboxes)
            if i not in self._excluded_body_indices
        ]

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
        if self._split_mode:
            # Split → After
            self._split_mode = False
            self._btn_split.configure(text="After ◨", fg_color="#1e3a1e", text_color="#3ecf8e")
        elif self._after_img is not None:
            # After → Split (nếu có cả 2 ảnh)
            self._split_mode = True
            self._btn_split.configure(text="Split ◧", fg_color="#23355a", text_color="#7ab3f7")
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

    def _get_viewport_crop(self, img: np.ndarray, cw: int, ch: int, split_x: int = None, is_after: bool = False):
        """
        Thuật toán Smart Crop: Chỉ render phần ảnh hiển thị trên màn hình.
        Trả về: (PIL Image, canvas_x, canvas_y)
        """
        ih, iw = img.shape[:2]
        sw = iw * self._zoom
        sh = ih * self._zoom

        cx = cw / 2 + self._pan_x
        cy = ch / 2 + self._pan_y

        left = cx - sw / 2
        top = cy - sh / 2
        right = cx + sw / 2
        bottom = cy + sh / 2

        # Vùng giới hạn bởi màn hình
        vis_left = max(0, left)
        vis_top = max(0, top)
        vis_right = min(cw, right)
        vis_bottom = min(ch, bottom)

        # Cắt thêm nếu đang ở chế độ Split
        if split_x is not None:
            if not is_after:
                vis_right = min(vis_right, split_x)
            else:
                vis_left = max(vis_left, split_x)

        if vis_left >= vis_right or vis_top >= vis_bottom:
            return None, 0, 0

        # Ánh xạ ngược tọa độ màn hình về tọa độ ảnh gốc
        src_left = int((vis_left - left) / self._zoom)
        src_top = int((vis_top - top) / self._zoom)
        src_right = int((vis_right - left) / self._zoom)
        src_bottom = int((vis_bottom - top) / self._zoom)

        # Chống tràn viền
        src_left = max(0, min(iw, src_left))
        src_right = max(0, min(iw, src_right))
        src_top = max(0, min(ih, src_top))
        src_bottom = max(0, min(ih, src_bottom))

        cropped = img[src_top:src_bottom, src_left:src_right]
        if cropped.size == 0:
            return None, 0, 0

        dst_w = int(vis_right - vis_left)
        dst_h = int(vis_bottom - vis_top)
        
        # Chọn interpolation tùy theo zoom
        interp = cv2.INTER_LANCZOS4 if self._zoom > 1.0 else cv2.INTER_AREA
        resized = cv2.resize(cropped, (dst_w, dst_h), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        return Image.fromarray(rgb), vis_left, vis_top


    def _draw_single(self, img: np.ndarray, cw: int, ch: int):
        pil, x, y = self._get_viewport_crop(img, cw, ch)
        if pil is None:
            return
            
        self._after_tk = ImageTk.PhotoImage(pil)
        self._canvas.create_image(x, y, anchor="nw", image=self._after_tk)
        self._lbl_info.configure(text=f"{img.shape[1]}×{img.shape[0]}px")
        
        if self._interactive_face_mode:
            # Vẽ bbox
            for bbox in self._ai_bboxes:
                x1, y1, x2, y2 = bbox
                cx1, cy1 = self._img_to_canvas(x1, y1)
                cx2, cy2 = self._img_to_canvas(x2, y2)
                # Chỉ vẽ nếu nằm trong vùng nhìn thấy
                if cx2 > 0 and cy2 > 0 and cx1 < cw and cy1 < ch:
                    self._canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#4f8ef7", width=2)
                    
            for bbox in self._manual_bboxes:
                x1, y1, x2, y2 = bbox
                cx1, cy1 = self._img_to_canvas(x1, y1)
                cx2, cy2 = self._img_to_canvas(x2, y2)
                if cx2 > 0 and cy2 > 0 and cx1 < cw and cy1 < ch:
                    self._canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00e676", width=2, dash=(4, 2))
                    
        if self._interactive_body_mode:
            # Vẽ body bbox
            for i, bbox in enumerate(self._ai_body_bboxes):
                x1, y1, x2, y2 = bbox
                cx1, cy1 = self._img_to_canvas(x1, y1)
                cx2, cy2 = self._img_to_canvas(x2, y2)
                if cx2 > 0 and cy2 > 0 and cx1 < cw and cy1 < ch:
                    if i in self._excluded_body_indices:
                        # Bbox bị bỏ qua: màu đỏ
                        self._canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#ff3333", width=2, dash=(4, 4))
                        self._canvas.create_text(cx1, cy1 - 10, text="Bỏ qua ✕", fill="#ff3333", font=("Inter", 10, "bold"), anchor="w")
                    else:
                        # Bbox hợp lệ: màu xanh lá
                        self._canvas.create_rectangle(cx1, cy1, cx2, cy2, outline="#00ff00", width=2, dash=(4, 4))
                        self._canvas.create_text(cx1, cy1 - 10, text="Người ✓", fill="#00ff00", font=("Inter", 10, "bold"), anchor="w")


    def _draw_split(self, cw: int, ch: int):
        split_x = int(cw * self._split_pos)

        # Before (Bên trái)
        pil_b, xb, yb = self._get_viewport_crop(self._before_img, cw, ch, split_x, is_after=False)
        if pil_b:
            self._before_tk = ImageTk.PhotoImage(pil_b)
            self._canvas.create_image(xb, yb, anchor="nw", image=self._before_tk)

        # After (Bên phải)
        pil_a, xa, ya = self._get_viewport_crop(self._after_img, cw, ch, split_x, is_after=True)
        if pil_a:
            self._after_tk = ImageTk.PhotoImage(pil_a)
            self._canvas.create_image(xa, ya, anchor="nw", image=self._after_tk)

        # Divider line
        self._canvas.create_line(split_x, 0, split_x, ch, fill="#4f8ef7", width=3)
        hx, hy = split_x, ch // 2
        self._canvas.create_oval(hx - 14, hy - 14, hx + 14, hy + 14, fill="#4f8ef7", outline="")
        self._canvas.create_line(hx - 6, hy, hx + 6, hy, fill="#ffffff", width=2)
        self._canvas.create_line(hx - 6, hy - 5, hx - 6, hy + 5, fill="#ffffff", width=2)
        self._canvas.create_line(hx + 6, hy - 5, hx + 6, hy + 5, fill="#ffffff", width=2)

        # Labels
        self._canvas.create_rectangle(split_x - 62, 8, split_x - 4, 28, fill="#111128", outline="", stipple="gray50")
        self._canvas.create_text(split_x - 33, 18, text="BEFORE", fill="#aac8ee", font=("Inter", 10, "bold"))
        self._canvas.create_rectangle(split_x + 4, 8, split_x + 60, 28, fill="#111128", outline="", stipple="gray50")
        self._canvas.create_text(split_x + 32, 18, text="AFTER", fill="#4f8ef7", font=("Inter", 10, "bold"))

        h, w = self._before_img.shape[:2]
        self._lbl_info.configure(text=f"{w}×{h}px")

    # ── Events ────────────────────────────────────────────────────────
    def _img_to_canvas(self, x, y):
        """Map tọa độ ảnh gốc sang Canvas."""
        cw = self._canvas.winfo_width() or 800
        ch = self._canvas.winfo_height() or 600
        if self._before_img is None: return 0, 0
        ih, iw = self._before_img.shape[:2]
        
        # Center of the image in canvas coordinates
        cx = cw / 2 + self._pan_x
        cy = ch / 2 + self._pan_y
        
        return cx + (x - iw / 2) * self._zoom, cy + (y - ih / 2) * self._zoom

    def _canvas_to_img(self, cx, cy):
        """Map tọa độ Canvas sang ảnh gốc."""
        cw = self._canvas.winfo_width() or 800
        ch = self._canvas.winfo_height() or 600
        if self._before_img is None: return 0, 0
        ih, iw = self._before_img.shape[:2]
        
        center_x = cw / 2 + self._pan_x
        center_y = ch / 2 + self._pan_y
        
        return (cx - center_x) / self._zoom + iw / 2, (cy - center_y) / self._zoom + ih / 2

    def _on_scroll(self, event):
        if event.num == 4 or event.delta > 0:
            self._zoom = min(self._zoom * 1.15, self._max_zoom)
        else:
            self._zoom = max(self._zoom / 1.15, self._min_zoom)
        self._redraw()

    def _on_press(self, event):
        # Space + drag = pan mode (Photoshop style), ưu tiên cao nhất
        if self._space_down:
            self._space_pan_start = (event.x - self._pan_x, event.y - self._pan_y)
            return
        
        if self._interactive_face_mode:
            # Vẽ hình chữ nhật mới
            self._drawing_box_start = (event.x, event.y)
            return
            
        if self._interactive_body_mode:
            # Click vào body bbox để toggle exclude/include
            img = self._before_img if self._before_img is not None else self._after_img
            if img is not None:
                for i, bbox in enumerate(self._ai_body_bboxes):
                    x1, y1, x2, y2 = bbox
                    cx1, cy1 = self._img_to_canvas(x1, y1)
                    cx2, cy2 = self._img_to_canvas(x2, y2)
                    if cx1 <= event.x <= cx2 and cy1 <= event.y <= cy2:
                        if i in self._excluded_body_indices:
                            self._excluded_body_indices.discard(i)
                        else:
                            self._excluded_body_indices.add(i)
                        self._redraw()
                        return
            return

        # Check nếu click gần divider — vùng bắt rộng 20px
        cw = self._canvas.winfo_width()
        split_x = int(cw * self._split_pos)
        if self._split_mode and abs(event.x - split_x) < 20:
            self._dragging_split = True
        else:
            self._drag_start = (event.x - self._pan_x, event.y - self._pan_y)
            self._dragging_split = False

    def _on_drag(self, event):
        # Space + drag = pan mọi lúc
        if self._space_down and self._space_pan_start:
            self._pan_x = event.x - self._space_pan_start[0]
            self._pan_y = event.y - self._space_pan_start[1]
            self._redraw()
            return
        
        if self._interactive_face_mode and self._drawing_box_start:
            if self._temp_box_id:
                self._canvas.delete(self._temp_box_id)
            sx, sy = self._drawing_box_start
            self._temp_box_id = self._canvas.create_rectangle(
                sx, sy, event.x, event.y, outline="#00e676", width=2, dash=(4, 2)
            )
            return

        if self._dragging_split:
            cw = self._canvas.winfo_width()
            self._split_pos = max(0.05, min(0.95, event.x / cw))
            self._redraw()
        elif self._drag_start:
            self._pan_x = event.x - self._drag_start[0]
            self._pan_y = event.y - self._drag_start[1]
            self._redraw()

    def _on_release(self, event):
        if self._interactive_face_mode and self._drawing_box_start:
            if self._temp_box_id:
                self._canvas.delete(self._temp_box_id)
                self._temp_box_id = None
            
            sx, sy = self._drawing_box_start
            ex, ey = event.x, event.y
            
            # Nếu khung quá bé thì bỏ qua (chống click nhầm)
            if abs(ex - sx) > 20 and abs(ey - sy) > 20:
                ix1, iy1 = self._canvas_to_img(min(sx, ex), min(sy, ey))
                ix2, iy2 = self._canvas_to_img(max(sx, ex), max(sy, ey))
                bbox = [ix1, iy1, ix2, iy2]
                
                # Gọi callback, nếu callback trả về True (thành công) thì lưu lại để vẽ
                if self._on_manual_box_cb:
                    success = self._on_manual_box_cb(bbox)
                    if success:
                        self._manual_bboxes.append(bbox)
                        self._redraw()
            
            self._drawing_box_start = None
            return

        self._drag_start = None
        self._space_pan_start = None
        self._dragging_split = False

    def _on_space_press(self, event):
        if not self._space_down:
            self._space_down = True
            self._canvas.configure(cursor="fleur")  # Con trỏ bàn tay mở

    def _on_space_release(self, event):
        self._space_down = False
        self._space_pan_start = None
        # Khôi phục con trỏ
        if self._interactive_face_mode:
            self._canvas.configure(cursor="crosshair")
        elif self._interactive_body_mode:
            self._canvas.configure(cursor="hand2")
        else:
            self._canvas.configure(cursor="crosshair")

    def _on_resize(self, event):
        self._redraw()

    def _on_mouse_move(self, event):
        if self._space_down:
            self._canvas.configure(cursor="fleur")
            return
        if self._interactive_face_mode:
            self._canvas.configure(cursor="crosshair")
            return
        if self._interactive_body_mode:
            self._canvas.configure(cursor="hand2")
            return
            
        cw = self._canvas.winfo_width()
        split_x = int(cw * self._split_pos)
        if self._split_mode and abs(event.x - split_x) < 20:
            self._canvas.configure(cursor="sb_h_double_arrow")
        else:
            self._canvas.configure(cursor="fleur" if self._drag_start else "crosshair")
