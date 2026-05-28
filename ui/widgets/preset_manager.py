"""
Preset Manager Widget — Lưu/Load/Xóa preset cài đặt.
"""
import customtkinter as ctk
import logging

logger = logging.getLogger(__name__)


class PresetManagerWidget(ctk.CTkFrame):
    """
    Widget nhỏ gọn để quản lý preset.
    Đặt ở phía dưới settings panel.
    """

    def __init__(self, master, on_load=None, on_save=None, **kwargs):
        super().__init__(master, fg_color="#0d0d1a", corner_radius=8, **kwargs)
        self._on_load = on_load or (lambda name: None)
        self._on_save = on_save or (lambda name: None)
        self._build_ui()
        self._refresh_list()

    def _build_ui(self):
        ctk.CTkLabel(self, text="📌  Preset", font=("Inter Bold", 12, "bold"),
                     text_color="#9C27B0").pack(padx=8, pady=(8, 4), anchor="w")

        # Dropdown
        self._preset_var = ctk.StringVar(value="")
        self._preset_menu = ctk.CTkOptionMenu(
            self, variable=self._preset_var, values=["(chưa có preset)"],
            font=("Inter", 11), command=self._on_select,
            fg_color="#1e1e3a",
        )
        self._preset_menu.pack(fill="x", padx=8, pady=2)

        # Buttons
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=4)

        btn_s = {"height": 28, "corner_radius": 6, "font": ("Inter", 11)}
        ctk.CTkButton(btn_row, text="Load", command=self._load, **btn_s,
                      fg_color="#1565C0").pack(side="left", padx=(0, 2), fill="x", expand=True)
        ctk.CTkButton(btn_row, text="Save As...", command=self._save_dialog, **btn_s,
                      fg_color="#2E7D32").pack(side="left", padx=2, fill="x", expand=True)
        ctk.CTkButton(btn_row, text="Xóa", command=self._delete, **btn_s,
                      fg_color="#8B1A1A").pack(side="right", padx=(2, 0), fill="x", expand=True)

    def _refresh_list(self):
        from utils.config import get_config
        presets = get_config().list_presets()
        if presets:
            self._preset_menu.configure(values=presets)
            self._preset_var.set(presets[0])
        else:
            self._preset_menu.configure(values=["(chưa có preset)"])
            self._preset_var.set("(chưa có preset)")

    def _on_select(self, val):
        pass  # sẽ load khi bấm Load

    def _load(self):
        name = self._preset_var.get()
        if name and not name.startswith("("):
            self._on_load(name)

    def _save_dialog(self):
        dialog = ctk.CTkInputDialog(text="Nhập tên preset:", title="Lưu Preset")
        name = dialog.get_input()
        if name and name.strip():
            self._on_save(name.strip())
            self._refresh_list()
            self._preset_var.set(name.strip())

    def _delete(self):
        name = self._preset_var.get()
        if name and not name.startswith("("):
            from utils.config import get_config
            get_config().delete_preset(name)
            self._refresh_list()
