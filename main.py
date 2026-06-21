"""
main.py — Downcharter+ GUI
"""
import tkinter as tk
from tkinter import filedialog
import threading
import os
import sys
import json

# Add the current folder to the path so the downcharter package can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from downcharter.processor import process_folder, revert_folder, find_midis, find_charts

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0a0a0a"
SURFACE = "#111111"
SURF2   = "#181818"
BORDER  = "#252525"
BORDER2 = "#333333"
RED     = "#D94040"
RED_DIM = "#7a2222"
FG      = "#e2e2e2"
FG2     = "#888888"
FG3     = "#444444"
GREEN   = "#4CAF74"
YELLOW  = "#C4A84A"
BLUE    = "#4A8AC4"
MONO    = "Courier New"


class StyledButton(tk.Canvas):
    def __init__(self, parent, text, command,
                 accent=False, danger=False, width=180, height=38):
        super().__init__(parent, height=height, width=width,
                         bg=SURFACE, highlightthickness=0, bd=0, cursor="hand2")
        self._text = text; self._cmd = command
        self._accent = accent; self._danger = danger
        self._on = True; self._hover = False
        self._draw()
        self.bind("<Enter>",    lambda _: self._sh(True))
        self.bind("<Leave>",    lambda _: self._sh(False))
        self.bind("<Button-1>", self._click)

    def _colors(self):
        if not self._on:     return BORDER,   FG3
        if self._hover:
            if self._accent: return "#c03535", FG
            if self._danger: return "#8b2020", FG
            return "#222222", FG
        if self._accent:     return RED,      FG
        if self._danger:     return RED_DIM,  FG2
        return SURF2, FG2

    def _draw(self):
        self.delete("all")
        bg, fg = self._colors()
        w, h = int(self["width"]), int(self["height"])
        self.create_rectangle(0, 0, w, h, fill=bg, outline=BORDER2, width=1)
        self.create_text(w//2, h//2, text=self._text,
                         font=(MONO, 10, "bold"), fill=fg, anchor="center")

    def _sh(self, v):  self._hover = v; self._draw()
    def _click(self, _):
        if self._on and self._cmd: self._cmd()

    def set_enabled(self, v):
        self._on = v; self["cursor"] = "hand2" if v else "arrow"; self._draw()


class CheckTile(tk.Canvas):
    def __init__(self, parent, text, variable, color=FG2, width=160, height=30):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0, cursor="hand2")
        self._text = text; self._var = variable; self._color = color
        self._enabled = True
        self._draw()
        variable.trace_add("write", lambda *_: self._draw())
        self.bind("<Button-1>", self._on_click)

    def _on_click(self, _):
        if not self._enabled:
            return
        self._var.set(not self._var.get())

    def set_enabled(self, v):
        self._enabled = bool(v)
        self.config(cursor="hand2" if v else "arrow")
        self._draw()

    def _draw(self):
        self.delete("all")
        checked = self._var.get()
        on = self._enabled
        bc = self._color if (checked and on) else BORDER2
        fc = FG if (checked and on) else FG3
        self.create_rectangle(2, 7, 18, 23, fill=bc, outline=bc)
        if checked:
            self.create_text(10, 15, text="✓", font=(MONO, 9, "bold"),
                             fill=BG, anchor="center")
        self.create_text(26, 15, text=self._text, font=(MONO, 9),
                         fill=fc, anchor="w")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Downcharter+")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._set_icon()
        self._set_dark_titlebar()
        cfg = self._load_settings()
        self._folder = cfg.get("folder", "") or ""
        self._threshold_ms   = tk.DoubleVar(value=cfg.get("threshold_ms", 125.0))
        self._do_expert_plus = tk.BooleanVar(value=cfg.get("expert_plus", True))
        self._do_hard        = tk.BooleanVar(value=cfg.get("hard", True))
        self._do_medium      = tk.BooleanVar(value=cfg.get("medium", True))
        self._do_easy        = tk.BooleanVar(value=cfg.get("easy", True))
        self._do_venue       = tk.BooleanVar(value=cfg.get("venue", True))
        self._do_hide_bg     = tk.BooleanVar(value=cfg.get("hide_bg", False))
        self._do_lipsync     = tk.BooleanVar(value=cfg.get("lipsync", False))
        # Persist whenever a toggle/slider changes
        for var in (self._threshold_ms, self._do_expert_plus, self._do_hard,
                    self._do_medium, self._do_easy, self._do_venue,
                    self._do_hide_bg, self._do_lipsync):
            var.trace_add("write", lambda *_: self._save_settings())
        self._build()
        if self._folder and os.path.isdir(self._folder):
            short = self._folder if len(self._folder) <= 52 else "…" + self._folder[-50:]
            self._folder_lbl.config(text=short, fg=FG2)
            self._btn_conv.set_enabled(True)
            self._btn_rev.set_enabled(True)
        else:
            self._folder = ""
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        self.geometry(f"+{(self.winfo_screenwidth()-w)//2}+{(self.winfo_screenheight()-h)//2}")

    def _build(self):
        # ── Header ──
        hdr = tk.Frame(self, bg=BG, padx=22, pady=14)
        hdr.pack(fill="x")
        tf = tk.Frame(hdr, bg=BG)
        tf.pack(side="left")
        tk.Label(tf, text="DOWN",    font=(MONO, 20, "bold"), fg=RED, bg=BG).pack(side="left")
        tk.Label(tf, text="CHARTER", font=(MONO, 20, "bold"), fg=FG,  bg=BG).pack(side="left")
        tk.Label(tf, text="+",       font=(MONO, 20, "bold"), fg=RED, bg=BG).pack(side="left")
        tk.Label(hdr, text="YARG · Rock Band · Clone Hero",
                 font=(MONO, 9), fg=FG3, bg=BG).pack(side="right", anchor="s", pady=3)

        tk.Frame(self, bg=RED,    height=1).pack(fill="x")
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG, padx=22, pady=16)
        body.pack(fill="both")

        # ── Folder ──
        self._lbl("SONGS FOLDER", body).pack(anchor="w")
        fr = tk.Frame(body, bg=BG)
        fr.pack(fill="x", pady=(5, 14))
        self._folder_lbl = tk.Label(fr, text="(none selected)",
                                    font=(MONO, 9), fg=FG3, bg=SURF2,
                                    anchor="w", padx=8, pady=6, width=46)
        self._folder_lbl.pack(side="left", fill="x", expand=True)
        StyledButton(fr, "  OPEN…", self._pick_folder, width=90, height=30).pack(
            side="right", padx=(8, 0))

        # ── Difficulties ──
        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        self._lbl("GENERATE DIFFICULTIES  (Guitar · Bass · Keys · Drums)", body).pack(anchor="w", pady=(0, 6))

        diff_row = tk.Frame(body, bg=BG)
        diff_row.pack(fill="x", pady=(0, 4))
        CheckTile(diff_row, "Hard",   self._do_hard,   color=YELLOW, width=100, height=28).pack(side="left", padx=(0, 8))
        CheckTile(diff_row, "Medium", self._do_medium, color=BLUE,   width=110, height=28).pack(side="left", padx=(0, 8))
        CheckTile(diff_row, "Easy",   self._do_easy,   color=GREEN,  width=100, height=28).pack(side="left")

        # ── Expert+ ──
        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(8, 10))
        CheckTile(body, "Expert+  (2× kick in PART DRUMS)",
                  self._do_expert_plus, color=RED, width=300, height=28).pack(anchor="w")

        thr_row = tk.Frame(body, bg=BG)
        thr_row.pack(fill="x", pady=(4, 2))
        self._nps_lbl = tk.Label(thr_row, text="8.0 notes/sec",
                                 font=(MONO, 13, "bold"), fg=FG, bg=BG, width=16, anchor="w")
        self._nps_lbl.pack(side="left")
        self._bpm_lbl = tk.Label(thr_row, text="= 240 BPM 16ths",
                                 font=(MONO, 9), fg=FG2, bg=BG)
        self._bpm_lbl.pack(side="left", padx=(4, 0))

        tk.Scale(body, variable=self._threshold_ms, from_=50, to=250, resolution=5,
                 orient="horizontal", showvalue=False,
                 bg=BG, fg=FG2, troughcolor=BORDER2, activebackground=RED,
                 highlightthickness=0, bd=0, sliderrelief="flat",
                 command=self._on_slider).pack(fill="x", pady=(2, 4))

        tk.Label(body, text="Doubles preserved  ·  3+ fast kicks → alternating = Expert+",
                 font=(MONO, 8), fg=FG3, bg=BG, anchor="w").pack(anchor="w", pady=(0, 10))

        # ── Venue ──
        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(8, 10))
        self._lbl("GENERATE VENUE  (camera · lights · post-proc)", body).pack(anchor="w", pady=(0, 6))
        CheckTile(body, "Venue",
                  self._do_venue, color=RED, width=360, height=28).pack(anchor="w", pady=(0, 6))
        # Hides background images in-game by renaming background.png/jpg →
        # background.bak.png/jpg (revert restores them).
        CheckTile(body, "Hide in-game background (image)",
                  self._do_hide_bg, color=RED, width=360, height=28).pack(anchor="w", pady=(0, 6))

        # ── Talkies ──
        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(8, 10))
        self._lbl("GENERATE TALKIES  (vocal stems recommended)", body).pack(anchor="w", pady=(0, 6))
        CheckTile(body, "Generate talkies from lyrics",
                  self._do_lipsync, color=RED, width=360, height=28).pack(anchor="w", pady=(0, 6))

        # ── Buttons ──
        tk.Frame(body, bg=BORDER, height=1).pack(fill="x", pady=(0, 12))
        btn_row = tk.Frame(body, bg=BG)
        btn_row.pack(fill="x", pady=(0, 14))
        self._btn_conv = StyledButton(btn_row, "⬡  PROCESS FOLDER",
                                      self._run_convert, accent=True, width=210, height=40)
        self._btn_conv.pack(side="left")
        self._btn_conv.set_enabled(False)
        self._btn_rev = StyledButton(btn_row, "↩  REVERT",
                                     self._run_revert, danger=True, width=120, height=40)
        self._btn_rev.pack(side="left", padx=(10, 0))
        self._btn_rev.set_enabled(False)

        # ── Log ──
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        lf = tk.Frame(self, bg=BG, padx=22, pady=12)
        lf.pack(fill="both", expand=True)
        self._log_box = tk.Text(lf, height=10, font=(MONO, 9),
                                bg=SURFACE, fg=FG2, insertbackground=FG,
                                relief="flat", bd=8, state="disabled", wrap="word",
                                selectbackground=BORDER2)
        self._log_box.pack(fill="both", expand=True)
        self._log_box.tag_config("ok",   foreground=GREEN)
        self._log_box.tag_config("err",  foreground=RED)
        self._log_box.tag_config("warn", foreground=YELLOW)
        self._log_box.tag_config("info", foreground=FG)
        self._log_box.tag_config("head", foreground=FG, font=(MONO, 9, "bold"))

        self._log("Downcharter+ — ready.\n\n", "info")

    def _settings_path(self):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, "Downcharter+", "settings.json")

    def _load_settings(self):
        try:
            with open(self._settings_path(), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_settings(self):
        data = {
            "folder":       self._folder,
            "threshold_ms": float(self._threshold_ms.get()),
            "expert_plus":  bool(self._do_expert_plus.get()),
            "hard":         bool(self._do_hard.get()),
            "medium":       bool(self._do_medium.get()),
            "easy":         bool(self._do_easy.get()),
            "venue":        bool(self._do_venue.get()),
            "hide_bg":      bool(self._do_hide_bg.get()),
            "lipsync":      bool(self._do_lipsync.get()),
        }
        try:
            path = self._settings_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _set_icon(self):
        base = os.path.dirname(os.path.abspath(__file__))
        ico = os.path.join(base, "assets", "downcharter.ico")
        try:
            if os.path.exists(ico):
                self.iconbitmap(ico)
        except Exception:
            pass
        # PNG via iconphoto (fallback / compatibility with some environments)
        try:
            png = os.path.join(base, "assets", "downcharter.png")
            if os.path.exists(png):
                self._icon_img = tk.PhotoImage(file=png)
                self.iconphoto(True, self._icon_img)
        except Exception:
            pass

    def _set_dark_titlebar(self):
        """Force a dark Windows title bar (DWM immersive dark mode)."""
        if sys.platform != "win32":
            return
        try:
            import ctypes
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            value = ctypes.c_int(1)
            # 20 = DWMWA_USE_IMMERSIVE_DARK_MODE (Win10 20H1+); 19 = older builds
            for attr in (20, 19):
                if ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        hwnd, attr, ctypes.byref(value),
                        ctypes.sizeof(value)) == 0:
                    break
        except Exception:
            pass

    def _lbl(self, t, p):
        return tk.Label(p, text=t, font=(MONO, 8, "bold"), fg=FG3, bg=BG)

    def _on_slider(self, _=None):
        ms  = float(self._threshold_ms.get())
        nps = 1000.0 / ms
        self._nps_lbl.config(text=f"{nps:.1f} notes/sec")
        self._bpm_lbl.config(text=f"= {nps*15:.0f} BPM 16ths")

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Charts folder")
        if not folder: return
        self._folder = folder
        self._save_settings()
        short = folder if len(folder) <= 52 else "…" + folder[-50:]
        self._folder_lbl.config(text=short, fg=FG2)
        self._btn_conv.set_enabled(True)
        self._btn_rev.set_enabled(True)
        n_mid = len(find_midis(folder))
        n_chart = len(find_charts(folder))
        extra = f" (+{n_chart} .chart)" if n_chart else ""
        self._log(f"Folder: {folder}\n  {n_mid} file(s){extra}\n\n", "info")

    def _run_convert(self):
        if not self._folder: return
        ms    = float(self._threshold_ms.get())
        xp    = self._do_expert_plus.get()
        venue = self._do_venue.get()
        hide_bg = self._do_hide_bg.get()
        lipsync = self._do_lipsync.get()
        diffs = [d for d, v in [("hard", self._do_hard),
                                  ("medium", self._do_medium),
                                  ("easy", self._do_easy)] if v.get()]
        if not xp and not diffs and not venue and not lipsync and not hide_bg:
            self._log("⚠  Nothing selected.\n", "warn"); return

        self._log("── PROCESS ──────────────────────────────\n", "head")
        if xp:    self._log(f"  Expert+: {1000/ms:.1f} nps\n")
        if diffs: self._log(f"  Diffs: {', '.join(diffs)}\n")
        if venue: self._log("  Venue: yes (theme from genre)\n")
        if hide_bg: self._log("  Hide background: yes (background.png/jpg → .bak)\n")
        if lipsync: self._log("  Talkies: yes (talky vocals charted from lyrics)\n")
        self._log("\n")
        self._btn_conv.set_enabled(False); self._btn_rev.set_enabled(False)

        def task():
            process_folder(self._folder, diffs, xp, ms, self._log, venue, lipsync,
                           do_hide_bg=hide_bg)
            self.after(0, lambda: self._btn_conv.set_enabled(True))
            self.after(0, lambda: self._btn_rev.set_enabled(True))
            self._log("\n")

        threading.Thread(target=task, daemon=True).start()

    def _run_revert(self):
        if not self._folder: return
        self._log("── REVERT ───────────────────────────────\n", "head")
        self._btn_conv.set_enabled(False); self._btn_rev.set_enabled(False)

        def task():
            revert_folder(self._folder, self._log)
            self.after(0, lambda: self._btn_conv.set_enabled(True))
            self.after(0, lambda: self._btn_rev.set_enabled(True))
            self._log("\n")

        threading.Thread(target=task, daemon=True).start()

    def _log(self, text, tag=None):
        def _w():
            self._log_box.config(state="normal")
            if tag: self._log_box.insert("end", text, tag)
            else:   self._log_box.insert("end", text)
            self._log_box.see("end")
            self._log_box.config(state="disabled")
        self.after(0, _w)


if __name__ == "__main__":
    App().mainloop()
