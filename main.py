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


# ── song.ini field definitions ───────────────────────────────────────────────
# Union of the YARG Wiki song.ini reference and the Clone Hero song.ini guide.
# Grouped as (section title, [(tag, hint, type)]) where type is "str" | "int" |
# "bool". The creator window renders one input per field and exports a clean
# [song] block with only the tags the user actually filled in (no comments).
SONGINI_GROUPS = [
    ("Core  (name is mandatory; the rest are strongly recommended)", [
        ("name", "The title of the song.", "str"),
        ("artist", "The artist / band.", "str"),
        ("album", "Album the song is from. Single? write the title then (Single).", "str"),
        ("genre", "Broad genre (see the common genre names list).", "str"),
        ("sub_genre", "More specific genre (YARG).", "str"),
        ("year", "Year of the song's (or album's) first release.", "str"),
        ("charter", "Who created the playable note track (you).", "str"),
        ("song_length", "Audio length in MILLISECONDS (215000 is 3:35).", "int"),
    ]),
    ("Album / playlist / preview", [
        ("album_track", "Track number on the album (default 16000).", "int"),
        ("playlist", "Name of the playlist this chart belongs to.", "str"),
        ("playlist_track", "Track number within the playlist.", "int"),
        ("icon", "Source icon (yarg, yargdlc, yarn, or a game/setlist id).", "str"),
        ("preview_start_time", "Preview start in MILLISECONDS (85064 is 1:25.064).", "int"),
        ("preview_end_time", "Preview end in MILLISECONDS.", "int"),
        ("loading_phrase", "Flavor text shown on the difficulty/instrument screen.", "str"),
        ("rating", "Age rating: 1 FF, 2 SR, 3 MC, 4 NR, 5 SC (YARG).", "int"),
        ("location", "Band origin, e.g. Example City, Example Country (YARG).", "str"),
    ]),
    ("Difficulties / intensities  (0-6, or -1 if the part is absent)", [
        ("diff_band", "Overall band intensity.", "int"),
        ("diff_guitar", "5-Fret Lead Guitar intensity.", "int"),
        ("diff_guitar_coop", "5-Fret Co-op (Melody) Guitar intensity.", "int"),
        ("diff_rhythm", "5-Fret Rhythm Guitar intensity.", "int"),
        ("diff_bass", "5-Fret Bass intensity.", "int"),
        ("diff_drums", "4-Lane Drums intensity.", "int"),
        ("diff_drums_real", "Pro Drums intensity.", "int"),
        ("diff_keys", "5-Lane Keys intensity.", "int"),
        ("diff_vocals", "Vocals intensity.", "int"),
    ]),
    ("6-Fret (GHL) difficulties  (only if the chart has GHL tracks)", [
        ("diff_guitarghl", "6-Fret Lead Guitar intensity.", "int"),
        ("diff_guitar_coop_ghl", "6-Fret Co-op Guitar intensity.", "int"),
        ("diff_rhythm_ghl", "6-Fret Rhythm Guitar intensity.", "int"),
        ("diff_bassghl", "6-Fret Bass intensity.", "int"),
    ]),
    ("Format flags & timing", [
        ("pro_drums", "Pro Drums present (when no tom notes to auto-detect).", "bool"),
        ("five_lane_drums", "Drums track is in 5-Lane format.", "bool"),
        ("modchart", "Mark this chart as a modchart (for sorting).", "bool"),
        ("end_events", "Tolerate the chart's end events.", "bool"),
        ("video_start_time", "Background-video start in MS (negative delays it).", "int"),
        ("vocal_scroll_speed", "Vocal track scroll speed, default 100 (YARG).", "int"),
        ("delay", "Legacy audio realignment in MS (deprecated).", "int"),
    ]),
    ("Credits  (optional)", [
        ("credit_written_by", "Who wrote the song.", "str"),
        ("credit_performed_by", "Who performed the song.", "str"),
        ("credit_composed_by", "Who composed the song.", "str"),
        ("credit_produced_by", "Who produced the song.", "str"),
        ("credit_album_art_by", "Who made the album art.", "str"),
        ("credit_license", "License the song was released under.", "str"),
    ]),
    ("Links  (optional)", [
        ("link_youtube", "Link to the song on YouTube.", "str"),
        ("link_spotify", "Link to the song on Spotify.", "str"),
        ("link_bandcamp", "Link to the song on Bandcamp.", "str"),
        ("link_soundcloud", "Link to the song on SoundCloud.", "str"),
    ]),
]


def build_songini_text(values: dict) -> str:
    """Clean [song] block with only the filled-in tags (no comments).

    `values` maps tag -> string ("" / "True" / "False" handled per type)."""
    lines = ["[song]"]
    for _title, fields in SONGINI_GROUPS:
        for tag, _hint, typ in fields:
            v = values.get(tag, "")
            if typ == "bool":
                if v == "True":
                    lines.append(f"{tag} = True")
            else:
                v = (v or "").strip()
                if v:
                    lines.append(f"{tag} = {v}")
    lines.append("")
    return "\n".join(lines)


class StyledButton(tk.Canvas):
    def __init__(self, parent, text, command,
                 accent=False, danger=False, color=None, width=180, height=38):
        super().__init__(parent, height=height, width=width,
                         bg=SURFACE, highlightthickness=0, bd=0, cursor="hand2")
        self._text = text; self._cmd = command
        self._accent = accent; self._danger = danger; self._color = color
        self._on = True; self._hover = False
        self._draw()
        self.bind("<Enter>",    lambda _: self._sh(True))
        self.bind("<Leave>",    lambda _: self._sh(False))
        self.bind("<Button-1>", self._click)

    @staticmethod
    def _shade(hexcol, factor):
        hexcol = hexcol.lstrip("#")
        r, g, b = (int(hexcol[i:i+2], 16) for i in (0, 2, 4))
        r, g, b = (min(255, int(c * factor)) for c in (r, g, b))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _colors(self):
        if not self._on:     return BORDER,   FG3
        if self._color:
            return (self._shade(self._color, 1.18) if self._hover else self._color), FG
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


class TabButton(tk.Canvas):
    """A flat tab header that underlines in RED when its tab is active."""
    def __init__(self, parent, text, command, app, width=110, height=30):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0, cursor="hand2")
        self._text = text; self._cmd = command; self._app = app
        self._active = False; self._hover = False
        self._draw()
        self.bind("<Enter>",    lambda _: self._sh(True))
        self.bind("<Leave>",    lambda _: self._sh(False))
        self.bind("<Button-1>", lambda _: self._cmd())

    def set_active(self, v):
        self._active = bool(v); self._draw()

    def _sh(self, v):
        self._hover = v; self._draw()

    def _draw(self):
        self.delete("all")
        w, h = int(self["width"]), int(self["height"])
        fg = FG if (self._active or self._hover) else FG2
        self.create_text(w//2, h//2 - 2, text=self._text,
                         font=(MONO, 10, "bold"), fill=fg, anchor="center")
        if self._active:
            self.create_rectangle(0, h-2, w, h, fill=RED, outline=RED)


class RadioTile(tk.Canvas):
    """A single-choice tile bound to a StringVar (radio behaviour)."""
    def __init__(self, parent, text, variable, value, color=RED, width=120, height=30):
        super().__init__(parent, width=width, height=height,
                         bg=BG, highlightthickness=0, bd=0, cursor="hand2")
        self._text = text; self._var = variable; self._value = value
        self._color = color
        self._draw()
        variable.trace_add("write", lambda *_: self._draw())
        self.bind("<Button-1>", lambda _: self._var.set(self._value))

    def _draw(self):
        self.delete("all")
        on = self._var.get() == self._value
        bc = self._color if on else BORDER2
        self.create_oval(3, 8, 17, 22, outline=bc, width=2,
                         fill=bc if on else BG)
        if on:
            self.create_oval(7, 12, 13, 18, fill=BG, outline=BG)
        self.create_text(26, 15, text=self._text, font=(MONO, 9),
                         fill=FG if on else FG3, anchor="w")


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
        self._do_export_venue = tk.BooleanVar(value=cfg.get("export_venue", False))  # hidden (no GUI widget, settings.json only)
        self._do_drum_anim   = tk.BooleanVar(value=cfg.get("drum_anim", True))
        self._do_hide_bg     = tk.BooleanVar(value=cfg.get("hide_bg", False))
        self._do_lipsync     = tk.BooleanVar(value=cfg.get("lipsync", True))       # talkies
        self._do_lipsync_trk = tk.BooleanVar(value=cfg.get("lipsync_track", True))   # LIPSYNC1 viseme track
        self._do_vocal_sep   = tk.BooleanVar(value=cfg.get("vocal_sep", True))
        self._pp_style       = tk.StringVar(value=cfg.get("pp_style", "authored"))  # hidden (no GUI widget, settings.json only)
        # ── Convert tab (native PS3 package generation) ──
        self._conv_folder = cfg.get("conv_folder", "") or ""
        self._conv_out = cfg.get("conv_out", "") or ""
        self._conv_pedal  = tk.StringVar(value=cfg.get("conv_pedal", "both"))  # 1x | 2x | both
        self._do_sng_preserve_dirs = tk.BooleanVar(
            value=cfg.get("sng_preserve_dirs", True))
        self._cancel = threading.Event()
        # Persist whenever a toggle/slider changes
        for var in (self._threshold_ms, self._do_expert_plus, self._do_hard,
                    self._do_medium, self._do_easy, self._do_venue, self._do_export_venue,
                    self._do_drum_anim,
                    self._do_hide_bg, self._do_lipsync, self._do_lipsync_trk,
                    self._do_vocal_sep,
                    self._pp_style,
                    self._conv_pedal, self._do_sng_preserve_dirs):
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
        self.geometry(f"{w}x{h}+{(self.winfo_screenwidth()-w)//2}+{(self.winfo_screenheight()-h)//2}")

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

        # ── Tab bar ──
        self._tabs: dict[str, tk.Frame] = {}
        self._tab_btns: dict[str, "TabButton"] = {}
        tabbar = tk.Frame(self, bg=BG, padx=22, pady=0)
        tabbar.pack(fill="x", pady=(10, 0))
        TabButton(tabbar, "PROCESS", lambda: self._show_tab("process"),
                  self).pack(side="left")
        self._tab_btns["process"] = tabbar.winfo_children()[-1]
        TabButton(tabbar, "CONVERT", lambda: self._show_tab("convert"),
                  self).pack(side="left", padx=(6, 0))
        self._tab_btns["convert"] = tabbar.winfo_children()[-1]

        # song.ini creator — opens a pre-filled template the user fills + saves.
        StyledButton(tabbar, "NEW song.ini", self._open_songini_creator,
                     width=150, height=28).pack(side="right")

        # ── Tab container (only one child shown at a time) ──
        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True)

        # ── Process tab ──────────────────────────────────────────────────────
        body = tk.Frame(container, bg=BG)
        self._tabs["process"] = body

        # Controls region (top, fixed height)
        p_ctrl = tk.Frame(body, bg=BG, padx=22, pady=16)
        p_ctrl.pack(fill="x")

        self._lbl("SONGS FOLDER", p_ctrl).pack(anchor="w")
        fr = tk.Frame(p_ctrl, bg=BG)
        fr.pack(fill="x", pady=(5, 14))
        self._folder_lbl = tk.Label(fr, text="(none selected)",
                                    font=(MONO, 9), fg=FG3, bg=SURF2,
                                    anchor="w", padx=8, pady=6, width=46)
        self._folder_lbl.pack(side="left", fill="x", expand=True)
        StyledButton(fr, "  OPEN…", self._pick_folder, width=90, height=30).pack(
            side="right", padx=(8, 0))

        tk.Frame(p_ctrl, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        self._lbl("GENERATE DIFFICULTIES  (Guitar · Bass · Keys · Drums)", p_ctrl).pack(anchor="w", pady=(0, 6))
        diff_row = tk.Frame(p_ctrl, bg=BG)
        diff_row.pack(fill="x", pady=(0, 4))
        CheckTile(diff_row, "Hard",   self._do_hard,   color=YELLOW, width=100, height=28).pack(side="left", padx=(0, 8))
        CheckTile(diff_row, "Medium", self._do_medium, color=BLUE,   width=110, height=28).pack(side="left", padx=(0, 8))
        CheckTile(diff_row, "Easy",   self._do_easy,   color=GREEN,  width=100, height=28).pack(side="left")

        tk.Frame(p_ctrl, bg=BORDER, height=1).pack(fill="x", pady=(8, 10))
        CheckTile(p_ctrl, "Expert+",
                  self._do_expert_plus, color=RED, width=300, height=28).pack(anchor="w")
        thr_row = tk.Frame(p_ctrl, bg=BG)
        thr_row.pack(fill="x", pady=(4, 2))
        self._nps_lbl = tk.Label(thr_row, text="8.0 notes/sec",
                                 font=(MONO, 13, "bold"), fg=FG, bg=BG, width=16, anchor="w")
        self._nps_lbl.pack(side="left")
        self._bpm_lbl = tk.Label(thr_row, text="= 240 BPM 16ths",
                                 font=(MONO, 9), fg=FG2, bg=BG)
        self._bpm_lbl.pack(side="left", padx=(4, 0))
        tk.Scale(p_ctrl, variable=self._threshold_ms, from_=50, to=250, resolution=5,
                 orient="horizontal", showvalue=False,
                 bg=BG, fg=FG2, troughcolor=BORDER2, activebackground=RED,
                 highlightthickness=0, bd=0, sliderrelief="flat",
                 command=self._on_slider).pack(fill="x", pady=(2, 4))
        tk.Label(p_ctrl, text="Doubles preserved  ·  3+ fast kicks → alternating = Expert+",
                 font=(MONO, 8), fg=FG3, bg=BG, anchor="w").pack(anchor="w", pady=(0, 10))

        tk.Frame(p_ctrl, bg=BORDER, height=1).pack(fill="x", pady=(8, 10))
        self._lbl("GENERATE VENUE  (camera · lights · post-proc)", p_ctrl).pack(anchor="w", pady=(0, 6))
        CheckTile(p_ctrl, "Venue",   self._do_venue, color=RED, width=360, height=28).pack(anchor="w", pady=(0, 6))
        CheckTile(p_ctrl, "Drum animations",  self._do_drum_anim, color=RED, width=360, height=28).pack(anchor="w", pady=(0, 6))
        CheckTile(p_ctrl, "Lipsync  (vocal stem recommended)",
                  self._do_lipsync_trk, color=RED, width=360, height=28).pack(anchor="w", pady=(0, 6))

        tk.Frame(p_ctrl, bg=BORDER, height=1).pack(fill="x", pady=(8, 10))
        self._lbl("GENERATE TALKIES  (vocal stem recommended)", p_ctrl).pack(anchor="w", pady=(0, 6))
        CheckTile(p_ctrl, "Generate talkies from lyrics",
                  self._do_lipsync, color=RED, width=360, height=28).pack(anchor="w", pady=(0, 6))

        tk.Frame(p_ctrl, bg=BORDER, height=1).pack(fill="x", pady=(8, 10))
        self._lbl("EXTRAS", p_ctrl).pack(anchor="w", pady=(0, 6))
        CheckTile(p_ctrl, "Hide in-game background (image only)",
                  self._do_hide_bg, color=RED, width=360, height=28).pack(anchor="w", pady=(0, 6))
        CheckTile(p_ctrl, "Vocal separation  (MDX-NET)  (recommended for lipsync and talkies)",
                  self._do_vocal_sep, color=RED, width=520, height=28).pack(anchor="w", pady=(0, 6))

        tk.Frame(p_ctrl, bg=BORDER, height=1).pack(fill="x", pady=(0, 12))
        btn_row = tk.Frame(p_ctrl, bg=BG)
        btn_row.pack(fill="x", pady=(0, 14))
        self._btn_conv = StyledButton(btn_row, "⬡  PROCESS FOLDER",
                                      self._run_convert, accent=True, width=210, height=40)
        self._btn_conv.pack(side="left")
        self._btn_conv.set_enabled(False)
        self._btn_rev = StyledButton(btn_row, "↩  REVERT",
                                     self._run_revert, danger=True, width=120, height=40)
        self._btn_rev.pack(side="left", padx=(10, 0))
        self._btn_rev.set_enabled(False)
        self._btn_cancel = StyledButton(btn_row, "✕  CANCEL",
                                        self._cancel_op, danger=True, width=120, height=40)

        # Process log + status
        self._build_tab_log(body, "process")

        # ── Convert tab ──
        conv = tk.Frame(container, bg=BG)
        self._tabs["convert"] = conv
        self._build_convert_tab(conv)

        # Show the default tab
        self._show_tab("process")

    def _build_tab_log(self, parent: tk.Frame, which: str):
        """Add a log box + status bar inside `parent` (a tab frame).
        `which` is \"process\" or \"convert\" — selects `self._*_log` / `self._*_status`."""
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x")
        lf = tk.Frame(parent, bg=BG, padx=22, pady=12)
        lf.pack(fill="both", expand=True)
        log_box = tk.Text(lf, height=8, font=(MONO, 9),
                          bg=SURFACE, fg=FG2, insertbackground=FG,
                          relief="flat", bd=8, state="disabled", wrap="word",
                          selectbackground=BORDER2)
        log_box.pack(fill="both", expand=True)
        for tag, fg in (("ok", GREEN), ("err", RED), ("warn", YELLOW),
                        ("info", FG), ("head", FG)):
            log_box.tag_config(tag, foreground=fg,
                               font=(MONO, 9, "bold") if tag == "head" else (MONO, 9))
        setattr(self, f"_{which}_log", log_box)

        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x")
        sf = tk.Frame(parent, bg=SURFACE, padx=22)
        sf.pack(fill="x", pady=3)
        sv = tk.StringVar(value="")
        setattr(self, f"_{which}_status", sv)
        tk.Label(sf, textvariable=sv,
                 font=(MONO, 9, "bold"), fg=RED, bg=SURFACE,
                 anchor="w").pack(side="left", fill="x", expand=True)

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
            "export_venue": bool(self._do_export_venue.get()),
            "drum_anim":    bool(self._do_drum_anim.get()),
            "hide_bg":      bool(self._do_hide_bg.get()),
            "lipsync":      bool(self._do_lipsync.get()),
            "lipsync_track": bool(self._do_lipsync_trk.get()),
            "vocal_sep":     bool(self._do_vocal_sep.get()),
            "pp_style":      self._pp_style.get(),
            "conv_folder":        self._conv_folder,
            "conv_out":           self._conv_out,
            "conv_pedal":         self._conv_pedal.get(),
            "sng_preserve_dirs":  bool(self._do_sng_preserve_dirs.get()),
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

    def _set_dark_titlebar(self, win=None):
        """Force a dark Windows title bar (DWM immersive dark mode)."""
        if sys.platform != "win32":
            return
        win = win or self
        try:
            import ctypes
            win.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(win.winfo_id())
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
        self._plog(f"Folder: {folder}\n  {n_mid} file(s){extra}\n\n", "info")

    def _run_convert(self):
        if not self._folder: return
        ms    = float(self._threshold_ms.get())
        xp    = self._do_expert_plus.get()
        venue = self._do_venue.get()
        drum_anim = self._do_drum_anim.get()
        hide_bg = self._do_hide_bg.get()
        lipsync = self._do_lipsync.get()
        lipsync_trk = self._do_lipsync_trk.get()
        vocal_sep = self._do_vocal_sep.get()
        pp_style = self._pp_style.get()
        diffs = [d for d, v in [("hard", self._do_hard),
                                  ("medium", self._do_medium),
                                  ("easy", self._do_easy)] if v.get()]
        if (not xp and not diffs and not venue and not lipsync
                and not lipsync_trk and not hide_bg and not drum_anim
                and not vocal_sep):
            self._plog("⚠  Nothing selected.\n", "warn"); return

        self._plog("── PROCESS ──────────────────────────────\n", "head")
        if xp:    self._plog(f"  Expert+: {1000/ms:.1f} nps\n")
        if diffs: self._plog(f"  Diffs: {', '.join(diffs)}\n")
        if venue: self._plog("  Venue: yes (theme from genre)\n")
        if drum_anim: self._plog("  Drum animations: yes (drummer limbs)\n")
        if hide_bg: self._plog("  Hide background: yes (background.png/jpg → .bak)\n")
        if lipsync_trk: self._plog("  Lipsync: yes (LIPSYNC1 viseme track from lyrics)\n")
        if lipsync: self._plog("  Talkies: yes (talky vocals charted from lyrics)\n")
        if vocal_sep: self._plog("  Vocal sep: yes (MDX-NET)\n")
        else: self._plog("  Vocal sep: no\n")
        self._plog("\n")
        self._btn_conv.set_enabled(False); self._btn_rev.set_enabled(False)
        self._cancel.clear()
        self._btn_cancel.pack(side="right")
        self._btn_cancel.set_enabled(True)

        def task():
            try:
                process_folder(self._folder, diffs, xp, ms, self._plog, venue, lipsync_trk,
                               do_hide_bg=hide_bg, do_talkies=lipsync,
                               do_drum_anim=drum_anim,
                               export_venue=self._do_export_venue.get(),
                               pp_style=pp_style,
                               do_vocal_sep=vocal_sep,
                               cancel=self._cancel,
                               status_fn=lambda t: self._pstatus(f"Processing:  {t}"),
                               done_fn=lambda ok, err, tot: self._pstatus(
                                   f"Processed:  {ok}/{tot}" +
                                   (f"  ({err} errored)" if err else "")))
            except Exception as e:
                self._plog(f"  ✗ Fatal: {e}\n", "err")
                import traceback
                self._plog(traceback.format_exc(), "err")
            finally:
                # self._status("")  # keep last status visible until next operation
                self.after(0, lambda: self._btn_conv.set_enabled(True))
                self.after(0, lambda: self._btn_rev.set_enabled(True))
                self.after(0, lambda: self._btn_cancel.pack_forget())
                self.after(0, self._cancel.clear)
                self._plog("\n")

        threading.Thread(target=task, daemon=True).start()

    def _cancel_op(self):
        """Cancel the current process or revert operation."""
        self._cancel.set()
        self._btn_cancel.pack_forget()
        self._plog("\n  ⚡ Cancelling… (will stop after the current song)\n", "warn")

    def _cancel_conv(self):
        """Cancel the current convert operation."""
        self._cancel.set()
        self._btn_cancel_conv.pack_forget()
        self._clog("\n  ⚡ Cancelling… (will stop after the current song)\n", "warn")

    def _run_revert(self):
        if not self._folder: return
        self._plog("── REVERT ───────────────────────────────\n", "head")
        self._btn_conv.set_enabled(False); self._btn_rev.set_enabled(False)
        self._cancel.clear()
        self._btn_cancel.pack(side="right")
        self._btn_cancel.set_enabled(True)

        def task():
            try:
                revert_folder(self._folder, self._plog, cancel=self._cancel,
                              status_fn=lambda t: self._pstatus(f"Reverting:  {t}"),
                              done_fn=lambda n: self._pstatus(
                                  f"Reverted:  {n} file(s)"))
            except Exception as e:
                self._plog(f"  ✗ Fatal: {e}\n", "err")
                import traceback
                self._plog(traceback.format_exc(), "err")
            finally:
                # self._status("")  # keep last status visible until next operation
                self.after(0, lambda: self._btn_conv.set_enabled(True))
                self.after(0, lambda: self._btn_rev.set_enabled(True))
                self.after(0, lambda: self._btn_cancel.pack_forget())
                self.after(0, self._cancel.clear)
                self._plog("\n")

        threading.Thread(target=task, daemon=True).start()

    # ── Tabs ───────────────────────────────────────────────────────────────
    def _show_tab(self, name: str):
        for n, frame in self._tabs.items():
            if n == name:
                frame.pack(fill="both")
            else:
                frame.pack_forget()
        for n, btn in self._tab_btns.items():
            btn.set_active(n == name)

    # ── Convert tab ────────────────────────────────────────────────────────
    def _build_convert_tab(self, body):
        # Controls region (top, fixed height)
        c_ctrl = tk.Frame(body, bg=BG, padx=22, pady=16)
        c_ctrl.pack(fill="x")

        self._lbl("SOURCE SONG FOLDER", c_ctrl).pack(anchor="w")
        fr = tk.Frame(c_ctrl, bg=BG)
        fr.pack(fill="x", pady=(5, 14))
        self._conv_folder_lbl = tk.Label(fr, text="(none selected)",
                                         font=(MONO, 9), fg=FG3, bg=SURF2,
                                         anchor="w", padx=8, pady=6, width=46)
        self._conv_folder_lbl.pack(side="left", fill="x", expand=True)
        StyledButton(fr, "  OPEN…", self._pick_conv_folder, width=90, height=30).pack(
            side="right", padx=(8, 0))

        tk.Frame(c_ctrl, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        self._lbl("OUTPUT FOLDER", c_ctrl).pack(anchor="w")
        ofr = tk.Frame(c_ctrl, bg=BG)
        ofr.pack(fill="x", pady=(5, 14))
        self._conv_out_lbl = tk.Label(ofr, text="(default: beside source)",
                                      font=(MONO, 9), fg=FG3, bg=SURF2,
                                      anchor="w", padx=8, width=46)
        self._conv_out_lbl.pack(side="left", fill="x", expand=True)
        btn_fr = tk.Frame(ofr, bg=BG)
        btn_fr.pack(side="right", padx=(8, 0))
        StyledButton(btn_fr, "  SET…", self._pick_conv_out, width=90, height=30).pack(
            side="left", padx=(0, 8))
        StyledButton(btn_fr, "  ✕", self._clear_conv_out, width=34, height=30).pack(
            side="left")
        if self._conv_out and os.path.isdir(self._conv_out):
            short = self._conv_out if len(self._conv_out) <= 52 \
                else "…" + self._conv_out[-50:]
            self._conv_out_lbl.config(text=short, fg=FG2)

        # ── ROCK BAND section (PS3 folder + Xbox CON — bass pedal applies) ──
        tk.Frame(c_ctrl, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        tk.Label(c_ctrl, text="ROCK BAND  (PS3 / Xbox 360)",
                 font=(MONO, 10, "bold"), fg=RED, bg=BG, anchor="w").pack(
                     anchor="w", pady=(0, 8))
        self._lbl("BASS PEDAL", c_ctrl).pack(
            anchor="w", pady=(0, 6))
        ped_row = tk.Frame(c_ctrl, bg=BG)
        ped_row.pack(fill="x", pady=(0, 4))
        RadioTile(ped_row, "1× pedal", self._conv_pedal, "1x",
                  width=110, height=28).pack(side="left", padx=(0, 8))
        RadioTile(ped_row, "2× pedal", self._conv_pedal, "2x",
                  width=110, height=28).pack(side="left", padx=(0, 8))
        RadioTile(ped_row, "Both", self._conv_pedal, "both",
                  width=90, height=28).pack(side="left")
        tk.Label(c_ctrl, text="1× removes Expert+ doubles  ·  2× forces doubles to always play",
                 font=(MONO, 8), fg=FG3, bg=BG, anchor="w").pack(anchor="w", pady=(2, 10))

        self._btn_ps3 = StyledButton(c_ctrl, "⬢  BUILD RPCS3 FOLDER",
                                     lambda: self._run_native_convert("ps3"),
                                     color=BLUE, width=220, height=40)
        self._btn_ps3.pack(anchor="w", pady=(0, 8))
        self._btn_con = StyledButton(c_ctrl, "⬢  BUILD CON",
                                     lambda: self._run_native_convert("xbox"),
                                     color=GREEN, width=220, height=40)
        self._btn_con.pack(anchor="w", pady=(0, 14))

        # ── SNG section (YARG / Clone Hero — verbatim repackage of the folder) ──
        tk.Frame(c_ctrl, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))
        tk.Label(c_ctrl, text="YARG / CLONE HERO",
                 font=(MONO, 10, "bold"), fg=RED, bg=BG, anchor="w").pack(
                     anchor="w", pady=(0, 6))
        tk.Label(c_ctrl, text="Packs the song folder as-is into a .sng container",
                 font=(MONO, 8), fg=FG3, bg=BG, anchor="w",
                 justify="left", wraplength=520).pack(anchor="w", pady=(0, 8))
        self._btn_sng = StyledButton(c_ctrl, "⬢  BUILD SNG",
                                     lambda: self._run_native_convert("sng"),
                                     color=RED, width=220, height=40)
        self._btn_sng.pack(anchor="w")
        CheckTile(c_ctrl, "Preserve folder structure",
                  self._do_sng_preserve_dirs, color=RED, width=220, height=28).pack(
                      anchor="w", pady=(4, 14))

        self._conv_btns = (self._btn_ps3, self._btn_con, self._btn_sng)
        self._btn_cancel_conv = StyledButton(c_ctrl, "✕  CANCEL CONVERT",
                                              self._cancel_conv, danger=True,
                                              width=200, height=40)
        # hidden until convert starts
        for b in self._conv_btns:
            b.set_enabled(bool(self._conv_folder and
                               os.path.isdir(self._conv_folder)))
        if self._conv_folder and os.path.isdir(self._conv_folder):
            short = self._conv_folder if len(self._conv_folder) <= 52 \
                else "…" + self._conv_folder[-50:]
            self._conv_folder_lbl.config(text=short, fg=FG2)

        # Convert log + status
        self._build_tab_log(body, "convert")

    def _pick_conv_folder(self):
        folder = filedialog.askdirectory(title="Source song folder")
        if not folder:
            return
        self._conv_folder = folder
        self._save_settings()
        short = folder if len(folder) <= 52 else "…" + folder[-50:]
        self._conv_folder_lbl.config(text=short, fg=FG2)
        for b in self._conv_btns:
            b.set_enabled(True)
        self._clog(f"Convert source: {folder}\n\n", "info")

    def _pick_conv_out(self):
        folder = filedialog.askdirectory(title="Output folder")
        if not folder:
            return
        self._conv_out = folder
        self._save_settings()
        short = folder if len(folder) <= 52 else "…" + folder[-50:]
        self._conv_out_lbl.config(text=short, fg=FG2)
        self._clog(f"Convert output: {folder}\n\n", "info")

    def _clear_conv_out(self):
        self._conv_out = ""
        self._save_settings()
        self._conv_out_lbl.config(text="(default: beside source)", fg=FG3)

    def _run_native_convert(self, fmt):
        if not self._conv_folder:
            return
        pedal = self._conv_pedal.get()
        out_base = self._conv_out or None
        fmt_label = {"ps3": "PS3 folder", "xbox": "Xbox CON",
                     "sng": "SNG"}.get(fmt, fmt)
        from downcharter.ps3build import build_ps3_song, source_has_double_kicks
        from downcharter.stfs import build_con_song
        from downcharter.sng import build_sng_song, _sanitize_path_component
        self._cancel.clear()
        self._clog(f"── CONVERT ({fmt_label}) ─────────────────────\n", "head")
        self._clog(f"  Source: {self._conv_folder}\n")
        self._clog(f"  Output: {out_base or '(beside source)'}\n")
        for b in self._conv_btns:
            b.set_enabled(False)
        self._btn_cancel_conv.pack(anchor="w", pady=(6, 0))
        self._btn_cancel_conv.set_enabled(True)

        total_songs = 0
        conv_ok = 0
        conv_err = 0
        # Track output paths to detect collisions (same output .sng path from different sources)
        sng_output_seen: dict[str, str] = {}

        def task():
            nonlocal total_songs, conv_ok, conv_err
            try:
                # The source may be a single song folder OR a parent holding many
                # song subfolders — convert every song under it. A "song folder" is
                # the directory of each notes.mid or notes.chart (same rule the Process tab uses).
                # Combine both to catch songs that have only .chart files.
                mid_dirs = {os.path.dirname(m) for m in find_midis(self._conv_folder)}
                chart_dirs = {os.path.dirname(c) for c in find_charts(self._conv_folder)}
                song_dirs = sorted(mid_dirs | chart_dirs)
                if not song_dirs:
                    song_dirs = [self._conv_folder]
                self._clog(f"  Songs: {len(song_dirs)}\n\n")

                def _label(sd):
                    return os.path.basename(sd.rstrip("/\\")) or sd

                total_songs = len(song_dirs)
                for si, sd in enumerate(song_dirs):
                    if self._cancel.is_set():
                        self._clog("  ⚡ Cancelled by user.\n", "warn")
                        break

                    self._cstatus(f"Converting:  {si+1}/{total_songs}  {_label(sd)}")

                    if fmt == "sng":
                        self._clog(f"  ▸ SNG: {_label(sd)}\n", "head")
                        try:
                            if self._do_sng_preserve_dirs.get():
                                # Preserve folder structure from conv_folder to the song.
                                # E.g., conv_folder="E:\Songs\Chorus", song at "E:\Songs\Chorus\Sub\SongA"
                                # with out_base="Y:\Output\Chorus" -> "Y:\Output\Chorus\Sub\SongA.sng"
                                try:
                                    # relpath from conv_folder (NOT its parent) to preserve internal structure
                                    rel = os.path.relpath(sd, self._conv_folder)
                                    rel_dir = os.path.dirname(rel)
                                    if out_base:
                                        sng_out = os.path.join(out_base, rel_dir) if rel_dir else out_base
                                    else:
                                        # No out_base: write alongside source, preserving internal structure
                                        # (the song's folder, NOT the parent of conv_folder)
                                        sng_out = os.path.join(self._conv_folder, rel_dir) if rel_dir else os.path.dirname(sd)
                                except ValueError:
                                    # Different drives - fall back to just the song folder
                                    sng_out = os.path.dirname(sd)
                            else:
                                sng_out = out_base
                            # Compute the final output path for collision detection
                            folder_name = os.path.basename(os.path.abspath(sd)) or "song"
                            safe_name = _sanitize_path_component(folder_name)
                            final_out_sng = os.path.join(sng_out, safe_name + ".sng")
                            # Check for collision: same output path from different source
                            if final_out_sng in sng_output_seen:
                                prev_src = sng_output_seen[final_out_sng]
                                self._clog(
                                    f"  ! Collision: {_label(sd)} would overwrite {_label(prev_src)}\n"
                                    f"    (both output to {os.path.basename(final_out_sng)})\n", "warn")
                                # Skip to avoid data loss
                                conv_err += 1
                                continue
                            sng_output_seen[final_out_sng] = sd
                            build_sng_song(sd, self._clog, out_base=sng_out)
                            conv_ok += 1
                        except Exception as e:
                            self._clog(f"  ✗ {_label(sd)}: {e}\n", "err")
                            conv_err += 1
                        continue

                    builder = (build_ps3_song if fmt == "ps3"
                               else build_con_song)
                    has_dk = source_has_double_kicks(sd)
                    self._clog(f"  ▸ {_label(sd)}\n", "head")
                    try:
                        if pedal == "both":
                            if has_dk:
                                modes = ["1x", "2x"]
                            else:
                                modes = ["1x"]
                                self._clog("    (no double-kicks → 1x only)\n", "info")
                        else:
                            modes = [pedal]
                        for mode in modes:
                            if self._cancel.is_set():
                                break
                            self._clog(f"    {fmt_label} ({mode})\n", "info")
                            builder(sd, mode, self._clog, out_base=out_base)
                        conv_ok += 1
                    except Exception as e:
                        import traceback
                        self._clog(f"  ✗ {_label(sd)}: {e}\n", "err")
                        self._clog(traceback.format_exc(), "err")
                        conv_err += 1
            except Exception as e:
                import traceback
                self._clog(f"  ✗ {e}\n", "err")
                self._clog(traceback.format_exc(), "err")
            finally:
                self.after(0, lambda: [b.set_enabled(True) for b in self._conv_btns])
                self.after(0, lambda: self._btn_cancel_conv.pack_forget())
                self.after(0, self._cancel.clear)
                self.after(0, lambda: self._cstatus(
                    f"Converted:  {conv_ok}/{total_songs}" +
                    (f"  ({conv_err} errored)" if conv_err else "")))
                self._clog("\n")

        threading.Thread(target=task, daemon=True).start()

    # ── song.ini creator ───────────────────────────────────────────────────
    def _open_songini_creator(self):
        if getattr(self, "_ini_win", None) and self._ini_win.winfo_exists():
            self._ini_win.lift()
            self._ini_win.focus_force()
            return

        win = tk.Toplevel(self)
        self._ini_win = win
        win.title("New song.ini")
        win.configure(bg=BG)
        win.geometry("680x720")
        win.minsize(560, 480)
        self._set_dark_titlebar(win)

        head = tk.Frame(win, bg=BG, padx=18, pady=12)
        head.pack(fill="x")
        tk.Label(head, text="song.ini  CREATOR", font=(MONO, 13, "bold"),
                 fg=RED, bg=BG).pack(side="left")
        tk.Label(head, text="fill what you need · empty fields are skipped on export",
                 font=(MONO, 8), fg=FG3, bg=BG).pack(side="left", padx=(10, 0), anchor="s",
                                                     pady=3)
        tk.Frame(win, bg=BORDER, height=1).pack(fill="x")

        # ── Scrollable form ──
        outer = tk.Frame(win, bg=BG)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0, bd=0)
        sb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        form = tk.Frame(canvas, bg=BG, padx=18, pady=10)
        win_id = canvas.create_window((0, 0), window=form, anchor="nw")
        form.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win_id, width=e.width))

        def _wheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _wheel)
        win.bind("<Destroy>",
                 lambda e: e.widget is win and canvas.unbind_all("<MouseWheel>"))

        self._ini_fields = {}        # tag -> ("str"/"int", Entry) | ("bool", Var)
        for title, fields in SONGINI_GROUPS:
            tk.Label(form, text=title, font=(MONO, 9, "bold"), fg=RED, bg=BG,
                     anchor="w").pack(fill="x", pady=(12, 2))
            tk.Frame(form, bg=BORDER, height=1).pack(fill="x", pady=(0, 6))
            for tag, hint, typ in fields:
                row = tk.Frame(form, bg=BG)
                row.pack(fill="x", pady=(0, 1))
                tk.Label(row, text=tag, font=(MONO, 9), fg=FG, bg=BG,
                         anchor="w", width=22).pack(side="left")
                if typ == "bool":
                    var = tk.BooleanVar(value=False)
                    CheckTile(row, "True", var, color=GREEN,
                              width=80, height=24).pack(side="left")
                    self._ini_fields[tag] = ("bool", var)
                else:
                    ent = tk.Entry(row, bg=SURF2, fg=FG, insertbackground=FG,
                                   font=(MONO, 9), bd=0, highlightthickness=1,
                                   highlightbackground=BORDER,
                                   highlightcolor=BORDER2)
                    ent.pack(side="left", fill="x", expand=True)
                    self._ini_fields[tag] = (typ, ent)
                tk.Label(form, text=hint, font=(MONO, 8), fg=FG3, bg=BG,
                         anchor="w").pack(fill="x", padx=(22, 0), pady=(0, 4))

        tk.Frame(win, bg=BORDER, height=1).pack(fill="x")
        btn_row = tk.Frame(win, bg=BG, padx=18, pady=12)
        btn_row.pack(fill="x")

        def collect():
            vals = {}
            for tag, (typ, w) in self._ini_fields.items():
                vals[tag] = "True" if typ == "bool" and w.get() else (
                    "" if typ == "bool" else w.get())
            return vals

        def _ini_log(text, tag=None):
            """Log to whichever tab is active (or process by default)."""
            if getattr(self, "_tabs", None):
                active = next((n for n, f in self._tabs.items()
                               if f.winfo_ismapped()), "process")
            else:
                active = "process"
            (self._clog if active == "convert" else self._plog)(text, tag)

        def do_save():
            vals = collect()
            if not (vals.get("name") or "").strip():
                _ini_log("  ✗ song.ini needs at least a 'name'.\n", "err")
                return
            path = filedialog.asksaveasfilename(
                title="Save song.ini", defaultextension=".ini",
                initialfile="song.ini",
                filetypes=[("Song info", "*.ini"), ("All files", "*.*")])
            if not path:
                return
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(build_songini_text(vals))
            except OSError as e:
                _ini_log(f"  ✗ song.ini save failed: {e}\n", "err")
                return
            _ini_log(f"song.ini saved: {path}\n\n", "info")
            win.destroy()

        StyledButton(btn_row, "💾  EXPORT song.ini", do_save, accent=True,
                     width=190, height=36).pack(side="left")
        StyledButton(btn_row, "  CLOSE", win.destroy,
                     width=110, height=36).pack(side="right")

    def _pstatus(self, text):
        """Update the Process tab status bar (thread-safe)."""
        self.after(0, lambda: self._process_status.set(text or ""))

    def _cstatus(self, text):
        """Update the Convert tab status bar (thread-safe)."""
        self.after(0, lambda: self._convert_status.set(text or ""))

    def _plog(self, text, tag=None):
        """Append to the Process tab log box (thread-safe)."""
        self.after(0, lambda: self._tab_log(self._process_log, text, tag))

    def _clog(self, text, tag=None):
        """Append to the Convert tab log box (thread-safe)."""
        self.after(0, lambda: self._tab_log(self._convert_log, text, tag))

    _LOG_MAX_LINES = 250      # truncate when exceeded
    _LOG_KEEP_LINES = 200     # lines to keep after truncation

    @staticmethod
    def _tab_log(log_box, text, tag=None):
        log_box.config(state="normal")
        if tag: log_box.insert("end", text, tag)
        else:   log_box.insert("end", text)
        # Truncate if too many lines accumulate (keeps memory bounded)
        total_lines = int(log_box.index("end-1c").split(".")[0])
        if total_lines > App._LOG_MAX_LINES:
            keep_from = f"{total_lines - App._LOG_KEEP_LINES + 1}.0"
            log_box.delete("1.0", keep_from)
        log_box.see("end")
        log_box.config(state="disabled")


if __name__ == "__main__":
    App().mainloop()
