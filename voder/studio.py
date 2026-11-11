"""Small native desktop workbench. Synthesis runs off the Tk UI thread."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .engine import BAND_CENTERS, Voice, synthesize
from .analysis import measure_audio
from .operator import DEMOS, EXAMPLES, LESSONS, audition_phone
from .phonemes import PHONES

BG = "#111b20"
PANEL = "#1b292f"
INK = "#e9e6db"
MUTED = "#9cacae"
ACCENT = "#e9ad66"


class Studio:
    def __init__(self, root):
        self.root = root
        self.root.title("Voder · Speech laboratory")
        self.root.geometry("1000x920")
        self.root.minsize(960, 850)
        self.root.configure(bg=BG)
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.utterance = None
        self.previous = None
        self.playing = None
        self.render_config = None
        self.pending_config = None
        self.pending_label = "Speech"
        self.pending_audition = False
        self.pending_playback = False
        self.operator_window = None
        self.tick_id = None
        self.play_started = None
        self.temp = tempfile.TemporaryDirectory(prefix="voder-")
        self.audio_path = Path(self.temp.name) / "preview.wav"
        self.previous_path = Path(self.temp.name) / "previous.wav"
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 11), padding=(14, 9))
        style.configure("TCheckbutton", background=BG, foreground=INK)
        shell = tk.Frame(root, bg=BG, padx=30, pady=22)
        shell.pack(fill="both", expand=True)
        tk.Label(shell, text="V O D E R  /  SPEECH LABORATORY", bg=BG, fg=ACCENT, font=("Consolas", 11)).pack(anchor="w")
        tk.Label(shell, text="Give electricity a voice.", bg=BG, fg=INK, font=("Segoe UI", 28, "bold")).pack(anchor="w", pady=(12, 4))
        tk.Label(shell, text="Text → phonemes → buzz + hiss → resonant filters → speech", bg=BG, fg=MUTED, font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 14))
        modes = tk.Frame(shell, bg=BG)
        modes.pack(fill="x", pady=(0, 12))
        tk.Label(modes, text="ENGINE", bg=BG, fg=MUTED, font=("Consolas", 10)).pack(side="left")
        self.mode = tk.StringVar(value="clear")
        ttk.Combobox(modes, textvariable=self.mode, values=("clear", "classic"), state="readonly", width=10).pack(side="left", padx=(10, 24))
        tk.Label(modes, text="PRESET", bg=BG, fg=MUTED, font=("Consolas", 10)).pack(side="left")
        self.preset = tk.StringVar(value="Neutral")
        presets = ttk.Combobox(modes, textvariable=self.preset, values=("Neutral", "Low & slow", "Bright", "Monotone"), state="readonly", width=14)
        presets.pack(side="left", padx=10)
        presets.bind("<<ComboboxSelected>>", self.apply_preset)
        self.autoplay = tk.BooleanVar(value=False)
        ttk.Checkbutton(modes, text="Play after rendering", variable=self.autoplay).pack(side="right")
        self.text = tk.Text(shell, height=4, wrap="word", font=("Segoe UI", 15), bg=PANEL, fg=INK,
                            insertbackground=ACCENT, relief="flat", padx=16, pady=14, undo=True)
        self.text.pack(fill="x")
        self.text.insert("1.0", "Hello world. I am the Voder. My voice is made of buzz and hiss.")
        self.raw = tk.BooleanVar()
        input_options = tk.Frame(shell, bg=BG)
        input_options.pack(fill="x", pady=8)
        ttk.Checkbutton(input_options, text="ARPABET input (e.g. HH AH0 L OW1)", variable=self.raw).pack(side="left")
        ttk.Button(input_options, text="Operator's card…", command=self.open_operator).pack(side="right")
        self.meter_view = tk.StringVar(value="Audio spectrum")
        meter_select = ttk.Combobox(input_options, textvariable=self.meter_view, values=("Audio spectrum", "Control targets"), state="readonly", width=16)
        meter_select.pack(side="right", padx=12)
        meter_select.bind("<<ComboboxSelected>>", lambda _: self.draw_meters())
        controls = tk.Frame(shell, bg=BG)
        controls.pack(fill="x", pady=8)
        self.pitch = self.slider(controls, "PITCH / Hz", 60, 300, 120, 1, 0)
        self.rate = self.slider(controls, "SPEED / ×", .5, 2, 1, .05, 1)
        self.expression = self.slider(controls, "EXPRESSION", 0, 2, 1, .05, 2)
        self.breath = self.slider(controls, "BREATH", 0, .3, .02, .01, 3)
        actions = tk.Frame(shell, bg=BG)
        actions.pack(fill="x", pady=(4, 14))
        self.render_button = ttk.Button(actions, text="Synthesize", command=self.start_render)
        self.render_button.pack(side="left")
        self.play_button = ttk.Button(actions, text="Play", command=self.play, state="disabled")
        self.play_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Stop", command=self.stop).pack(side="left")
        self.compare_button = ttk.Button(actions, text="Play previous", command=lambda: self.play(previous=True), state="disabled")
        self.compare_button.pack(side="left", padx=8)
        self.save_button = ttk.Button(actions, text="Export WAV…", command=self.save, state="disabled")
        self.save_button.pack(side="right")
        self.score_button = ttk.Button(actions, text="Score…", command=self.save_score, state="disabled")
        self.score_button.pack(side="right", padx=8)
        self.canvas = tk.Canvas(shell, height=155, bg=PANEL, highlightthickness=0)
        self.canvas.pack(fill="x")
        self.canvas.bind("<Configure>", lambda _: self.draw())
        self.meters = tk.Canvas(shell, height=92, bg=PANEL, highlightthickness=0)
        self.meters.pack(fill="x", pady=(6, 0))
        self.meters.bind("<Configure>", lambda _: self.draw_meters())
        self.status = tk.StringVar(value="Ready. Fully local synthesis · no recorded voices.")
        tk.Label(shell, textvariable=self.status, bg=BG, fg=ACCENT, font=("Segoe UI", 10), anchor="w", wraplength=870).pack(fill="x", pady=(12, 6))
        self.phones = tk.Text(shell, height=3, wrap="word", font=("Consolas", 10), bg=PANEL, fg=MUTED, relief="flat", padx=12, pady=8, state="disabled")
        self.phones.pack(fill="both", expand=True)
        tk.Label(shell, text="Inspired by the 1939 Voder. An experimental, mechanical English voice.", bg=BG, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(12, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Control-Return>", lambda _: self.start_render())
        self.root.bind("<Escape>", lambda _: self.stop())
        for var in (self.pitch, self.rate, self.expression, self.breath, self.mode, self.raw):
            var.trace_add("write", self.mark_dirty)
        self.text.edit_modified(False)
        self.text.bind("<<Modified>>", self.text_changed)
        self.tick_id = self.root.after(50, self.tick)

    def configuration(self):
        return (self.text.get("1.0", "end-1c"), self.raw.get(), self.pitch.get(), self.rate.get(),
                self.expression.get(), self.breath.get(), self.mode.get())

    def mark_dirty(self, *_):
        if self.future is None and self.render_config is not None and self.configuration() != self.render_config:
            self.status.set("Settings changed. Synthesize to update; playback and export still use the last render.")

    def text_changed(self, _):
        if self.text.edit_modified():
            self.mark_dirty()
            self.text.edit_modified(False)

    def apply_preset(self, _=None):
        values = {"Neutral": (120, 1, 1, .02), "Low & slow": (95, .85, .8, .035),
                  "Bright": (175, 1.05, 1.25, .01), "Monotone": (120, 1, 0, 0)}
        for var, value in zip((self.pitch, self.rate, self.expression, self.breath), values[self.preset.get()]):
            var.set(value)

    def slider(self, parent, label, low, high, value, step, column):
        frame = tk.Frame(parent, bg=BG)
        frame.grid(row=0, column=column, sticky="ew", padx=(0, 14))
        parent.columnconfigure(column, weight=1)
        tk.Label(frame, text=label, bg=BG, fg=MUTED, font=("Consolas", 10)).pack(anchor="w")
        var = tk.DoubleVar(value=value)
        tk.Scale(frame, from_=low, to=high, resolution=step, variable=var, orient="horizontal", bg=BG,
                 fg=INK, troughcolor=PANEL, activebackground=ACCENT, highlightthickness=0,
                 relief="flat", length=180).pack(fill="x")
        return var

    def start_render(self):
        if self.future is not None:
            return
        text = self.text.get("1.0", "end-1c")
        if not text.strip():
            self.status.set("Enter some text or phonemes first.")
            return
        self._start_job(synthesize, (text, self.current_voice()), {"phonemes": self.raw.get()}, self.configuration(), "Speech")

    def current_voice(self):
        return Voice(pitch=self.pitch.get(), rate=self.rate.get(), expression=self.expression.get(), breath=self.breath.get(), mode=self.mode.get())

    def _start_job(self, function, args, kwargs, config, label, *, audition=False):
        if self.future is not None:
            return
        self.stop()
        self.pending_config = config
        self.pending_label = label
        self.pending_audition = audition
        self.pending_playback = audition or self.autoplay.get()
        self.render_button.configure(state="disabled")
        self.play_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.score_button.configure(state="disabled")
        self.compare_button.configure(state="disabled")
        self.status.set("Synthesizing source, filter gestures, and waveform…")
        self.future = self.pool.submit(function, *args, **kwargs)

    def preview_phone(self, symbol):
        self._start_job(audition_phone, (symbol, self.current_voice()), {}, None,
                        f"Preview {symbol} ({EXAMPLES[symbol]})", audition=True)

    def load_demo(self, title):
        self.stop()
        self.text.delete("1.0", "end")
        self.text.insert("1.0", DEMOS[title])
        self.raw.set(False)
        self.status.set(f"Loaded: {title}. Click Synthesize to hear it with the current voice.")

    def open_operator(self):
        if self.operator_window is not None and self.operator_window.winfo_exists():
            self.operator_window.lift()
            return
        dialog = tk.Toplevel(self.root)
        self.operator_window = dialog
        dialog.title("Voder · Operator's card")
        dialog.configure(bg=BG)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        frame = tk.Frame(dialog, bg=BG, padx=22, pady=18)
        frame.pack(fill="both", expand=True)
        tk.Label(frame, text="Learn the voice, one sound at a time.", bg=BG, fg=INK, font=("Segoe UI", 18, "bold")).pack(anchor="w")
        tk.Label(frame, text="Click a sound for a short preview using the current engine and pitch.\nThe text editor stays intact. Stops keep their short closure and release.",
                 bg=BG, fg=MUTED, justify="left", font=("Segoe UI", 10)).pack(anchor="w", pady=(8, 12))
        groups = (("Vowels", ("vowel",)), ("Breath / friction", ("breath", "fricative")),
                  ("Nasals / glides", ("nasal", "liquid")), ("Stops / affricates", ("stop", "affricate")))
        for title, kinds in groups:
            tk.Label(frame, text=title.upper(), bg=BG, fg=ACCENT, font=("Consolas", 10)).pack(anchor="w", pady=(8, 4))
            row = tk.Frame(frame, bg=BG)
            row.pack(anchor="w")
            for i, (symbol, phone) in enumerate((s, p) for s, p in PHONES.items() if p.kind in kinds):
                ttk.Button(row, text=f"{symbol}\n{EXAMPLES[symbol]}", width=8,
                           command=lambda s=symbol: self.preview_phone(s)).grid(row=i // 8, column=i % 8, padx=2, pady=2)
        tk.Label(frame, text="DEMONSTRATION PHRASES / LOAD INTO EDITOR", bg=BG, fg=ACCENT, font=("Consolas", 10)).pack(anchor="w", pady=(14, 6))
        demo_row = tk.Frame(frame, bg=BG)
        demo_row.pack(fill="x")
        for title in DEMOS:
            ttk.Button(demo_row, text=title, command=lambda t=title: self.load_demo(t)).pack(side="left", padx=(0, 5))
        lesson = tk.StringVar(value=LESSONS[0][0])
        chooser = ttk.Combobox(frame, textvariable=lesson, values=[x[0] for x in LESSONS], state="readonly", width=22)
        chooser.pack(anchor="w", pady=(14, 6))
        lesson_body = tk.StringVar(value=LESSONS[0][1])
        chooser.bind("<<ComboboxSelected>>", lambda _: lesson_body.set(dict(LESSONS)[lesson.get()]))
        tk.Label(frame, textvariable=lesson_body, bg=PANEL, fg=INK, wraplength=740, justify="left", padx=12, pady=12,
                 font=("Segoe UI", 10)).pack(fill="x")
        tk.Label(frame, textvariable=self.status, bg=BG, fg=MUTED, wraplength=740, justify="left").pack(anchor="w", pady=(10, 4))
        ttk.Button(frame, text="Stop preview / Esc", command=self.stop).pack(anchor="e")
        def close_card():
            self.stop()
            dialog.destroy()
            self.operator_window = None
        dialog.protocol("WM_DELETE_WINDOW", close_card)
        dialog.bind("<Escape>", lambda _: self.stop())

    def tick(self):
        if self.future is not None and self.future.done():
            rendered = False
            try:
                result = self.future.result()
                staging = Path(self.temp.name) / "next.wav"
                result.save(staging)
                if self.utterance:
                    self.utterance.save(self.previous_path)
                    self.previous = self.utterance
                staging.replace(self.audio_path)
                self.utterance = result
                self.render_config = self.pending_config
                rendered = True
                self.phones.configure(state="normal")
                self.phones.delete("1.0", "end")
                self.phones.insert("1.0", self.utterance.phonemes)
                self.phones.configure(state="disabled")
                self.status.set(f"{self.pending_label} · {self.utterance.voice.mode.title()} · {self.utterance.duration:.2f} seconds · {len(self.utterance.gestures)} phonemes" +
                                (" · Approximate: " + ", ".join(self.utterance.unknown_words) if self.utterance.unknown_words else ""))
                self.draw()
                self.draw_meters()
            except Exception as exc:
                self.status.set(f"Synthesis failed: {exc}")
            finally:
                self.future = None
                self.render_button.configure(state="normal")
                if self.utterance:
                    self.save_button.configure(state="normal")
                    self.score_button.configure(state="normal")
                    self.play_button.configure(state="normal" if sys.platform == "win32" else "disabled")
                    self.compare_button.configure(state="normal" if self.previous and sys.platform == "win32" else "disabled")
                if rendered:
                    self.mark_dirty()
            if rendered and self.pending_playback and (self.pending_audition or self.configuration() == self.render_config):
                self.play()
            self.pending_audition = False
            self.pending_playback = False
        if self.play_started is not None and self.playing:
            elapsed = time.monotonic() - self.play_started
            if elapsed >= self.playing.duration:
                self.play_started = None
                self.playing = None
                self.canvas.delete("cursor")
                self.draw_meters()
            else:
                self.canvas.delete("cursor")
                if self.playing is self.utterance:
                    x = elapsed / self.utterance.duration * self.canvas.winfo_width()
                    self.canvas.create_line(x, 0, x, 155, fill=INK, tags="cursor")
                self.draw_meters(elapsed)
        self.tick_id = self.root.after(40, self.tick)

    def draw_meters(self, elapsed=None):
        self.meters.delete("all")
        active = self.playing or self.utterance
        if not active:
            self.meters.create_text(16, 24, anchor="w", text="Source and filter controls appear during playback.", fill=MUTED, font=("Consolas", 10))
            return
        g = next((g for g in active.gestures if elapsed is not None and g.start <= elapsed < g.start + g.duration), None)
        label = "Previous" if self.playing is self.previous and self.previous else "Current"
        detail = f"{label} score targets · {active.voice.mode.title()}"
        values = [0] * 10
        if g:
            t = min(1, max(0, (elapsed - g.start) / g.duration))
            detail += f" · {g.phone} · {g.pitch_start + t * (g.pitch_end - g.pitch_start):.0f} Hz · buzz {g.voiced:.2f} / hiss {g.noise:.2f}"
            if active.voice.mode == "clear":
                f = [a + t * (b - a) for a, b in zip(g.formants, g.formants_end)]
                detail += " · F1–3 " + "/".join(f"{v:.0f}" for v in f)
            if active.voice.mode == "classic" or g.kind in ("fricative", "stop", "affricate"):
                values = g.bands
        if self.meter_view.get() == "Audio spectrum":
            levels = measure_audio(active.samples, active.sample_rate, elapsed or 0)
            detail = f"{label} PCM · {g.phone if g else '—'} · RMS {levels.rms_dbfs:.1f} dBFS · peak {levels.peak_dbfs:.1f} dBFS · bands −60…0 dBFS"
            values = [np.clip((db + 60) / 60, 0, 1) for db in levels.bands_dbfs]
        self.meters.create_text(16, 14, anchor="w", text=detail, fill=INK, font=("Consolas", 9))
        width = max(1, (self.meters.winfo_width() - 32) / 10)
        for i, (freq, gain) in enumerate(zip(BAND_CENTERS, values)):
            x = 16 + i * width
            self.meters.create_rectangle(x, 36, x + width - 8, 60, fill=BG, outline="")
            self.meters.create_rectangle(x, 60 - float(gain) * 24, x + width - 8, 60, fill=ACCENT, outline="")
            self.meters.create_text(x + width / 2 - 4, 75, text=f"{freq:g}", fill=MUTED, font=("Consolas", 8))

    def draw(self):
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 1)
        if not self.utterance:
            self.canvas.create_text(22, 76, anchor="w", text="Your synthesized waveform will appear here.", fill=MUTED, font=("Consolas", 11))
            return
        samples = self.utterance.samples
        for x, chunk in enumerate(np.array_split(samples, min(width, len(samples)))):
            peak = float(np.max(np.abs(chunk), initial=0))
            self.canvas.create_line(x, 62 - peak * 47, x, 62 + peak * 47, fill=ACCENT)
        for g in self.utterance.gestures:
            x = g.start / self.utterance.duration * width
            if g.duration / self.utterance.duration * width > 18:
                self.canvas.create_text(x + 3, 132, anchor="w", text=g.phone, fill=MUTED, font=("Consolas", 8))

    def play(self, previous=False):
        target = self.previous if previous else self.utterance
        if target and sys.platform == "win32":
            import winsound
            try:
                winsound.PlaySound(str(self.previous_path if previous else self.audio_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
                self.playing = target
                self.play_started = time.monotonic()
            except RuntimeError as exc:
                self.status.set(f"Playback unavailable: {exc}. You can still export WAV.")

    def stop(self):
        self.pending_audition = False
        self.pending_playback = False
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(None, 0)
        self.play_started = None
        self.playing = None
        self.canvas.delete("cursor")
        self.draw_meters()

    def save_score(self):
        if not self.utterance:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Control score", "*.json")], initialfile="voder-score.json")
        if path:
            try:
                Path(path).write_text(json.dumps(self.utterance.score(), indent=2), encoding="utf-8")
                self.status.set(f"Saved {path}")
            except OSError as exc:
                messagebox.showerror("Could not save score", str(exc))

    def save(self):
        if not self.utterance:
            return
        path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("Wave audio", "*.wav")], initialfile="voder.wav")
        if path:
            try:
                self.utterance.save(path)
                self.status.set(f"Saved {path}")
            except OSError as exc:
                messagebox.showerror("Could not save audio", str(exc))

    def close(self):
        self.stop()
        if self.tick_id:
            self.root.after_cancel(self.tick_id)
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.temp.cleanup()
        self.root.destroy()


def main():
    root = tk.Tk()
    Studio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
