#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EMG Acquisition + LSL Streaming + Real-time Monitor (PyQtGraph)

Code ownership:
  Lun Shu
  Koike Yasuharu Laboratory
  Institute of Science Tokyo

Supported NI devices (project-specific):
  - Dev4: 16 channels, ai0-ai7 and ai16-ai23
  - Dev7: 4 channels, ai0-ai3

Features:
  - Device input (editable) + total channel count input
  - Total channel max automatically constrained by selected device (Dev4=16, Dev7=4)
  - Start button starts NI-DAQ continuous acquisition
  - Streams ALL acquired channels via LSL
  - Displays ONLY the first 8 channels (or fewer if total < 8) in green
  - Fixed 5-second time axis (0..5 s); partial data is shown immediately (rest is blank)
  - Stop button stops acquisition and saves EMG.mat
  - Plot titles show actual physical channel names from NI-DAQ task (robust across devices)
"""

from __future__ import annotations

import sys
import time
import threading
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import savemat

import nidaqmx
from nidaqmx.constants import AcquisitionType

from pylsl import StreamInfo, StreamOutlet

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets


__version__ = "2.1.0"


# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class AppDefaults:
    sampling_rate_hz: int = 1000
    window_seconds: float = 5.0
    event_samples: int = 20

    display_max_channels: int = 8
    y_min: float = -1.0
    y_max: float = 1.0
    ui_fps: int = 20

    lsl_name: str = "Delsys"
    lsl_type: str = "EMG Signal"
    lsl_source_id: str = "EMG_Signal"

    save_path: str = "EMG.mat"


# -----------------------------
# Device-specific channel plan
# -----------------------------
DEVICE_AI_PLAN: Dict[str, List[str]] = {
    # Dev4: ai0..ai7 and ai16..ai23 (16ch)
    "Dev4": [f"ai{i}" for i in range(0, 8)] + [f"ai{i}" for i in range(16, 24)],
    # Dev7: ai0..ai3 (4ch)
    "Dev7": [f"ai{i}" for i in range(0, 4)],
}


def normalize_dev_name(name: str) -> str:
    """Normalize device string for lookup (e.g., 'dev4' -> 'Dev4')."""
    s = (name or "").strip()
    if not s:
        return s
    # Keep original if it already matches known keys
    if s in DEVICE_AI_PLAN:
        return s
    # Try case-insensitive match
    for k in DEVICE_AI_PLAN.keys():
        if s.lower() == k.lower():
            return k
    return s


def build_ai_list_for_device(dev_name: str, total_channels: int) -> List[str]:
    """
    Return the ai channel list for the given device and requested total_channels.
    If device is unknown, falls back to simple ai0..ai(N-1) (NOT recommended).
    """
    dev_key = normalize_dev_name(dev_name)

    if dev_key in DEVICE_AI_PLAN:
        full = DEVICE_AI_PLAN[dev_key]
        max_ch = len(full)
        if not (1 <= total_channels <= max_ch):
            raise ValueError(f"{dev_key} supports 1..{max_ch} channels, got {total_channels}.")
        return full[:total_channels]

    # Fallback for unknown devices: contiguous ai0..ai(N-1)
    # If you only use Dev4/Dev7, you will rarely hit this.
    if not (1 <= total_channels <= 32):
        raise ValueError("For unknown device, total_channels must be in [1, 32].")
    return [f"ai{i}" for i in range(total_channels)]


def max_channels_for_device(dev_name: str) -> int:
    dev_key = normalize_dev_name(dev_name)
    if dev_key in DEVICE_AI_PLAN:
        return len(DEVICE_AI_PLAN[dev_key])
    # Unknown device: allow up to 16 by default (conservative)
    return 16


# -----------------------------
# Main GUI application
# -----------------------------
class EMGMonitorWindow(QtWidgets.QMainWindow):
    def __init__(self, defaults: AppDefaults):
        super().__init__()
        self.defaults = defaults

        # Runtime state
        self._running: bool = False
        self._lock = threading.Lock()

        # Session parameters (set on Start)
        self._dev_name: Optional[str] = None
        self._fs: int = self.defaults.sampling_rate_hz
        self._total_channels: int = 0
        self._display_channels: int = 0
        self._ai_list: List[str] = []

        # Buffers (allocated on Start)
        self._window_samples: int = int(round(self.defaults.window_seconds * self._fs))
        self._x: np.ndarray = np.arange(self._window_samples, dtype=np.float32) / float(self._fs)

        self._ring: Optional[np.ndarray] = None  # (window_samples, display_channels)
        self._write_idx: int = 0
        self._filled: int = 0

        self._chunks: List[np.ndarray] = []

        # NI-DAQ + LSL
        self._task: Optional[nidaqmx.Task] = None
        self._outlet: Optional[StreamOutlet] = None
        self._has_push_chunk: bool = False

        # Physical channel names read from the NI-DAQ task (for titles)
        self._phys_names: List[str] = []

        # UI
        self._timer: Optional[QtCore.QTimer] = None
        self._plots = {}
        self._curves = {}
        self._graph: Optional[pg.GraphicsLayoutWidget] = None

        self._combo_dev: Optional[QtWidgets.QComboBox] = None
        self._spin_ch: Optional[QtWidgets.QSpinBox] = None
        self._btn_start: Optional[QtWidgets.QPushButton] = None
        self._btn_stop: Optional[QtWidgets.QPushButton] = None
        self._lbl_status: Optional[QtWidgets.QLabel] = None

        self._build_ui()
        self._populate_devices()
        self._setup_timer()

    # -----------------------------
    # UI setup
    # -----------------------------
    def _build_ui(self) -> None:
        self.setWindowTitle("EMG Monitor (DAQ -> LSL -> Plot)")

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        ctrl = QtWidgets.QGroupBox("Control")
        ctrl_layout = QtWidgets.QGridLayout(ctrl)
        layout.addWidget(ctrl)

        lbl_dev = QtWidgets.QLabel("Device (editable):")
        self._combo_dev = QtWidgets.QComboBox()
        self._combo_dev.setEditable(True)
        self._combo_dev.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self._combo_dev.currentTextChanged.connect(self._on_device_changed)

        lbl_ch = QtWidgets.QLabel("Total channels:")
        self._spin_ch = QtWidgets.QSpinBox()
        self._spin_ch.setRange(1, 16)
        self._spin_ch.setValue(16)

        self._btn_start = QtWidgets.QPushButton("Start")
        self._btn_start.clicked.connect(self.start_acquisition)

        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_stop.clicked.connect(self.stop_and_save)
        self._btn_stop.setEnabled(False)

        self._lbl_status = QtWidgets.QLabel("Status: idle")
        self._lbl_status.setWordWrap(True)

        ctrl_layout.addWidget(lbl_dev, 0, 0)
        ctrl_layout.addWidget(self._combo_dev, 0, 1, 1, 3)
        ctrl_layout.addWidget(lbl_ch, 1, 0)
        ctrl_layout.addWidget(self._spin_ch, 1, 1)
        ctrl_layout.addWidget(self._btn_start, 1, 2)
        ctrl_layout.addWidget(self._btn_stop, 1, 3)
        ctrl_layout.addWidget(self._lbl_status, 2, 0, 1, 4)

        self._graph = pg.GraphicsLayoutWidget()
        layout.addWidget(self._graph, stretch=1)

        pg.setConfigOptions(antialias=True)
        self.resize(980, 720)

    def _populate_devices(self) -> None:
        assert self._combo_dev is not None

        self._combo_dev.clear()
        # Prefer listing the known devices first (Dev4, Dev7)
        for k in DEVICE_AI_PLAN.keys():
            self._combo_dev.addItem(k)

        # Also query NI-DAQmx to show connected devices (nice-to-have)
        try:
            system = nidaqmx.system.System.local()
            devs = list(system.devices.device_names)
            for d in devs:
                if d not in DEVICE_AI_PLAN:
                    self._combo_dev.addItem(d)
            if devs:
                self._combo_dev.setCurrentText(devs[0])
            else:
                self._combo_dev.setCurrentText("Dev4")
        except Exception:
            self._combo_dev.setCurrentText("Dev4")

        # Apply device constraint to spinbox immediately
        self._on_device_changed(self._combo_dev.currentText())

    def _setup_timer(self) -> None:
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._update_plots)
        interval_ms = int(round(1000.0 / max(1, self.defaults.ui_fps)))
        self._timer.start(interval_ms)

    def _set_status(self, msg: str) -> None:
        assert self._lbl_status is not None
        self._lbl_status.setText(f"Status: {msg}")

    def _on_device_changed(self, dev_text: str) -> None:
        """Auto-adjust allowed channel range based on device."""
        if self._running:
            return
        assert self._spin_ch is not None

        dev_key = normalize_dev_name(dev_text)
        max_ch = max_channels_for_device(dev_key)

        self._spin_ch.setMaximum(max_ch)
        # If current value exceeds max, clamp it. Otherwise, if user hasn't changed, set a reasonable default.
        if self._spin_ch.value() > max_ch:
            self._spin_ch.setValue(max_ch)

        # Friendly status
        if dev_key in DEVICE_AI_PLAN:
            self._set_status(f"{dev_key} detected plan: max {max_ch} channels ({', '.join(DEVICE_AI_PLAN[dev_key])})")
        else:
            self._set_status(f"Unknown device '{dev_text}'. Max channels set to {max_ch} (fallback ai0..).")

    # -----------------------------
    # Plot setup per session
    # -----------------------------
    def _reset_plots(self) -> None:
        assert self._graph is not None
        self._graph.clear()
        self._plots.clear()
        self._curves.clear()

        green_pen = pg.mkPen("g", width=1)

        for ch in range(1, self._display_channels + 1):
            p = self._graph.addPlot()
            p.setXRange(0, self.defaults.window_seconds)
            p.setYRange(self.defaults.y_min, self.defaults.y_max)
            p.setLabel("bottom", "Time", units="s")
            p.showGrid(x=True, y=True, alpha=0.2)

            phys = self._phys_names[ch - 1] if (ch - 1) < len(self._phys_names) else "AI"
            p.setTitle(f"ch{ch} ({phys})")

            self._plots[ch] = p
            self._curves[ch] = p.plot([], [], pen=green_pen)

            if ch % 2 == 0:
                self._graph.nextRow()

    # -----------------------------
    # Start / Stop lifecycle
    # -----------------------------
    def start_acquisition(self) -> None:
        if self._running:
            return

        assert self._combo_dev is not None
        assert self._spin_ch is not None
        assert self._btn_start is not None
        assert self._btn_stop is not None

        dev_name = self._combo_dev.currentText().strip()
        if not dev_name:
            self._set_status("Please input a valid device name (e.g., Dev4 or Dev7).")
            return

        total_ch = int(self._spin_ch.value())

        try:
            ai_list = build_ai_list_for_device(dev_name, total_ch)
        except Exception as e:
            self._set_status(f"Invalid device/channel setting: {e}")
            return

        # Session parameters
        self._dev_name = normalize_dev_name(dev_name)
        self._total_channels = total_ch
        self._ai_list = ai_list
        self._display_channels = min(self.defaults.display_max_channels, self._total_channels)

        # Buffers
        self._fs = self.defaults.sampling_rate_hz
        self._window_samples = int(round(self.defaults.window_seconds * self._fs))
        self._x = np.arange(self._window_samples, dtype=np.float32) / float(self._fs)

        with self._lock:
            self._ring = np.zeros((self._window_samples, self._display_channels), dtype=np.float32)
            self._write_idx = 0
            self._filled = 0
            self._chunks = []

        # LSL outlet (streams all acquired channels)
        info = StreamInfo(
            name=self.defaults.lsl_name,
            type=self.defaults.lsl_type,
            channel_count=self._total_channels,
            nominal_srate=self._fs,
            channel_format="float32",
            source_id=self.defaults.lsl_source_id,
        )
        self._outlet = StreamOutlet(info)
        self._has_push_chunk = hasattr(self._outlet, "push_chunk")

        # Create NI-DAQ task
        try:
            self._task = nidaqmx.Task()

            self._add_ai_channels_by_ranges(self._task, self._dev_name, self._ai_list)

            # Read actual physical channel names from task (most reliable)
            self._phys_names = [ch.physical_channel.name for ch in self._task.ai_channels][: self._display_channels]

            # Build plots with correct titles
            self._reset_plots()

            self._task.timing.cfg_samp_clk_timing(
                rate=self._fs,
                sample_mode=AcquisitionType.CONTINUOUS,
            )
            self._task.register_every_n_samples_acquired_into_buffer_event(
                self.defaults.event_samples, self._daq_callback
            )

            self._running = True
            self._task.start()

        except Exception as e:
            self._running = False
            try:
                if self._task is not None:
                    self._task.close()
            except Exception:
                pass
            self._task = None
            self._set_status(f"Start failed: {e}")
            return

        # Lock UI
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._combo_dev.setEnabled(False)
        self._spin_ch.setEnabled(False)

        self._set_status(
            f"running | dev={self._dev_name} | fs={self._fs}Hz | total={self._total_channels}ch | "
            f"display={self._display_channels}ch | ai_list={','.join(self._ai_list)}"
        )

    def stop_and_save(self) -> None:
        if not self._running:
            return

        assert self._btn_start is not None
        assert self._btn_stop is not None
        assert self._combo_dev is not None
        assert self._spin_ch is not None

        self._set_status("stopping...")
        self._running = False

        try:
            if self._task is not None:
                self._task.stop()
                self._task.close()
        except Exception:
            pass
        finally:
            self._task = None

        with self._lock:
            if self._chunks:
                data = np.concatenate(self._chunks, axis=0).astype(np.float32, copy=False)
            else:
                data = np.zeros((0, self._total_channels), dtype=np.float32)

        meta = {
            "owner": "Lun Shu",
            "lab": "Koike Yasuharu Laboratory",
            "institute": "Institute of Science Tokyo",
            "device": str(self._dev_name),
            "ai_list": self._ai_list,
            "fs_hz": float(self._fs),
            "total_channels": float(self._total_channels),
            "display_channels": float(self._display_channels),
            "window_seconds": float(self.defaults.window_seconds),
            "physical_channels_displayed": self._phys_names,
            "created_unix_time": float(time.time()),
            "version": __version__,
        }

        try:
            savemat(self.defaults.save_path, {"EMG": data, "meta": meta})
            self._set_status(f"saved to {self.defaults.save_path} | shape={data.shape}")
        except Exception as e:
            self._set_status(f"save failed: {e}")

        # Unlock UI
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._combo_dev.setEnabled(True)
        self._spin_ch.setEnabled(True)

        # Apply device constraint again (in case user changes device)
        self._on_device_changed(self._combo_dev.currentText())

    def closeEvent(self, event) -> None:
        if self._running:
            self.stop_and_save()
        event.accept()

    # -----------------------------
    # NI-DAQ callback and helpers
    # -----------------------------
    def _daq_callback(self, task_handle, every_n_samples_event_type, number_of_samples, callback_data):
        if not self._running or self._task is None or self._outlet is None:
            return 0

        try:
            raw = self._task.read(number_of_samples_per_channel=number_of_samples)
            samples = np.asarray(raw, dtype=np.float32)  # (C_total, N)
            if samples.ndim != 2:
                return 0

            chunk_all = samples.T  # (N, C_total)

            if self._has_push_chunk:
                self._outlet.push_chunk(chunk_all.tolist())
            else:
                for i in range(chunk_all.shape[0]):
                    self._outlet.push_sample(chunk_all[i].tolist())

            with self._lock:
                self._chunks.append(chunk_all.copy())

            chunk_disp = chunk_all[:, : self._display_channels]
            self._update_ring(chunk_disp)

        except Exception:
            return 0

        return 0

    def _update_ring(self, chunk_disp: np.ndarray) -> None:
        if self._ring is None:
            return

        with self._lock:
            N = int(chunk_disp.shape[0])
            if N <= 0:
                return

            if N >= self._window_samples:
                chunk_disp = chunk_disp[-self._window_samples :, :]
                self._ring[:, :] = chunk_disp
                self._write_idx = 0
                self._filled = self._window_samples
                return

            end = self._write_idx + N
            if end <= self._window_samples:
                self._ring[self._write_idx:end, :] = chunk_disp
            else:
                first = self._window_samples - self._write_idx
                self._ring[self._write_idx:, :] = chunk_disp[:first, :]
                self._ring[: end % self._window_samples, :] = chunk_disp[first:, :]

            self._write_idx = (self._write_idx + N) % self._window_samples
            self._filled = min(self._window_samples, self._filled + N)

    def _get_display_block(self) -> np.ndarray:
        assert self._ring is not None

        with self._lock:
            block = np.full((self._window_samples, self._display_channels), np.nan, dtype=np.float32)

            if self._filled == 0:
                return block

            if self._filled < self._window_samples:
                block[: self._filled, :] = self._ring[: self._filled, :]
                return block

            older = self._ring[self._write_idx :, :]
            newer = self._ring[: self._write_idx, :]
            block[:, :] = np.vstack((older, newer))
            return block

    def _update_plots(self) -> None:
        if not self._running or self._ring is None:
            return

        block = self._get_display_block()
        x = self._x

        for ch in range(1, self._display_channels + 1):
            y = block[:, ch - 1]
            self._curves[ch].setData(x, y)

    @staticmethod
    def _add_ai_channels_by_ranges(task: nidaqmx.Task, dev: str, ai_list: List[str]) -> None:
        idxs = sorted(int(ch.replace("ai", "")) for ch in ai_list)

        ranges: List[Tuple[int, int]] = []
        start = idxs[0]
        prev = idxs[0]
        for v in idxs[1:]:
            if v == prev + 1:
                prev = v
            else:
                ranges.append((start, prev))
                start = v
                prev = v
        ranges.append((start, prev))

        for a, b in ranges:
            if a == b:
                task.ai_channels.add_ai_voltage_chan(f"{dev}/ai{a}")
            else:
                task.ai_channels.add_ai_voltage_chan(f"{dev}/ai{a}:{b}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    defaults = AppDefaults()
    app = QtWidgets.QApplication(sys.argv)
    win = EMGMonitorWindow(defaults)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
