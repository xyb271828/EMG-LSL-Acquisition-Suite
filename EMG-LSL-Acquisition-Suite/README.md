
# EMG LSL Acquisition Suite

A reproducible experimental framework for **EMG acquisition via NI-DAQ streamed to LSL**
and **experimental marker annotation via MATLAB GUI**, synchronized using **LabRecorder**.

This repository is designed for research-grade EMG experiments where precise temporal
alignment between biosignals and task events is required.

---

## Repository Structure

- `acquisition/` – Python-based EMG acquisition (NI-DAQ → real-time plot → LSL stream)
- `markers/` – MATLAB GUI for task control and LSL marker streaming
- `docs/` – Usage notes and troubleshooting
- `configs/` – Parameter templates (device name, channels, sampling rate)
- `examples/` – Example workflows and placeholders
- `data/` – **Local data only** (`.xdf` recordings are not tracked by Git)
- `analysis/` – Post-processing and feature extraction (SI, RMS, MNF, MDF, etc.)

---

## Components

### 1. EMG Acquisition (Python)
**File:** `acquisition/daq_lsl_gui.py`

Key features (from source code):
- Supports NI devices commonly used in this project:
  - `Dev4`: 16 channels (ai0–7, ai16–23)
  - `Dev7`: 4 channels (ai0–3)
- GUI-based selection of device name and channel count
- All channels are streamed via LSL
- Real-time visualization of up to the first 8 channels

### 2. Marker GUI (MATLAB)
**File:** `markers/grip_marker_gui.m`

Key features:
- LSL marker stream name: `GripMarkers`
- GUI-controlled grip-force experiment
- Default parameters:
  - Number of trials: `40`
  - Grip duration: `2 s`
  - Rest duration: `2 s`
- Marker labels:
  - `EXP_START`, `EXP_END`
  - `REST_k`
  - `GRIP_START_k`
  - `GRIP_END_k`

---

## Environment Setup

### Python (Acquisition Side)

```bash
conda create -n emg_daq python=3.10 -y
conda activate emg_daq
pip install -r requirements.txt
```

> **Important:** NI-DAQmx drivers must be installed at the system level.
> Installing `nidaqmx` via pip alone is not sufficient.

### MATLAB (Marker Side)

Install and configure the **LSL MATLAB bindings** (`liblsl`).
Ensure the following functions are available in MATLAB:
- `lsl_loadlib`
- `lsl_streaminfo`
- `lsl_outlet`

---

## Recommended Recording Workflow

1. Connect NI hardware and EMG sensors  
2. Start the Python acquisition script:
   ```bash
   python acquisition/daq_lsl_gui.py
   ```
3. Start the MATLAB marker GUI:
   ```matlab
   grip_marker_gui
   ```
4. Open **LabRecorder**
   - Select both streams:
     - EMG stream (Python)
     - `GripMarkers` (MATLAB)
   - Start recording and save the `.xdf` file to `data/`
5. Perform offline analysis using scripts in `analysis/`

---

## File Naming Recommendation

```
YYYYMMDD_subjectID_sessionID.xdf
```

Consistent naming greatly simplifies downstream analysis.

---

## FAQ

**Q1. Streams do not appear in LabRecorder**
- Ensure Python/MATLAB scripts are running before opening LabRecorder
- Check firewall and network permissions (LSL uses local network discovery)
- Avoid running acquisition inside virtual machines if possible

**Q2. NI device not detected**
- Verify device name in NI MAX
- Confirm NI-DAQmx driver installation
- Check user permissions on the system

---

## License
MIT License
