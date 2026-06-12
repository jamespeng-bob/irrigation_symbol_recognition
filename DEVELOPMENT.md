# Development workflow

This project is developed on a local **MacBook** and trained on a remote
**Linux server** with two NVIDIA RTX 6000 Ada GPUs. The two machines play
strictly different roles — read this before editing anything.

```
                                 git push                       git pull
   ┌──────────────────┐  ───────────────────►  ┌──────┐  ───────────────────►  ┌────────────────────┐
   │  Local MacBook   │                        │GitHub│                         │  RTX6000 server    │
   │  (dev + commit)  │  ◄───────────────────  └──────┘                         │  (train + infer)   │
   └──────────────────┘         git pull                                        └────────────────────┘
   /Users/james.peng/                                                           /home/rtx6000/james/
   Desktop/Irrigation/                                                          irrigation_symbol_recognition
```

## Golden rules

1. **All editing happens on the MacBook.** Files on the server must never
   be modified in place; treat that checkout as read-only except for the
   `.venv/` you create there and the `data/`, `runs/`, etc. that
   training writes.
2. **All `git commit` / `git push` happens on the MacBook.** The server
   shares one Linux account (`rtx6000`) across colleagues, so we do not
   add our GitHub credentials there. On the server, only `git fetch`
   and `git pull` are allowed.
3. **Dataset locations are mirrored on the two machines on purpose.**
   `configs/dataset.yaml` uses the relative path
   `../datasets/landscaping-detection.v61-complete_annotations.yolo26`
   which resolves correctly on both:
   - MacBook: `/Users/james.peng/Desktop/Irrigation/datasets/...`
   - Server : `/home/rtx6000/james/datasets/...`
   If you ever change one machine's layout, change the other to match
   (or override `--dataset-config` at runtime).
4. **Each machine has its own `.venv/`.** `.venv/` is git-ignored.
   PyTorch wheels differ between platforms (Mac arm64 / CPU vs Linux
   x86_64 / CUDA), so the venvs are not transferable.
5. **`credentials/` is git-ignored and only exists on the MacBook.**
   The dormant `external_api/` code expects the Google service-account
   JSON there. If we ever need to run it on the server, `scp` the file
   over manually; do NOT commit it.
6. **GPU sharing convention (two-GPU server, multiple Cursor projects).**
   The server has two RTX 6000 Ada cards and is used by several projects
   running concurrently from different Cursor workspaces. To keep things
   collision-free:
   * **This project (`irrigation_symbol_recognition`) defaults to
     `cuda:0`.** All scripts use `--device cuda:0` and `configs/detection.yaml`
     sets `training.device: "auto"` which resolves to `cuda:0` via
     `_select_device(...)`.
   * **Other Cursor workspaces default to `cuda:1`.**
   * **Always check `nvidia-smi` before kicking off training**, in case
     plans changed.
   * **If — and only if — both GPUs are idle**, you may opt into DDP to
     halve wall time:
     ```bash
     python scripts/train_detection.py --device 0,1
     ```
     Coordinate with the other workspace's owner before doing this. If
     they start using `cuda:1` while your DDP run is in flight, your
     training will crash; we'd rather lose 2× speed than 3 h of wall time.

## Typical end-to-end flow

```bash
# --- on the MacBook --------------------------------------------------
# 1. edit code
# 2. inspect / fill configs/irrigation_classes.yaml
# 3. commit + push from the Mac
git add -A
git commit -m "<message>"
git push

# --- on the server ---------------------------------------------------
ssh bobyard-server-6000
cd ~/james/irrigation_symbol_recognition
git pull --ff-only

# First time only: bootstrap the server-side venv
bash scripts/setup_server.sh

# Then run things
source .venv/bin/activate
python scripts/prepare_detection_dataset.py
python scripts/train_detection.py --device cuda:0
# (Use --device 0,1 to use both RTX 6000s if you want DDP.)
```

## SSH

There are **two host aliases** for the same machine, picked based on where
the MacBook is sitting on the network:

```bash
# On the company WiFi (direct LAN reach to the server):
ssh bobyard-server-6000

# Off-network (home, coffee shop, hotel, etc.) -- goes through a
# RustDesk TCP tunnel:
ssh bobyard-server-6000-tunnel
```

Both aliases are configured in the MacBook's `~/.ssh/config` and use the
same key — no password should be required for either. If you ever get
prompted, re-check the SSH config; do NOT paste any passphrase or git
credential into the server.

**Before using `bobyard-server-6000-tunnel`**, make sure RustDesk on the
MacBook is open, connected to the server, and that the **TCP tunnel is
enabled** in the RustDesk session (the tunnel maps a local port on the
Mac to the SSH port on the server; the `-tunnel` SSH alias points at
that local port). If the tunnel isn't running, the SSH connection will
time out or fail to authenticate.

Quick decision tree:

| where am I | command |
|---|---|
| Bobyard office WiFi | `ssh bobyard-server-6000` |
| Anywhere else | open RustDesk → enable TCP tunnel → `ssh bobyard-server-6000-tunnel` |

## Cheat sheet: file paths on each machine

| What                                | Mac path                                                                                  | Server path                                                                  |
|-------------------------------------|-------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Repo                                | `/Users/james.peng/Desktop/Irrigation/irrigation_symbol_recognition`                      | `/home/rtx6000/james/irrigation_symbol_recognition`                          |
| Datasets (parent dir)               | `/Users/james.peng/Desktop/Irrigation/datasets`                                           | `/home/rtx6000/james/datasets`                                               |
| Roboflow YOLO export                | `…/datasets/landscaping-detection.v61-complete_annotations.yolo26`                        | same (mirrored under the server's datasets/ dir)                             |
| Generated YOLO splits (filtered)    | `…/irrigation_symbol_recognition/data/detection/irrigation_yolo`                          | same (relative to the server's repo root)                                    |
| Sliced YOLO dataset (training)      | `…/irrigation_symbol_recognition/data/detection/irrigation_yolo_sliced`                   | same                                                                         |
| Training runs                       | `…/irrigation_symbol_recognition/runs/<exp_name>/`                                        | same                                                                         |
| `credentials/`                      | only on MacBook                                                                           | absent (intentional)                                                         |

The repo-relative paths in `configs/*.yaml` resolve correctly on both
machines because the layout is mirrored. Nothing in the code or configs
hard-codes a machine-specific absolute path.

## Don't accidentally commit machine-local junk

The `.gitignore` already excludes:

- `.venv/`, `__pycache__/`, `*.egg-info/`
- `Ultralytics/` (created on first ultralytics import for its
  `settings.json`)
- `runs/`, `outputs/`, `checkpoints/`, `*.ckpt`, `*.pt`, `*.pth`, `*.onnx`
- `data/` (the generated YOLO / sliced datasets)
- `credentials/`, `*.json.key`, `*.pem`, `.env*`

Before any commit, run `git status` and make sure nothing under those
patterns has crept in.

## "I made a small change directly on the server" — recovery

That's against the rules but it happens. To recover without polluting
the shared server checkout:

```bash
# on the server: copy the diff out
cd ~/james/irrigation_symbol_recognition
git diff > /tmp/server.diff
git checkout -- .   # restore the server checkout to clean state

# on the MacBook: apply, review, commit, push
scp bobyard-server-6000:/tmp/server.diff /tmp/server.diff
git apply /tmp/server.diff
# ... review, then commit + push as usual
```
