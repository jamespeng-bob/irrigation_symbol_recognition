# Irrigation Symbol Recognition

Build, train, and evaluate a deep-learning model that detects and classifies
**irrigation-related valve / fitting symbols** on landscaping construction
drawings.

The source dataset (Roboflow's `landscaping-detection v61` export, format
**YOLO26**) lives at
`../datasets/landscaping-detection.v61-complete_annotations.yolo26/` and
contains every symbol annotated on each drawing — both irrigation-related
and unrelated. The first modeling decision is therefore to declare which
class names are considered "irrigation". That declaration lives in
[`configs/irrigation_classes.yaml`](configs/irrigation_classes.yaml).

> **Workflow note.** Development happens on a local MacBook; training
> runs on a Linux server with two RTX 6000 Ada GPUs (`ssh
> bobyard-server-6000`). **All edits / commits / pushes happen on the
> Mac. The server only does `git pull` + `setup_server.sh`** (it shares
> one Linux account, so we don't put our GitHub credentials there).
> See [`DEVELOPMENT.md`](DEVELOPMENT.md) for the full set of rules.

---

## Repository layout

```
irrigation_symbol_recognition/
├── configs/
│   ├── dataset.yaml              # dataset paths + global metadata
│   ├── irrigation_classes.yaml   # <-- fill in the irrigation class names here
│   ├── detection.yaml            # YOLO slicing / training / inference parameters
│   └── all_class_names.csv       # reference: every class name + train/test counts
├── credentials/                  # service-account JSON (git-ignored)
├── external_api/                 # colleagues' Vertex AI client (kept; NOT USED right now)
│   └── call_symbol_localizer.py
├── scripts/
│   ├── inspect_dataset.py            # CLI: prints dataset stats / filter report
│   ├── prepare_detection_dataset.py  # YOLO filter + tile slicing
│   ├── train_detection.py            # Ultralytics YOLO training
│   ├── inference_detection.py        # sliding-window inference + NMS
│   └── setup_server.sh               # one-shot venv bootstrap on the Linux server
├── src/irrigation_symbol_recognition/
│   ├── data/                     # YOLO dataset loading (data.yaml + per-image .txt)
│   ├── detection/                # tiled-YOLO pipeline (filter, slicing, train, inference, NMS)
│   ├── models/                   # (placeholder) future model code
│   ├── training/                 # (placeholder) custom training loops if needed
│   └── utils/                    # config loaders
├── .gitignore
├── README.md
└── requirements.txt
```

### Why these subfolders for the shared files?

The two files my colleagues shared are organized as follows:

| File                                  | New location                                       | Why |
|---------------------------------------|----------------------------------------------------|-----|
| `call_symbol_localizer.py`            | `external_api/call_symbol_localizer.py`            | It's an external-API client, not modeling code. Keeping it isolated makes it easy to re-sync with their copy and to swap implementations later. |
| `inference-428300-7af7f5da75dc.json`  | `credentials/inference-428300-7af7f5da75dc.json`   | It's a Google service-account key — must never be committed. `credentials/` is `.gitignore`d. |

**Note:** `external_api/` is currently dormant. We may wire it back in
later for a hybrid embedding-retrieval baseline, but the YOLO pipeline
does not depend on it.

The script's `DEFAULT_CREDENTIALS_FILE` was updated to point at
`credentials/...` relative to the file's own location, so the existing CLI
keeps working without any environment configuration.

### What the two endpoints actually do

Reading `external_api/call_symbol_localizer.py`, your understanding is
exactly right:

1. **Localization endpoint** (`endpoints/1648990364733800448`) — given an
   image URL, returns a list of candidate symbol bounding boxes (with
   `x1/y1/x2/y2`, a confidence, a class id/name guess, and a `crop_id`).
   The script runs NMS on the raw output.
2. **Classification / embedding endpoint** (`endpoints/9163730128117694464`)
   — given the `crop_id`s returned by step 1, returns a record per crop
   that contains the **embedding vector** (plus classification metadata).
   `scripts/inspect_dataset.py` doesn't call these, but the
   `IsolatedSymbolClient` in `external_api/` is the entry point we'll use
   when we want to build a vector index of known irrigation symbols or
   ensemble the company model with whatever we train here.

---

## Dataset at a glance

Counted directly from the YOLO export's label files (see `scripts/inspect_dataset.py`):

| Property                                 | Value      |
|------------------------------------------|------------|
| Splits provided by Roboflow              | `train`, `test` (no `valid` — we carve one off `train`) |
| Images (train / test)                    | **390 / 9** |
| Annotations (train / test)               | **177,380 / 20,469** |
| Classes (`nc` in `data.yaml`)            | **4,541** (Roboflow's YOLO export drops the supercategory `Symbols-mQMX`) |
| 4-digit string class names               | **4,436 / 4,541  (~97.7 %)** |
| Non-4-digit class names                  | **105** (short letter codes like `A`, `AB`, `CYP`; two 5-digit numbers `20073`, `45666`; a handful of plant names like `Furcraea`; placeholders like `null-1`) |
| Classes with ≥1 annotation in train      | 4,359 |
| Classes never annotated in train         | 182 |

> **Confirmed:** ~97.7 % of class names are four-digit strings. Use the
> string form with leading zeros when filling the irrigation config — e.g.
> `"0072"`, not `72`. The YOLO `class_id` is whatever index that name has
> in the alphabetically-sorted `names:` list in `data.yaml`; you don't
> have to worry about it — the filter step looks classes up by name.

Top-5 most frequent classes (train + test combined) are:
`1642`, `0072`, `1645`, `0524`, `1656` — these are very likely irrigation
symbols and a good starting point to investigate.

---

## Quickstart

```bash
# 1) Set up the environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Look at the dataset
python scripts/inspect_dataset.py

# 3) Fill in configs/irrigation_classes.yaml with the irrigation class names.
#    configs/all_class_names.csv is sorted by total annotation count and is
#    the easiest place to pick them from.

# 4) Validate your config by re-running with --apply-filter
python scripts/inspect_dataset.py --apply-filter
```

---

## How to fill `configs/irrigation_classes.yaml`

1. Open `configs/all_class_names.csv`. It lists every class along with its
   `yolo_id`, `train_count` and `test_count` (sorted by total annotation count).
2. Copy the `name` (NOT the `yolo_id`) of each irrigation-related class
   into the `irrigation_classes:` list. Wrap each in quotes to keep the
   leading zeros:

   ```yaml
   irrigation_classes:
     - "0072"
     - "0524"
     - "1642"
   ```

3. If multiple raw class names actually represent the same physical symbol
   (a drafting variant), group them with `class_groups`:

   ```yaml
   class_groups:
     rotor_head:
       - "0072"
       - "0079"
   ```

4. Pick how the model should treat non-irrigation symbols via
   `non_irrigation_policy`:
   - `"background"` (default) — collapse them into one extra
     `__background__` class so the detector learns to suppress them.
   - `"drop"` — pretend they don't exist (cleaner label space, but harder
     for the detector to handle distractors at inference time).
   - `"negative"` — keep the image but discard the boxes.

After editing, re-run `python scripts/inspect_dataset.py --apply-filter` to
see how many annotations survive and to catch typos in class names.

---

## YOLO detection pipeline

We train a YOLO model on **tiled** crops of the drawings, following the
same slicing technique used by colleagues in
`../../electrical_dev-main/detection/` (also added to the workspace as a
reference folder). The technique has three pieces:

1. **Sliding window crop.** Each full drawing is split into
   `slice_size × slice_size` tiles (default `1280`) with `overlap` (default
   `0.25`). The last row/column is anchored to `(W − S, H − S)` so the
   right/bottom edge is fully covered; if the crop still goes past the
   image boundary, it is zero-padded.
2. **Bounding-box handling near the cut line.** Implemented in
   `src/irrigation_symbol_recognition/detection/slicing.py`
   (`adjust_cut_bbox_in_slice`):
   - With `fit_box_to_slice: true`, the box is **clipped** to the slice
     boundary (SAHI-style).
   - With `fit_box_to_slice: false` (the project default), if more than
     50 % of the original box area is inside the slice, the **full
     original box is emitted** (it may poke past the tile edge — YOLO
     learns the true symbol extent). Otherwise the annotation is **dropped
     for that tile** (avoids teaching the model on half-symbols).
3. **Sliding-window inference + NMS.** Same tile grid, batched
   `model.predict`, local boxes lifted back into full-image coordinates,
   then per-class NMS merges duplicates that the overlap region produces.

### About `external_api/`

The Vertex AI clients in `external_api/` (and the credentials in
`credentials/`) are **kept in the repo but not used by the YOLO
pipeline**. We may revisit them later to compare YOLO against the
company's localization + embedding endpoints, or to build a hybrid
retrieval baseline.

### Step-by-step

```bash
# 0) Once: install dependencies (uses ultralytics + opencv + pyyaml + ...)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Fill `configs/irrigation_classes.yaml` with the irrigation class names.
#    Use `configs/all_class_names.csv` as the reference list.

# 2) Filter the source YOLO export down to the irrigation label space and
#    slice every page into tiles.
#    Reads: configs/{dataset,irrigation_classes,detection}.yaml
#    Writes: data/detection/irrigation_yolo/         (filtered YOLO + data.yaml)
#            data/detection/irrigation_yolo_sliced/  (sliced tiles + data.yaml)
python scripts/prepare_detection_dataset.py
# Note: images are symlinked into the filtered dataset by default (saves disk).
# Add `--copy-images` if you want full copies.

# 3) Train YOLO on the sliced dataset.
python scripts/train_detection.py
# Auto-selects CUDA -> MPS -> CPU. Override knobs with --epochs / --batch /
# --device / --name, or edit `configs/detection.yaml`.

# 4) Run sliding-window inference on a new full-page drawing.
python scripts/inference_detection.py \
    --model runs/irrigation_yolo_v1/weights/best.pt \
    --image path/to/drawing.jpg \
    --out-dir runs/irrigation_yolo_v1/predictions
```

### Where every reference-pipeline piece lives in our code

| Reference (`electrical_dev-main/detection/`) | Our port |
|----------------------------------------------|----------|
| `data.py: bbox_intersects_slice`             | `detection/slicing.py: bbox_intersects_slice` |
| `data.py: adjust_cut_bbox_in_slice`          | `detection/slicing.py: adjust_cut_bbox_in_slice` |
| `data.py: adjust_bbox_for_slice`             | `detection/slicing.py: adjust_bbox_for_slice` |
| `data.py: slice_image_and_labels`            | `detection/slicing.py: slice_image_and_labels` |
| `data.py: process_split` / `slice_data`      | `detection/slicing.py: process_split` / `slice_data` |
| `train.py: train_with_ultralytics`           | `detection/train.py: train_with_ultralytics` |
| `inference.py: inference_detection`          | `detection/inference.py: inference_detection` |
| `utils.py: nmm_boxes` (electrical-specific)  | `detection/nms.py: nms_boxes` (per-class NMS — see note) |

> **Note on NMS vs NMM.** The reference uses domain-specific Non-Maximum
> Merging tuned for electrical classes (`text`, `panel`, `symbol`,
> `symbol-with-text`, `neon`). The merge groups don't apply to our
> 4-digit irrigation labels, so we use a straight per-class NMS at the
> stitched-image level. If we later want to merge across related
> irrigation classes (e.g. multiple drafting variants of the same head),
> the right place to add merge groups is `detection/nms.py`.

## Pipeline-internals reference

When `prepare_detection_dataset.py` runs against the YOLO26 export it
does **not** re-encode bounding boxes — it just rewrites the per-image
`.txt` files. Concretely, for every line `src_id xc yc w h` in a source
label:

1. Look up the source class **name** via the source `data.yaml`.
2. If the name is in `irrigation_classes` (or a member of a group),
   replace `src_id` with the corresponding contiguous irrigation label
   id; pass `xc yc w h` through verbatim.
3. Otherwise, apply `non_irrigation_policy`:
   - `drop` -> skip the line.
   - `background` -> emit with the `__background__` label id.
   - `negative` -> skip the line but keep the (empty) label file so
     the image remains in the training set as a negative example.

The filtered dataset is written with a fresh `data.yaml` (`path`,
`train`, `val`, `test`, `nc`, `names`) so Ultralytics can train on it
directly. The reference pipeline's slicing step then runs on top of
this filtered tree.

## Next steps (modeling)

1. Fill in `configs/irrigation_classes.yaml`.
2. Run `python scripts/prepare_detection_dataset.py` and inspect the
   per-split / per-label annotation counts in the report — make sure
   every label has enough training boxes after tiling.
3. Train a YOLO baseline (`configs/detection.yaml` defaults — `model_size: m`,
   `image_size: 640`, `epochs: 50`). Tweak `slice_size` if the symbols
   come out too small at the model input size.
4. Evaluate per-class precision / recall on the `test` split and on the
   page-grouped `valid` split.
5. Revisit `external_api/` for a hybrid embedding-retrieval comparison
   once the YOLO baseline is established.
