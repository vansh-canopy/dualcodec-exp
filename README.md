# DualCodec ‑ Vansh's guide

## 1. Clone & create a virtual environment

```bash
# create a fresh python environment (Python ≥ 3.9)
python3 -m venv .venv
source .venv/bin/activate  # or use conda/uv if you prefer
```

---

## 2. Install Python dependencies

The repository ships with a `requirements.txt` file.  Install everything with:

```bash
pip install -r requirements.txt
```

The **`datasets`** package **must be pinned to version `3.6.0`**.  If a newer
version gets installed, force-install the correct one afterwards:

```bash
pip install --upgrade "datasets==3.6.0"
```

(verify with `pip freeze"` that dualcodec is installed in the editable mode)

---

## 3. Authenticate with external services

We store models & datasets on the [Hugging Face Hub](https://huggingface.co) and
track experiments with [Weights & Biases](https://wandb.ai).  Log in once from
the command line so that the tooling can access your credentials:

```bash
huggingface-cli login
wandb login
```

---

## 4. Download pre-trained checkpoints

Run the helper script (it uses `huggingface-cli` under the hood):

```bash
bash download.sh
```

This will create new directories and install dualcodec checkpoints, wav2vec, whisper and any other models you may need.

---

## 5. Set up & run evaluations

The evaluation scripts rely on [VISQOL](https://github.com/google/visqol) for
perceptual quality scoring and on several other metrics implemented in
`metrics.py` (SI-SDR, Multiscale-STFT, etc\.).  A convenience script takes care
of installing the requirements and building VISQOL.

```bash
bash evals.sh  # installs Bazel & clones/compiles VISQOL in ../visqol
```

What the script does:

1. Installs **Bazel** (required to build VISQOL)
2. Clones the official VISQOL repository into the **parent directory** of this
   project (i\.e\. `../visqol`)
3. Builds the C++ library via `bazel build` (see the script for details)

Once it finishes you can run the Python evaluation harness:

```bash
python evals.py
```

All metric implementations live in `metrics.py`; feel free to extend or swap
out as needed.

---

## 6. Ready to hack 🚀

Check the Hydra configuration files in `dualcodec/conf/` for training &
finetuning options.  The dataset used by default is
`emilia_hf_raw_audio_static_batch`.

> ⚠️  If you encounter errors when loading the `emilia_hf` dataset it is usually
> a file-system permission issue\.  A quick fix is to make the project and the
> dataset directory world-readable/writeable:
>
> ```bash
> sudo chmod -R 777 .
> ```
>
If you run into any issues, may the force be with you.

