# Install

This page installs DOROTHY and checks that it works. DOROTHY needs Python 3.10
or later; pip installs PyTorch 2.4 or later, NumPy 2.0 or later and the other
dependencies.

## Step 1: Install PyTorch for your GPU

If you will train on an NVIDIA GPU, first install the PyTorch build for your
CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).
On a CPU or on Apple silicon, skip this step: pip installs a PyTorch that
works there.

## Step 2: Install DOROTHY

```bash
git clone https://github.com/joshspeagle/dorothy.git
cd dorothy
pip install -e .
```

This installs the `dorothy` package and the `dorothy` command. To also install
the test and lint tools, run `pip install -e ".[dev]"` instead.

## Step 3: Check the install

```bash
dorothy --version
```

The command prints the version:

```
dorothy 0.1.0
```

## Step 4: Check the device

```bash
python -c "from dorothy.utils import resolve_device; print(resolve_device())"
```

This prints the device that DOROTHY uses when a configuration or a command
sets the device to `auto`: `cuda` for an NVIDIA GPU, else `mps` for an Apple
silicon GPU, else `cpu`. If it prints `cpu` on a machine with an NVIDIA GPU,
reinstall PyTorch as in step 1.

To choose a device, set `device` in the configuration, or pass `--device` to
`dorothy train`, `dorothy evaluate` or `dorothy predict`
([Commands](../reference/commands.md)).

## Next

The [quick start](quickstart.md) trains, evaluates and uses a model on a
small synthetic catalogue, so it checks the whole install.
