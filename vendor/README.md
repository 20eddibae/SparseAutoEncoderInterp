# vendor/

Vendored snapshot of [`shehper/sparse-dictionary-learning`](https://github.com/shehper/sparse-dictionary-learning) — the "instrument" code that trains the 1-layer transformer + sparse autoencoder. This project's analysis layer imports from it via `src/scf/sdl_bridge.py`.

## Why vendored?
The analysis here is a probability project (`docs/PROJECT.md`); the SDL repo can update independently. Pinning a snapshot keeps results reproducible.

## Unpack

On any machine (laptop or HPC):

```bash
cd <wherever-you-keep-code>
tar -xzf <path-to-this-repo>/vendor/sparse-dictionary-learning.tar.gz
# produces ./sparse-dictionary-learning/
```

Then tell the analysis layer where it landed:

- **Edit** `configs/default.yaml` (or `configs/discovery.yaml`) and set `sdl_repo` to the unpacked path.
- **Or** set the env var: `export SCF_SDL_REPO=/abs/path/to/sparse-dictionary-learning`. The env var trumps the config.

## What's stripped
- `.git` (the upstream history; recover by cloning fresh from `shehper/sparse-dictionary-learning` if needed)
- `*.pt`, `*.bin`, `__pycache__/`, `.DS_Store` (large/derived files; none existed in the snapshot anyway)

## Upstream pin
Snapshot taken from `aa667b1` ("updated gitignore"), the local HEAD at the time of bundling.
