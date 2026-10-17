# DANYA optical / mechanical design archive

2026-10-17: preserved the DANYA design workspace previously located at
`~/Documents/Codex/2026-09-21/bl` before removing the local workspace to free disk space.
The existing avatar application in this repository remains separate and unchanged.

## Contents

- `source/`: browsable copies of source code and documents under 1 MiB, preserving original relative paths. Generated work/asset JSON is kept in the archive only. This is a reference subset, not a complete runnable installation.
- `manifest.json`: SHA-256 and byte size of every regular file, plus all symbolic-link targets in the complete workspace.
- `archive-parts.json`: SHA-256 and sizes of the split compressed archive.
- [Complete workspace release](https://github.com/KonChelMa47/Project-DANYA/releases/tag/danya-design-20261017): all models, STL/Blender files, simulation assets, source, guides, historical iterations and existing dependencies. Concatenate the parts before extraction.

## What to open after restoration

Start with `bl/START_HERE.html` and `bl/00_CURRENT/`.
The latter consists of links, not separate copies:

- Head: face, magnet carrier, upper/lower back, unified-head-v5 package.
- Prototype: complete-latest all-parts package.
- Apps: optical simulator and comparison browser projector.
- Guides: prototype and mirror assembly PDFs; read the head-v5 change notes too.

The optical simulator README records unresolved optical feasibility and hardware verification issues. These files are preserved design work, not validated manufacturing instructions.

## Restore

Download every `danya-design-20261017.tar.gz.part*` asset and `SHA256SUMS`
from the release into one empty directory, then run:

```sh
shasum -a 256 -c SHA256SUMS
mkdir restored
cat danya-design-20261017.tar.gz.part* | tar -xzf - -C restored
```

Do not extract over an existing workspace. The archive contains the original `bl/`
layout and symlinks. Some historical scripts use absolute paths, and the link to
`Project-DANYA/open-campus-demo-v2` expects the original workspace location.
For full compatibility, restore `bl` to `~/Documents/Codex/2026-09-21/bl` and
keep this repository at `~/Documents/Project-DANYA`, or adjust those paths first.
The simulator also accepts `server.py --project /path/to/Project-DANYA`.

`manifest.json` is the authoritative inventory for integrity checks; the archive
preserves historical versions to avoid losing unique design data during cleanup.
