#!/bin/bash
# Rebuild the "Allreduce after the refactor" page and its PNGs.
#
#   reproduce.sh            two-rank run on the cards, then extract/generate/render
#   reproduce.sh --no-run   reuse logs/ and only extract/generate/render
#
# Run from anywhere; TS_DIR defaults to ../torch-spyre next to this directory.
# Needs 2 Spyre cards and the usual Spyre env (senlib/flex/spyre_comms) for
# the run step. Rendering needs resvg-py, installed into a private venv
# (RENDER_VENV, default /tmp/diag-venv) on first use.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_VENV="${RENDER_VENV:-/tmp/diag-venv}"
RUN=1
[[ "${1:-}" == "--no-run" ]] && RUN=0

if [[ $RUN -eq 1 ]]; then
    TS_DIR="${TS_DIR:-$(cd "$HERE/../torch-spyre" && pwd)}"
    # Record the exact code the logs came from.
    {
        for r in torch-spyre flex spyre-comms; do
            d="$TS_DIR/../$r"
            [[ -d "$d/.git" ]] && echo "$r $(git -C "$d" rev-parse --abbrev-ref HEAD) $(git -C "$d" rev-parse --short HEAD)"
        done
        date -u
    } > "$HERE/logs.versions.txt"

    TMP_LOGS="$(mktemp -d)"
    (
        cd "$TS_DIR"
        export _LOGDIR="$TMP_LOGS" DTLOG_LEVEL=debug TORCH_LOGS=output_code TS_DIR
        torchrun --nproc-per-node 2 --no-python bash tests/distributed/split_output.sh \
            python3 -u "$HERE/run_ar.py"
    )
    grep -q "### ITER 1 END" "$TMP_LOGS/output-at-rank-0.txt"
    mkdir -p "$HERE/logs"
    cp "$TMP_LOGS"/output-at-rank-*.txt "$HERE/logs/"
    rm -rf "$TMP_LOGS"
fi

python3 -I "$HERE/extract.py" "$HERE/logs" > "$HERE/extract.out.txt"
python3 -I "$HERE/gen.py"

if [[ ! -x "$RENDER_VENV/bin/python" ]]; then
    python3 -m venv "$RENDER_VENV"
    "$RENDER_VENV/bin/pip" -q install resvg-py
fi
"$RENDER_VENV/bin/python" -I "$HERE/render_png.py"
