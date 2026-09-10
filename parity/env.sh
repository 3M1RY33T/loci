# Source me from the rust-port worktree: `. parity/env.sh`
#
# LOCI_HOME points every loci invocation at the frozen snapshot, never at
# ~/.loci, which belongs to the editable v0.5.0 install in the main worktree.
# paths.py rejects a relative path, so this is absolute on purpose.
export LOCI_HOME="$HOME/.loci-rust"

# `loci` is an EDITABLE install of the MAIN worktree: its .pth resolves
# `import loci` to <main>/src/loci, not to this one. Without this line every
# test and every recorded answer executes the code being ported AWAY from,
# and Phase 0b would edit files that never run. Caught by an ImportError on a
# function that existed here and not there -- silent for anything else.
LOCI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export PYTHONPATH="$LOCI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# rustup was installed with --no-modify-path, so the toolchain is deliberately
# NOT on the login shell's PATH -- it belongs to this port, not to the machine.
[ -d "$HOME/.cargo/bin" ] && export PATH="$HOME/.cargo/bin:$PATH"

export TOKENIZERS_PARALLELISM=false   # a fork warning on stderr is still noise
echo "LOCI_HOME=$LOCI_HOME"

# Build the extension. ALWAYS --release.
#
# `maturin develop` defaults to the dev profile -- unoptimized, with debug
# info -- and the tokenizer measured 40.6ms there against Python's 14.6ms on
# the same input. The same code built --release runs 4.4ms. A debug build does
# not merely look slow, it inverts the reason for the port, so no timing number
# in this work may come from one.
loci_build () {
  ( cd "$LOCI_ROOT" \
    && VIRTUAL_ENV="$LOCI_ROOT/.venv-rust" "$LOCI_ROOT/.venv-rust/bin/maturin" \
       develop --release -m rust/loci-py/Cargo.toml "$@" )
}
