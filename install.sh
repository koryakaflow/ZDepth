#!/bin/sh
# Installs or updates ZDepth, the depth-map effect helper for EffectCraft.
#
#   sh install.sh [--version vX.Y.Z] [--models "Small Video-Small"] [--no-setup]
#                 [--archive zdepth.tar.gz] [--dir HELPERS_FOLDER]
#   sh install.sh --uninstall [--purge-models] [--dir HELPERS_FOLDER]
#
# 1. downloads the release archive from GitHub and checks its SHA-256,
# 2. puts it in EffectCraft's helpers folder as `zdepth/` (replacing an older ZDepth there),
# 3. runs `effectcraft-helper-zdepth --setup`: installs PyTorch (a few GB) into
#    ~/.local/share/zdepth/venv and downloads the models named.
# Running it again updates ZDepth. Nothing is installed outside those two places and the
# Hugging Face cache. --uninstall removes both places; --purge-models also removes ZDepth's
# models from the Hugging Face cache (other apps may share that cache, so not by default).
set -eu

REPO="koryakaflow/ZDepth"
VERSION="latest"
MODELS="Small Video-Small"
SETUP=1
ARCHIVE=""
HELPERS=""
UNINSTALL=0
PURGE=0

die() { echo "zdepth install: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --no-setup) SETUP=0; shift ;;
        --archive) ARCHIVE="$2"; shift 2 ;;
        --dir) HELPERS="$2"; shift 2 ;;
        --uninstall) UNINSTALL=1; shift ;;
        --purge-models) PURGE=1; shift ;;
        -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

# EffectCraft's settings folder (docs/preferences.md): helpers/ in it is searched, with its
# direct subfolders.
if [ -z "$HELPERS" ]; then
    case "$(uname -s)" in
        Darwin) HELPERS="$HOME/Library/Application Support/EffectCraft/helpers" ;;
        Linux) HELPERS="${XDG_CONFIG_HOME:-$HOME/.config}/effectcraft/helpers" ;;
        *) die "only Linux and macOS are supported for now" ;;
    esac
fi
TARGET="$HELPERS/zdepth"
DATA="${ZDEPTH_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/zdepth}"

if [ "$UNINSTALL" = 1 ]; then
    if command -v pgrep >/dev/null 2>&1 && pgrep -f "from zdepth.helper import main" >/dev/null 2>&1; then
        die "ZDepth is running in EffectCraft: quit EffectCraft first"
    fi
    HUB="${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}"
    set -- "$TARGET" "$DATA"
    if [ "$PURGE" = 1 ]; then
        for m in Depth-Anything-V2-Small-hf Depth-Anything-V2-Base-hf Depth-Anything-V2-Large-hf \
                 Video-Depth-Anything-Small Video-Depth-Anything-Large; do
            set -- "$@" "$HUB/models--depth-anything--$m"
        done
    fi
    gone=0
    for path in "$@"; do
        [ -e "$path" ] || continue
        echo "Removing $path ($(du -sh "$path" 2>/dev/null | cut -f1))"
        rm -rf "$path"
        gone=1
    done
    [ "$gone" = 1 ] || echo "Nothing to remove: ZDepth isn't installed here."
    [ "$PURGE" = 1 ] || [ ! -d "$HUB" ] || ! ls -d "$HUB"/models--depth-anything--* >/dev/null 2>&1 ||
        echo "Models kept in $HUB (models--depth-anything--*); --purge-models removes them."
    for other in "$HELPERS"/effectcraft-helper-zdepth "$HELPERS"/*/effectcraft-helper-zdepth; do
        [ -e "$other" ] || [ -L "$other" ] || continue
        echo "Another ZDepth is still in the helpers folder (not installed by this script): $other"
    done
    exit 0
fi

command -v tar >/dev/null 2>&1 || die "needs tar"
command -v uv >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 ||
    die "needs uv (https://docs.astral.sh/uv/) or python3"

# Another ZDepth in the helpers folder (a symlink from a development checkout, say) would offer
# the same effect twice; EffectCraft refuses the second, so which one runs would be a guess.
for other in "$HELPERS"/effectcraft-helper-zdepth "$HELPERS"/*/effectcraft-helper-zdepth; do
    [ -e "$other" ] || [ -L "$other" ] || continue
    case "$other" in "$TARGET"/*) continue ;; esac
    die "another ZDepth is installed at $other; remove it first"
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fetch() {  # url file
    if command -v curl >/dev/null 2>&1; then curl -fsSL "$1" -o "$2"
    elif command -v wget >/dev/null 2>&1; then wget -q "$1" -O "$2"
    else die "needs curl or wget"; fi
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

if [ -n "$ARCHIVE" ]; then
    cp "$ARCHIVE" "$TMP/zdepth.tar.gz"
    [ -f "$ARCHIVE.sha256" ] && cp "$ARCHIVE.sha256" "$TMP/zdepth.tar.gz.sha256"
else
    if [ "$VERSION" = latest ]; then BASE="https://github.com/$REPO/releases/latest/download"
    else BASE="https://github.com/$REPO/releases/download/$VERSION"; fi
    echo "Downloading ZDepth ($VERSION)..."
    fetch "$BASE/zdepth.tar.gz" "$TMP/zdepth.tar.gz" || die "download failed: $BASE/zdepth.tar.gz"
    fetch "$BASE/zdepth.tar.gz.sha256" "$TMP/zdepth.tar.gz.sha256" || die "download failed: checksum"
fi
if [ -f "$TMP/zdepth.tar.gz.sha256" ]; then
    want="$(cut -d' ' -f1 < "$TMP/zdepth.tar.gz.sha256")"
    [ "$(sha256 "$TMP/zdepth.tar.gz")" = "$want" ] || die "checksum mismatch: the download is damaged"
elif [ -z "$ARCHIVE" ]; then
    die "no checksum to check the download against"
fi

tar -xzf "$TMP/zdepth.tar.gz" -C "$TMP"
[ -x "$TMP/zdepth/effectcraft-helper-zdepth" ] || die "the archive has no zdepth/effectcraft-helper-zdepth"
NEW="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$TMP/zdepth/zdepth/__init__.py")"

# Swap the folder in whole, so a failed update leaves the old version working.
mkdir -p "$HELPERS"
if [ -e "$TARGET" ]; then
    OLD="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' "$TARGET/zdepth/__init__.py" 2>/dev/null || true)"
    mv "$TARGET" "$TMP/previous"
    mv "$TMP/zdepth" "$TARGET" || { mv "$TMP/previous" "$TARGET"; die "could not replace $TARGET"; }
    echo "Updated ZDepth ${OLD:-?} -> $NEW in $TARGET"
else
    mv "$TMP/zdepth" "$TARGET"
    echo "Installed ZDepth $NEW in $TARGET"
fi

if [ "$SETUP" = 1 ]; then
    echo "Setting up (PyTorch and models: $MODELS). The first time this downloads a few GB."
    # shellcheck disable=SC2086 # model names are separate arguments
    "$TARGET/effectcraft-helper-zdepth" --setup $MODELS
else
    echo "Skipped setup: run \"$TARGET/effectcraft-helper-zdepth\" --setup before using ZDepth."
fi

echo
echo "Done. In EffectCraft: Effect > Scan for Effect Helpers (or restart it if ZDepth was"
echo "already running). ZDepth is under 3D Channel."
