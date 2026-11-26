#!/bin/sh
# Builds the release archive from a commit: dist/zdepth.tar.gz (a `zdepth/` folder, ready to
# drop into EffectCraft's helpers folder) and dist/zdepth.tar.gz.sha256. The asset names carry
# no version, so .../releases/latest/download/zdepth.tar.gz always names the newest.
#
#   tools/make_release.sh [commit-ish]   (default HEAD)
set -eu
cd "$(dirname "$0")/.."
REF="${1:-HEAD}"
mkdir -p dist
git archive --format=tar --prefix=zdepth/ "$REF" | gzip -9n > dist/zdepth.tar.gz
cp install.sh dist/install.sh
cd dist
if command -v sha256sum >/dev/null 2>&1; then sha256sum zdepth.tar.gz > zdepth.tar.gz.sha256
else shasum -a 256 zdepth.tar.gz > zdepth.tar.gz.sha256; fi
cat zdepth.tar.gz.sha256
