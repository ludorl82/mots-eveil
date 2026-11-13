#!/usr/bin/env bash
# Prépare l'environnement Python de l'entraîneur (une fois par version de
# requirements.txt et de microWakeWord) et affiche la commande à lancer.
#
#   ./installer.sh [dossier-du-venv]       # défaut : ./venv-eveil
#
# Prérequis : TensorFlow 2.21 déjà installé pour python3 (l'image
# tensorflow/tensorflow:2.21.0-gpu le fournit), accès à github.com et PyPI.
# microWakeWord est installé par ses SOURCES (un fichier .pth), pas par pip :
# ses dossiers audio/ et layers/ n'ont pas d'__init__.py et pip les oublie.
set -euo pipefail
ICI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-$ICI/venv-eveil}"
MWW_COMMIT="${EVEIL_MWW_COMMIT:-4665173cd35f1cff9a61e06fc427f124766c488e}"  # « Pin dependencies (#98) », 2026-07-06

if [ ! -x "$VENV/bin/python" ]; then
  rm -rf "$VENV.tmp"
  python3 -m venv --system-site-packages "$VENV.tmp"
  "$VENV.tmp/bin/pip" install --quiet --no-cache-dir -r "$ICI/entraineur/requirements.txt"
  mkdir -p "$VENV.tmp/mww"
  python3 -c 'import sys, urllib.request; sys.stdout.buffer.write(urllib.request.urlopen(sys.argv[1]).read())' \
    "https://github.com/kahrendt/microWakeWord/archive/$MWW_COMMIT.tar.gz" \
    | tar -xz -C "$VENV.tmp/mww" --strip-components=1
  SITE=$("$VENV.tmp/bin/python" -c 'import site; print(site.getsitepackages()[0])')
  echo "$VENV/mww" > "$SITE/microwakeword-sources.pth"
  mv "$VENV.tmp" "$VENV"
fi
"$VENV/bin/python" -c 'import microwakeword.audio.augmentation, microwakeword.layers'
SITE=$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')
echo "Prêt. Pour l'entraînement sur carte NVIDIA :"
echo "  export LD_LIBRARY_PATH=\"\$(ls -d $SITE/nvidia/*/lib | tr '\\n' ':')\${LD_LIBRARY_PATH:-}\""
echo "  EVEIL_DONNEES=./donnees EVEIL_CACHE=./cache EVEIL_MOT=cumulus $VENV/bin/python -u $ICI/entraineur/entrainer.py"
