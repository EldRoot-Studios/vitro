#!/usr/bin/env bash
# Builds bin/vitro_view.exe (trace inspector) and bin/vitro_demo.exe (3D step player) with MinGW g++.
# Both binaries link statically against Windows GDI/Win32 APIs and zlib so they run standalone on Windows.
#
#   bash build.sh           # builds both binaries
#   bash build.sh view      # builds only bin/vitro_view.exe
#   bash build.sh demo      # builds only bin/vitro_demo.exe
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/cpp/src"
INC="$HERE/cpp/include"
BIN="$HERE/bin"
OBJ="$BIN/_obj"
MINGW="${MINGW_BIN:-/c/msys64/mingw64/bin}"
if [ -d "$MINGW" ]; then
    export PATH="$MINGW:$PATH"
fi
mkdir -p "$BIN" "$OBJ"
FLAGS=(-O2 -std=c++17 -Wall -Wextra -ffp-contract=off "-I$INC")
LIBS=(-static -lgdi32 -luser32 -lcomdlg32 -lshell32 -lwinmm -lz)
which_one="${1:-all}"

build_bin() {
    local name="$1"; shift
    g++ "${FLAGS[@]}" "$@" "${LIBS[@]}" -o "$OBJ/$name.new.exe"
    if [ -f "$BIN/$name.exe" ]; then
        mv -f "$BIN/$name.exe" "$OBJ/$name.old.exe"
    fi
    mv -f "$OBJ/$name.new.exe" "$BIN/$name.exe"
    echo "built $BIN/$name.exe"
}

if [ "$which_one" = all ] || [ "$which_one" = view ]; then
    build_bin vitro_view "$SRC"/view_*.cpp "$SRC"/selftest.cpp
fi
if [ "$which_one" = all ] || [ "$which_one" = demo ]; then
    build_bin vitro_demo "$SRC"/demo_*.cpp
fi
