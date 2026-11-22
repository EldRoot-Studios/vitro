#!/usr/bin/env bash
# Builds the Vitro programs with MinGW g++ (Windows):
#   bin/vitro.exe         the app: inspector and generation replay in one window (the web app of web/, shown by the
#                         Microsoft Edge engine that Windows already has). One file: the app, the sample traces and
#                         the licence notices are inside it.
#   bin/vitro_view.exe    the older trace inspector (Win32 and GDI)
#   bin/vitro_demo.exe    the older 3D step player (Win32 and GDI)
# All of them link statically, so they run standalone.
#
#   bash build.sh           # builds everything
#   bash build.sh app       # builds only bin/vitro.exe
#   bash build.sh view      # builds only bin/vitro_view.exe
#   bash build.sh demo      # builds only bin/vitro_demo.exe
#
# bin/vitro.exe is built from web/dist, which is committed; `npm run build` in web/ renews it after a change to the
# web app. The first build downloads the Microsoft WebView2 SDK (tools/fetch_webview2.py) and needs Python 3.
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

# Replaces bin/<name>.exe with the freshly built $OBJ/<name>.new.exe, keeping the previous one as $OBJ/<name>.old.exe.
install_bin() {
    local name="$1"
    if [ -f "$BIN/$name.exe" ]; then
        mv -f "$BIN/$name.exe" "$OBJ/$name.old.exe"
    fi
    mv -f "$OBJ/$name.new.exe" "$BIN/$name.exe"
    echo "built $BIN/$name.exe"
}

build_bin() {
    local name="$1"; shift
    g++ "${FLAGS[@]}" "$@" "${LIBS[@]}" -o "$OBJ/$name.new.exe"
    install_bin "$name"
}

build_app() {
    local py="${PYTHON:-}"
    if [ -z "$py" ]; then
        py="$(command -v python3 || command -v python || true)"
    fi
    if [ -z "$py" ]; then
        echo "build.sh: Python 3 is needed to build bin/vitro.exe (set PYTHON=... if it is not on the PATH)" >&2
        exit 1
    fi
    local version
    version="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$HERE/pyproject.toml" | head -n 1)"
    "$py" "$HERE/tools/fetch_webview2.py"
    "$py" "$HERE/tools/make_app_resources.py" --out "$OBJ/app"
    windres "$OBJ/app/vitro_app.rc" -O coff -o "$OBJ/app/vitro_app.res.o"
    g++ -O2 -std=c++17 -Wall -Wextra "-I$INC" -isystem "$HERE/_deps/webview2" \
        -DUNICODE -D_UNICODE -DWINVER=0x0A00 -D_WIN32_WINNT=0x0A00 "-DVITRO_VERSION=\"$version\"" \
        "$SRC/app_main.cpp" "$OBJ/app/vitro_app.res.o" \
        -static -municode -mwindows -lole32 -loleaut32 -luuid -lshlwapi -lshell32 -ldwmapi -ladvapi32 -luser32 -lgdi32 \
        -o "$OBJ/vitro.new.exe"
    install_bin vitro
}

if [ "$which_one" = all ] || [ "$which_one" = app ]; then
    build_app
fi
if [ "$which_one" = all ] || [ "$which_one" = view ]; then
    build_bin vitro_view "$SRC"/view_*.cpp "$SRC"/selftest.cpp
fi
if [ "$which_one" = all ] || [ "$which_one" = demo ]; then
    build_bin vitro_demo "$SRC"/demo_*.cpp
fi
