#!/usr/bin/env bash
set -euo pipefail

PJ_VERSION_DEFAULT="2.15.1"
PJ_VERSION="${PJ_VERSION:-$PJ_VERSION_DEFAULT}"

VENV_DIR="venv"
PREFIX_DIR="local"
SRC_DIR="pjproject"

NPROC="$(nproc)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log()  { echo -e "\e[1;32m[+] $*\e[0m"; }

install_apt_deps() {
  log "Установка системных зависимостей..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential git autoconf libtool pkg-config \
    swig python3-dev python3-venv libssl-dev
}

create_venv() {
  if [[ -d "$VENV_DIR" ]]; then
    log "Virtualenv $VENV_DIR уже существует, пропускаем создание."
  else
    log "Создаём virtualenv ($VENV_DIR)..."
    python3 -m venv "$VENV_DIR"
  fi
  source "$VENV_DIR/bin/activate"
  pip install -U pip setuptools wheel
}

clone_pjsip() {
  git clone --branch "$PJ_VERSION" --depth 1 https://github.com/pjsip/pjproject.git "$SRC_DIR"
}

build_pjsip() {
  log "Сборка PJSIP (prefix=$SCRIPT_DIR/$PREFIX_DIR)..."
  cd "$SRC_DIR"
  ./configure \
    CFLAGS="$(python3-config --includes) ${CFLAGS-}" \
    --prefix="$SCRIPT_DIR/$PREFIX_DIR" \
    --enable-shared \
    --disable-video --disable-audio --disable-v4l2 --disable-sdl --disable-opus
  make -j"$NPROC"
  make install
  cd "$SCRIPT_DIR"
}

patch_activate() {
  ACT_FILE="$VENV_DIR/bin/activate"
  MARKER="# >>> PJSIP LD_LIBRARY_PATH >>>"
  if ! grep -q "$MARKER" "$ACT_FILE"; then
    cat >> "$ACT_FILE" <<EOF

$MARKER
export LD_LIBRARY_PATH="\$VIRTUAL_ENV/../$PREFIX_DIR/lib:\${LD_LIBRARY_PATH:-}"
# <<< PJSIP LD_LIBRARY_PATH <<<
EOF
  fi
  source "$VENV_DIR/bin/activate"
}

build_python_module() {
  cd "$SRC_DIR/pjsip-apps/src/swig/python"
  make -j"$NPROC"
  pip install --force-reinstall .
  cd "$SCRIPT_DIR"
}

test_import() {
  python - <<'PY'
import pjsua2, pathlib
print("✔ pjsua2:", pathlib.Path(pjsua2.__file__).resolve())
ep = pjsua2.Endpoint()
ep.libCreate()
print("✔ PJSIP:", pjsua2.Endpoint.instance().libVersion().full)
ep.libDestroy()
PY
}

log "PJSUA2 build script запущен (директория: $SCRIPT_DIR)"
install_apt_deps
create_venv
clone_pjsip
build_pjsip
patch_activate
build_python_module
test_import
log "Готово!"

