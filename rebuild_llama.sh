#!/bin/bash
set -e 

LLAMA_DIR="/mnt/storage/blackbeard"
BUILD_DIR="$LLAMA_DIR/build"
LOG_FILE="$LLAMA_DIR/rebuild.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "===================================================="
echo "ULTIMATE REBUILD STARTED: $(date)"
echo "===================================================="

error_handler() {
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: Script failed at line $1. Check $LOG_FILE for details."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "===================================================="
}
trap 'error_handler $LINENO' ERR

# Ensure PATH and LD_LIBRARY_PATH point to your 13.1 installation
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

if ! command -v nvcc &> /dev/null; then
    echo "--- Error: nvcc not found in PATH ---"
    exit 1
fi

cd "$LLAMA_DIR"

echo "--- Checking for updates ---"
# Ensure we're on master (not detached HEAD) so git pull works
git checkout master 2>/dev/null || true
git pull || echo "--- Warning: git pull failed, proceeding with local source ---"

echo "--- Pre-fetching UI assets ---"
# llama.cpp cmake tries to download UI assets during build (npm build or HF bucket download).
# Both often fail (no npm, SSL/HTTP errors). Pre-download from GitHub releases instead.
UI_DIST_DIR="$LLAMA_DIR/tools/ui/dist"
if [ -d "$UI_DIST_DIR" ] && [ -n "$(ls -A "$UI_DIST_DIR" 2>/dev/null)" ]; then
    echo "  UI dist already exists at $UI_DIST_DIR (delete it to force re-download)"
else
    rm -rf "$UI_DIST_DIR"
    mkdir -p "$UI_DIST_DIR"
    
    # Get the latest release tag from GitHub API
    echo "  Checking latest release..."
    LATEST_TAG=$(curl -sfL 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' \
      | python3 -c "import sys,json; print(json.load(sys.stdin).get('tag_name',''))" 2>/dev/null || echo "")
    
    if [ -n "$LATEST_TAG" ]; then
        UI_URL="https://github.com/ggml-org/llama.cpp/releases/download/$LATEST_TAG/llama-$LATEST_TAG-ui.tar.gz"
        echo "  Downloading UI assets from $LATEST_TAG..."
        if curl -fL "$UI_URL" | tar xzf - -C "$UI_DIST_DIR" --strip-components=1; then
            echo "  UI assets ($LATEST_TAG) extracted to tools/ui/dist/"
        else
            echo "  WARNING: Failed to download UI assets from $LATEST_TAG"
            echo "  Build will succeed without an embedded UI"
        fi
    else
        echo "  WARNING: Could not fetch latest release tag from GitHub API"
        echo "  Build will succeed without an embedded UI"
    fi
fi

if [ -d "$BUILD_DIR" ]; then
    echo "--- Purging old CMake build cache ---"
    rm -rf "$BUILD_DIR"
fi

echo "--- Configuring CMake for RTX 5090 + Intel 285K ---"
# We lied about the compiler ID, so now we must provide the default standards manually
cmake -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_C_COMPILER=gcc-15 \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_FLAGS="-ccbin /usr/bin/g++-15 -isystem /usr/local/cuda/include" \
  -DCMAKE_CUDA_COMPILER_ID=NVIDIA \
  -DCMAKE_CUDA_COMPILER_VERSION=13.3 \
  -DCMAKE_CUDA_STANDARD_COMPUTED_DEFAULT=17 \
  -DCMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT=ON \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DGGML_LTO=ON \
  -DGGML_CPU_KLEIDIAI=OFF \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCMAKE_LINK_DEPENDS_USE_LINKER=OFF

echo "--- Starting the build ---"
cmake --build "$BUILD_DIR" --config Release -j "$(nproc)"

echo "--- Build complete ---"
echo "===================================================="
echo "REBUILD COMPLETED SUCCESSFULLY: $(date)"
echo "===================================================="
