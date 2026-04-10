#!/bin/bash
set -e
cd /home/drixs2050/Documents/Mortal

CONFIGS=(
    "bench_ddp_nocompile_bs2560"
    "bench_ddp_compile_bs2560"
    "bench_ddp_compile_bs3072"
    "bench_ddp_compile_bs4096"
)

PORTS=(29533 29534 29531 29532)

# Clean ALL benchmark artifacts upfront
rm -f artifacts/checkpoints/bench_ddp_*.pth artifacts/checkpoints/bench_compile_*.pth 2>/dev/null || true
rm -f artifacts/tmp/bench_ddp_*.pth artifacts/tmp/bench_compile_*.pth 2>/dev/null || true
echo "Cleaned all benchmark artifacts"

for i in "${!CONFIGS[@]}"; do
    CFG="${CONFIGS[$i]}"
    PORT="${PORTS[$i]}"
    LOG="/tmp/bench_${CFG}.log"

    echo "========================================"
    echo "RUN $((i+1))/4: $CFG"
    echo "========================================"

    # Run
    MORTAL_CFG="configs/${CFG}.toml" CUDA_VISIBLE_DEVICES=0,1 \
        torchrun --standalone --nproc_per_node 2 --master-port "$PORT" \
        mortal/train_bc.py 2>&1 | tee "$LOG"

    # Extract final result
    echo ""
    echo "--- RESULT: $CFG ---"
    grep 'steps=500' "$LOG" || echo "(no step=500 line found)"
    echo ""

    # Clean this run's checkpoints so they don't interfere
    grep -oP "(?<=')[^']*bench_[^']*\.pth" "configs/${CFG}.toml" | xargs rm -f 2>/dev/null || true

    # Wait for GPU memory to release
    sleep 5
done

echo "========================================"
echo "ALL DONE. Summary:"
echo "========================================"
for CFG in "${CONFIGS[@]}"; do
    LOG="/tmp/bench_${CFG}.log"
    echo "--- $CFG ---"
    grep 'steps=500' "$LOG" 2>/dev/null | grep -oP 'samples_per_s=[\d.]+|mem_resv_gib=[\d.]+|loader_wait=[\d.]+|fwbw=[\d.]+|ddp_sync=[\d.]+' || echo "(not found)"
    echo ""
done
