#!/usr/bin/env bash
set -euo pipefail

# --- edit these to your paths/preferences ---
ENV_PATH="/depot/chan129/users/tharindu/virtual_envs/wan22"

# Change between 5B <-> A14B
SAVE_PATH="./results/wan22_A14B"
CKPT_DIR="./Wan2.2-T2V-A14B"
TASK="t2v-A14B"

# 832*480 for A14B 1280*704 for 5B
SIZE="832*480"      # keep quoted; '*' would glob otherwise
FRAME_NUM=81                 
OFFLOAD="True"
EXTRA_FLAGS="--convert_model_dtype"   # add e.g. --t5_cpu if needed
DEFAULT_PROMPTS_FILE="prompts.txt"
# -------------------------------------------

# Parse args
PROMPTS_FILE="$DEFAULT_PROMPTS_FILE"
LINE=""                # run only this line number if set
SINGLE_PROMPT=""       # run only this prompt if set

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts|-p) PROMPTS_FILE="$2"; shift 2 ;;
    --line|-l)    LINE="$2"; shift 2 ;;
    --prompt)     SINGLE_PROMPT="$2"; shift 2 ;;
    -h|--help)
      echo "Usage:"
      echo "  $0 --prompt \"your prompt here\""
      echo "  $0 --prompts prompts.txt               # run all lines"
      echo "  $0 --prompts prompts.txt --line N      # run line N only"
      echo "  (If SLURM_ARRAY_TASK_ID is set, it will use that as --line)"
      exit 0 ;;
    *)
      echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo ">>> Using prompts file: $PROMPTS_FILE"
echo ">>> Non-empty, non-comment prompt count: $(grep -E '^[[:space:]]*[^#[:space:]].*' -c "$PROMPTS_FILE")"

# Activate your conda env (cluster modules style)
if command -v module >/dev/null 2>&1; then module load conda; fi
conda activate "$ENV_PATH"

mkdir -p runs logs

run_one() {
  local prompt="$1"
  local stamp="$(date +%Y%m%d_%H%M%S)_$RANDOM"
  printf '%s\n' "$prompt" > "runs/prompt_${stamp}.txt"
  echo ">>> $(date) | Running WAN2.2 (${TASK}) with checkpoint: ${CKPT_DIR}"
  echo ">>> Prompt: $prompt"
  python generate.py \
    --task "${TASK}" \
    --size "${SIZE}" \
    --ckpt_dir "${CKPT_DIR}" \
    --frame_num "${FRAME_NUM}" \
    --save_path "${SAVE_PATH}" \
    --offload_model "${OFFLOAD}" \
    ${EXTRA_FLAGS} \
    --prompt "${prompt}" | tee "runs/run_${stamp}.log"
  echo ">>> Done. Logs: runs/run_${stamp}.log | Prompt: runs/prompt_${stamp}.txt"
}

# Mode 1: single prompt supplied explicitly
if [[ -n "$SINGLE_PROMPT" ]]; then
  run_one "$SINGLE_PROMPT"
  exit 0
fi

# If running under a SLURM array and no --line was provided, use the array index
if [[ -z "$LINE" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  LINE="$SLURM_ARRAY_TASK_ID"
fi

# Check prompts file if needed
if [[ ! -f "$PROMPTS_FILE" ]]; then
  echo "Prompt file '$PROMPTS_FILE' not found. Use --prompts FILE or --prompt \"text\"." >&2
  exit 1
fi

# Mode 2: run only a specific line
if [[ -n "$LINE" ]]; then
  prompt="$(sed -n "${LINE}p" "$PROMPTS_FILE")"
  if [[ -z "${prompt// }" || "${prompt:0:1}" == "#" ]]; then
    echo "Line $LINE is empty or a comment in '$PROMPTS_FILE'; nothing to do."
    exit 0
  fi
  run_one "$prompt"
  exit 0
fi

# Mode 3: all non-empty, non-comment lines (CR-safe, newline-safe)
lineno=0
set +e
while IFS= read -r prompt || [[ -n "$prompt" ]]; do
  ((lineno++))
  prompt="${prompt%$'\r'}"   # strip Windows CR if present
  [[ -z "${prompt//[[:space:]]/}" || "${prompt:0:1}" == "#" ]] && continue
  run_one "$prompt"
done < "$PROMPTS_FILE"
set -e

