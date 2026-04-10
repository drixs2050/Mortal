#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DATE="${RUN_DATE:-$(date -I)}"
END_DATE="${END_DATE:-$RUN_DATE}"
END_DATE_EXPLICIT=0
MIN_DATE=""
SERIES_PREFIX="phoenix_batch_series"
USAGE_PREFIX="corpus-expansion"
BATCH_COUNT=16
WINDOW_DAYS=182
LIMIT=100000
RULESET="四鳳南喰赤－"
CONVERTER_VERSION="tenhou-xml-v0"
CONVERTER_SOURCE="xml"
ARCHIVE_JOBS=8
STAGE_JOBS=16
DOWNLOAD_TIMEOUT=30
DOWNLOAD_RETRIES=2
RETRY_BACKOFF_SECONDS=1.0
ARCHIVE_PUBLISH_LAG_DAYS=14
YEAR_ARCHIVE_CACHE_DIR=""
STOP_AFTER="ingest"
START_BATCH=1
WITH_RELEASE_ARTIFACTS=0
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_tenhou_pipeline_series.sh [options]

Description:
  Launch a recent-first series of Tenhou Phoenix archive collection batches.
  Each batch covers a fixed date window and runs scripts/run_tenhou_pipeline.py.

Options:
  --run-date YYYY-MM-DD        Snapshot prefix date. Default: today.
  --end-date YYYY-MM-DD        Most recent archive date in the first batch. Default: --run-date minus --archive-publish-lag-days.
  --min-date YYYY-MM-DD        Optional oldest allowed archive date. Stops once the next batch would end before this date.
  --series-prefix NAME         Batch name prefix. Default: phoenix_batch_series
  --usage-prefix NAME          usage_status prefix. Default: corpus-expansion
  --batches N                  Number of batches to launch. Default: 16
  --window-days N              Days per batch window. Default: 182
  --limit N                    Replay cap per batch. Default: 100000
  --ruleset LABEL              Tenhou archive ruleset label. Default: 四鳳南喰赤－
  --converter-version NAME     Converter version string. Default: tenhou-xml-v0
  --converter-source MODE      Converter source: auto|official_json|xml. Default: xml
  --archive-jobs N             Concurrent archive fetch jobs. Default: 8
  --stage-jobs N               Concurrent replay staging jobs. Default: 16
  --download-timeout SECONDS   Network timeout for archive and replay downloads. Default: 30
  --download-retries N         Retry attempts for transient download failures. Default: 2
  --retry-backoff-seconds S    Base exponential backoff in seconds. Default: 1.0
  --archive-publish-lag-days N Treat recent archive 404s inside this lag window as unpublished. Default: 14
  --year-archive-cache-dir DIR Cache directory for old yearly Tenhou archive zips. Default: fetch-script default
  --stop-after STEP            Pipeline stop step: fetch|select|stage|ingest|release. Default: ingest
  --start-batch N              1-based batch index to start from. Default: 1
  --with-release-artifacts     Also run QA/split/overlap per batch.
  --overwrite                  Pass --overwrite to each pipeline batch.
  --dry-run                    Print planned commands without executing them.
  -h, --help                   Show this help.

Naming:
  Batch 1 uses the most recent window. Later batches walk backward in time.
  Snapshot ids look like:
    <run-date>_<series-prefix><NN>
  Dataset ids look like:
    tenhou_<series-prefix><NN>_v0
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

iso_date_or_die() {
  local raw="$1"
  date -I -d "$raw" >/dev/null 2>&1 || die "invalid date: $raw"
}

parse_args() {
  while (($# > 0)); do
    case "$1" in
      --run-date)
        RUN_DATE="$2"
        shift 2
        ;;
      --end-date)
        END_DATE="$2"
        END_DATE_EXPLICIT=1
        shift 2
        ;;
      --min-date)
        MIN_DATE="$2"
        shift 2
        ;;
      --series-prefix)
        SERIES_PREFIX="$2"
        shift 2
        ;;
      --usage-prefix)
        USAGE_PREFIX="$2"
        shift 2
        ;;
      --batches)
        BATCH_COUNT="$2"
        shift 2
        ;;
      --window-days)
        WINDOW_DAYS="$2"
        shift 2
        ;;
      --limit)
        LIMIT="$2"
        shift 2
        ;;
      --ruleset)
        RULESET="$2"
        shift 2
        ;;
      --converter-version)
        CONVERTER_VERSION="$2"
        shift 2
        ;;
      --converter-source)
        CONVERTER_SOURCE="$2"
        shift 2
        ;;
      --archive-jobs)
        ARCHIVE_JOBS="$2"
        shift 2
        ;;
      --stage-jobs)
        STAGE_JOBS="$2"
        shift 2
        ;;
      --download-timeout)
        DOWNLOAD_TIMEOUT="$2"
        shift 2
        ;;
      --download-retries)
        DOWNLOAD_RETRIES="$2"
        shift 2
        ;;
      --retry-backoff-seconds)
        RETRY_BACKOFF_SECONDS="$2"
        shift 2
        ;;
      --archive-publish-lag-days)
        ARCHIVE_PUBLISH_LAG_DAYS="$2"
        shift 2
        ;;
      --year-archive-cache-dir)
        YEAR_ARCHIVE_CACHE_DIR="$2"
        shift 2
        ;;
      --stop-after)
        STOP_AFTER="$2"
        shift 2
        ;;
      --start-batch)
        START_BATCH="$2"
        shift 2
        ;;
      --with-release-artifacts)
        WITH_RELEASE_ARTIFACTS=1
        shift
        ;;
      --overwrite)
        OVERWRITE=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        die "unknown argument: $1"
        ;;
    esac
  done
}

validate_args() {
  iso_date_or_die "$RUN_DATE"
  iso_date_or_die "$END_DATE"
  if [[ -n "$MIN_DATE" ]]; then
    iso_date_or_die "$MIN_DATE"
  fi

  [[ "$BATCH_COUNT" =~ ^[0-9]+$ ]] || die "--batches must be a positive integer"
  [[ "$WINDOW_DAYS" =~ ^[0-9]+$ ]] || die "--window-days must be a positive integer"
  [[ "$LIMIT" =~ ^[0-9]+$ ]] || die "--limit must be a non-negative integer"
  [[ "$START_BATCH" =~ ^[0-9]+$ ]] || die "--start-batch must be a positive integer"
  [[ "$ARCHIVE_JOBS" =~ ^[0-9]+$ ]] || die "--archive-jobs must be a positive integer"
  [[ "$STAGE_JOBS" =~ ^[0-9]+$ ]] || die "--stage-jobs must be a positive integer"
  [[ "$DOWNLOAD_RETRIES" =~ ^[0-9]+$ ]] || die "--download-retries must be a non-negative integer"
  [[ "$ARCHIVE_PUBLISH_LAG_DAYS" =~ ^[0-9]+$ ]] || die "--archive-publish-lag-days must be a non-negative integer"

  (( BATCH_COUNT >= 1 )) || die "--batches must be at least 1"
  (( WINDOW_DAYS >= 1 )) || die "--window-days must be at least 1"
  (( START_BATCH >= 1 )) || die "--start-batch must be at least 1"
  (( START_BATCH <= BATCH_COUNT )) || die "--start-batch must be <= --batches"
  (( ARCHIVE_JOBS >= 1 )) || die "--archive-jobs must be at least 1"
  (( STAGE_JOBS >= 1 )) || die "--stage-jobs must be at least 1"

  case "$CONVERTER_SOURCE" in
    auto|official_json|xml) ;;
    *)
      die "--converter-source must be one of: auto, official_json, xml"
      ;;
  esac

  case "$STOP_AFTER" in
    fetch|select|stage|ingest|release) ;;
    *)
      die "--stop-after must be one of: fetch, select, stage, ingest, release"
      ;;
  esac
}

date_minus_days() {
  local raw_date="$1"
  local days="$2"
  date -I -d "$raw_date - $days days"
}

build_batch_window() {
  local batch_index="$1"
  local newest_end="$2"
  local min_date="$3"

  local batch_offset_days=$(( (batch_index - 1) * WINDOW_DAYS ))
  local batch_end
  local batch_start

  batch_end="$(date_minus_days "$newest_end" "$batch_offset_days")"
  batch_start="$(date_minus_days "$batch_end" "$((WINDOW_DAYS - 1))")"

  if [[ -n "$min_date" ]]; then
    if [[ "$batch_end" < "$min_date" ]]; then
      return 1
    fi
    if [[ "$batch_start" < "$min_date" ]]; then
      batch_start="$min_date"
    fi
  fi

  printf '%s %s\n' "$batch_start" "$batch_end"
}

run_batch() {
  local batch_index="$1"
  local batch_start="$2"
  local batch_end="$3"

  local batch_suffix
  batch_suffix="$(printf '%02d' "$batch_index")"

  local batch_slug="${SERIES_PREFIX}${batch_suffix}"
  local snapshot_id="${RUN_DATE}_${batch_slug}"
  local dataset_id="tenhou_${batch_slug}_v0"
  local archive_dir="/tmp/tenhou_scc_${batch_slug}"
  local usage_status="${USAGE_PREFIX}-${batch_slug}"

  local -a cmd=(
    "$PYTHON_BIN"
    "scripts/run_tenhou_pipeline.py"
    "--snapshot-id" "$snapshot_id"
    "--dataset-id" "$dataset_id"
    "--start-date" "$batch_start"
    "--end-date" "$batch_end"
    "--archive-dir" "$archive_dir"
    "--ruleset" "$RULESET"
    "--limit" "$LIMIT"
    "--archive-jobs" "$ARCHIVE_JOBS"
    "--stage-jobs" "$STAGE_JOBS"
    "--download-timeout" "$DOWNLOAD_TIMEOUT"
    "--download-retries" "$DOWNLOAD_RETRIES"
    "--retry-backoff-seconds" "$RETRY_BACKOFF_SECONDS"
    "--archive-publish-lag-days" "$ARCHIVE_PUBLISH_LAG_DAYS"
    "--usage-status" "$usage_status"
    "--converter-version" "$CONVERTER_VERSION"
    "--converter-source" "$CONVERTER_SOURCE"
    "--stop-after" "$STOP_AFTER"
  )

  if [[ -n "$YEAR_ARCHIVE_CACHE_DIR" ]]; then
    cmd+=("--year-archive-cache-dir" "$YEAR_ARCHIVE_CACHE_DIR")
  fi

  if (( WITH_RELEASE_ARTIFACTS )); then
    cmd+=("--with-release-artifacts")
  fi
  if (( OVERWRITE )); then
    cmd+=("--overwrite")
  fi
  if (( DRY_RUN )); then
    cmd+=("--dry-run")
  fi

  echo
  echo "=== Batch ${batch_index}/${BATCH_COUNT}: ${batch_start} -> ${batch_end} ==="
  echo "snapshot_id=${snapshot_id}"
  echo "dataset_id=${dataset_id}"
  printf 'command='
  printf '%q ' "${cmd[@]}"
  echo

  (
    cd "$ROOT"
    "${cmd[@]}"
  )
}

main() {
  parse_args "$@"
  if (( ! END_DATE_EXPLICIT )); then
    END_DATE="$(date_minus_days "$RUN_DATE" "$ARCHIVE_PUBLISH_LAG_DAYS")"
  fi
  validate_args

  echo "Series config:"
  echo "  run_date=${RUN_DATE}"
  echo "  end_date=${END_DATE}"
  echo "  min_date=${MIN_DATE:-<none>}"
  echo "  batches=${BATCH_COUNT}"
  echo "  window_days=${WINDOW_DAYS}"
  echo "  limit=${LIMIT}"
  echo "  ruleset=${RULESET}"
  echo "  archive_jobs=${ARCHIVE_JOBS}"
  echo "  stage_jobs=${STAGE_JOBS}"
  echo "  download_timeout=${DOWNLOAD_TIMEOUT}"
  echo "  download_retries=${DOWNLOAD_RETRIES}"
  echo "  retry_backoff_seconds=${RETRY_BACKOFF_SECONDS}"
  echo "  archive_publish_lag_days=${ARCHIVE_PUBLISH_LAG_DAYS}"
  echo "  year_archive_cache_dir=${YEAR_ARCHIVE_CACHE_DIR:-<default>}"
  echo "  start_batch=${START_BATCH}"
  echo "  stop_after=${STOP_AFTER}"
  echo "  with_release_artifacts=${WITH_RELEASE_ARTIFACTS}"
  echo "  overwrite=${OVERWRITE}"
  echo "  dry_run=${DRY_RUN}"

  local launched=0
  local batch_window
  local batch_start
  local batch_end

  for ((batch_index = START_BATCH; batch_index <= BATCH_COUNT; batch_index++)); do
    if ! batch_window="$(build_batch_window "$batch_index" "$END_DATE" "$MIN_DATE")"; then
      echo
      echo "Stopping before batch ${batch_index}: next batch window would be older than min_date=${MIN_DATE}."
      break
    fi

    read -r batch_start batch_end <<<"$batch_window"
    run_batch "$batch_index" "$batch_start" "$batch_end"
    launched=$((launched + 1))
  done

  echo
  echo "Finished. Batches launched: ${launched}"
}

main "$@"
