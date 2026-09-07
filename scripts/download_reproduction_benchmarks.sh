#!/usr/bin/env bash
set -euo pipefail

repo_dir=${1:-$(pwd)}
output_dir=${2:-$repo_dir/benchmark-originals}
record_id=22650001
base_url="https://zenodo.org/api/records/$record_id/files"
manifest=benchmark-originals-20260907.sha256

files=(
  M6.edges
  cage15.edges
  des_perf.edges
  leon2.edges
  leon3mp.edges
  netcard.edges
  nlpkkt120.edges
  vga_lcd.edges
)

mkdir -p "$output_dir"

download() {
  local name=$1
  local force=${2:-0}
  local destination="$output_dir/$name"
  local partial="$destination.part"
  if [[ "$force" != 1 && -s "$destination" ]]; then
    printf 'download_present file=%s bytes=%s\n' \
      "$name" "$(stat -c '%s' "$destination")"
    return
  fi
  printf 'download_start file=%s\n' "$name"
  if [[ "$force" == 1 ]]; then
    curl --fail --location --show-error \
      --retry 8 --retry-delay 10 --retry-all-errors \
      --output "$partial" "$base_url/$name/content"
  else
    curl --fail --location --show-error \
      --retry 8 --retry-delay 10 --retry-all-errors \
      --continue-at - --output "$partial" "$base_url/$name/content"
  fi
  mv "$partial" "$destination"
  printf 'download_done file=%s bytes=%s\n' \
    "$name" "$(stat -c '%s' "$destination")"
}

download "$manifest" 1

for name in "${files[@]}"; do
  expected=$(awk -v name="$name" '$2 == name { print $1 }' \
    "$output_dir/$manifest")
  [[ -n "$expected" ]] || {
    printf 'manifest_missing file=%s\n' "$name" >&2
    exit 1
  }
  if [[ -s "$output_dir/$name" ]] \
      && printf '%s  %s\n' "$expected" "$output_dir/$name" \
        | sha256sum --check --status; then
    printf 'checksum_cached file=%s\n' "$name"
    continue
  fi
  if [[ -e "$output_dir/$name" ]]; then
    stale="$output_dir/$name.invalid.$(date +%Y%m%d_%H%M%S)"
    mv "$output_dir/$name" "$stale"
    printf 'checksum_stale file=%s saved_as=%s\n' "$name" "$stale"
  fi
  download "$name"
  printf '%s  %s\n' "$expected" "$output_dir/$name" \
    | sha256sum --check
done

(
  cd "$output_dir"
  sha256sum --check "$manifest"
)

printf 'benchmark_download_complete files=%d directory=%s doi=%s\n' \
  "${#files[@]}" "$output_dir" '10.5281/zenodo.22650001'
