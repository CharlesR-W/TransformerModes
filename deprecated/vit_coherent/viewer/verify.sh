#!/usr/bin/env bash
# Screenshot the main views and linked interactions of a built viewer, and print console errors.
#   bash verify.sh out/dinov2s /tmp/vc_shots [port]
# Each shot is a URL-hash state (the page keeps all of its state in the hash).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$(cd "$HERE" && cd "${1:-out/dinov2s}" && pwd)
SHOTS=${2:-/tmp/vc_shots}
PORT=${3:-8765}
mkdir -p "$SHOTS"
python3 -m http.server "$PORT" --bind 127.0.0.1 -d "$OUT" >/dev/null 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; playwright-cli -s=vcheck close >/dev/null 2>&1 || true' EXIT
sleep 1
cd /tmp
P="playwright-cli -s=vcheck"
U="http://127.0.0.1:$PORT/"
$P open "$U" --browser=chrome >/dev/null
$P resize 1440 900 >/dev/null
shot() { # hash name [wait] [extra js]
  $P goto "${U}#$1" >/dev/null; $P reload >/dev/null; sleep "${3:-5}"
  if [ -n "${4:-}" ]; then $P eval "$4" >/dev/null; sleep 3; fi
  $P screenshot --filename="$SHOTS/$2.png" --full-page >/dev/null && echo "shot $SHOTS/$2.png"
}
K=${K:-}
shot "tab=layer&l=9&t=1&r=16" 01_layer_empty 6
shot "tab=layer&l=9&t=1&r=16&sel=mode:cls:9:1" 02_layer_mode 7
shot "tab=layer&l=9&t=1&r=16&sel=state:cls:9:3&img=1234" 03_layer_state 7
shot "tab=layer&l=9&t=1&r=16&sel=neuron:9:100" 04_layer_neuron 9
shot "tab=layer&l=7&t=1&r=8&sel=head:7:2&c=state" 05_layer_head_states 8
shot "tab=block&l=9&t=1&r=16&a=1&b=12&sel=dmode:cls:1:12:0" 06_block_dmode 9
shot "tab=depth&l=6&t=1&r=16&a=6&b=9&sel=state:cls:6:2" 07_depth 9
shot "tab=diag&l=9&t=1&r=16" 08_diag 7
shot "tab=defs" 09_defs 4
shot "tab=layer&l=3&t=1&r=16&sel=mode:cls:3:0" 10_layer_mode_early 8
shot "tab=block&k=st7&l=9&t=1&r=16&a=1&b=2&sel=dmode:cls:1:2:0" 11_block_st7_adjacent 9
shot "tab=layer&l=9&t=16&r=16&sel=mode:cls:9:0" 12_degenerate_t 7
echo "--- console errors ---"
$P console error || true
