#!/bin/bash
# Summary table emitter — runs every 5 minutes and dumps a snapshot
# of the overnight chain progress.
while true; do
  date_str=$(date +"%H:%M:%S")
  echo "============================================================"
  echo " STATUS @ $date_str"
  echo "============================================================"
  # which batch is running
  running=$(pgrep -af "strict_PR_search" | grep -v grep || true)
  if [ -z "$running" ]; then
    echo "  (no strict_PR_search process — chain advancing or done)"
  else
    echo "  running: $(echo "$running" | sed 's|.*python3 -u ||')"
  fi
  echo "  --- last 8 result lines from overnight.log ---"
  tail -8 /home/user/REZN/python/overnight.log 2>/dev/null \
      | grep -v "^=== \|^cfg " \
      | sed 's|^|  |'
  # PR seeds saved so far
  saved=$(ls /home/user/REZN/python/PR_seed_*.pkl 2>/dev/null | wc -l)
  echo "  PR seeds saved so far: $saved"
  echo
  sleep 300        # 5 minutes
done
