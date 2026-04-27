#!/bin/bash
set -u
cd /home/user/REZN/python

# Wait for any running search
while pgrep -f strict_PR_search >/dev/null 2>&1; do sleep 30; done
echo "=== batch 1 done at $(date) ===" >> overnight.log

python3 -u strict_PR_search_2.py >> overnight.log 2>&1
echo "=== batch 2 done at $(date) ===" >> overnight.log

python3 -u strict_PR_search_3.py >> overnight.log 2>&1
echo "=== batch 3 done at $(date) ===" >> overnight.log

echo "ALL BATCHES DONE at $(date)" >> overnight.log
