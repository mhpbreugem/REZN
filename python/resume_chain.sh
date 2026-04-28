#!/bin/bash
set -u
cd /home/user/REZN/python

# Batch 7 — long-Newton retry on the most promising configs
python3 -u strict_PR_search_7.py >> overnight.log 2>&1
echo "=== batch 7 done at $(date) ===" >> overnight.log

# Then resume batches 4, 5, 6 (skip 3 — too slow, no gains likely)
python3 -u strict_PR_search_4.py >> overnight.log 2>&1
echo "=== batch 4 done at $(date) ===" >> overnight.log

python3 -u strict_PR_search_5.py >> overnight.log 2>&1
echo "=== batch 5 done at $(date) ===" >> overnight.log

python3 -u strict_PR_search_6.py >> overnight.log 2>&1
echo "=== batch 6 done at $(date) ===" >> overnight.log

echo "ALL BATCHES (7, 4, 5, 6) DONE at $(date)" >> overnight.log
