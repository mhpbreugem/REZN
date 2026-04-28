#!/bin/bash
# Sleep-time chain: wait for v3 to finish, then run v4 (linear kernel).
# Started after user went to sleep with the request to "keep going".
set -u
cd /home/user/REZN/python

# Wait for v3 (already launched at PID 25548) to exit
echo "=== sleep_chain: waiting for v3 to exit ===" >> sleep_chain.log
while pgrep -f precision_test_b9_v3 > /dev/null; do
    sleep 30
done
echo "=== v3 done at $(date) ===" >> sleep_chain.log

# Now run v4 (linear kernel)
python3 -u precision_test_b9_v4_linear.py >> b9v4.log 2>&1
echo "=== v4 done at $(date) ===" >> sleep_chain.log
echo "ALL SLEEP-CHAIN BATCHES DONE" >> sleep_chain.log
