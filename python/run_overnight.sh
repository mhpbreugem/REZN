#!/bin/bash
# Runs forward sweep → backward sweep → asymmetric sweep → plots, in series.
# All results go to G=11 logit-PCHIP CSVs and the existing plot scripts.

set -e
cd /home/user/REZN/python

echo "=== STAGE 1: forward G=11 logit-PCHIP sweep ===" | tee -a run_overnight.log
date | tee -a run_overnight.log
python3 -u pchip_continuation.py >> run_overnight.log 2>&1
echo "[stage 1 done $(date)]" | tee -a run_overnight.log

# Update plot_branches.py to read from G=11 logit CSV (or copy snapshot)
cp pchip_G9_forward.csv pchip_continuation_results.csv

echo "=== STAGE 2: backward post-jump sweep ===" | tee -a run_overnight.log
date | tee -a run_overnight.log
python3 -u pchip_backward_sweep.py >> run_overnight.log 2>&1 || \
  echo "[backward failed but continuing]" | tee -a run_overnight.log
echo "[stage 2 done $(date)]" | tee -a run_overnight.log

echo "=== STAGE 3: asymmetric γ=(5,3,1) sweep ===" | tee -a run_overnight.log
date | tee -a run_overnight.log
python3 -u pchip_asymmetric_sweep.py >> run_overnight.log 2>&1 || \
  echo "[asymmetric failed but continuing]" | tee -a run_overnight.log
echo "[stage 3 done $(date)]" | tee -a run_overnight.log

echo "=== STAGE 4: plots ===" | tee -a run_overnight.log
date | tee -a run_overnight.log
python3 -u plot_paper.py >> run_overnight.log 2>&1
python3 -u plot_branches.py >> run_overnight.log 2>&1
echo "[stage 4 done $(date)]" | tee -a run_overnight.log

echo "=== ALL STAGES COMPLETE $(date) ===" | tee -a run_overnight.log

# Commit and push the final plots
cd /home/user/REZN
git add python/plot_*.png python/pchip_G9_forward.csv python/pchip_continuation_results.csv python/pchip_backward_results.csv python/pchip_asymmetric_results.csv 2>/dev/null || true
git commit -m "Overnight run: G=11 logit-PCHIP sweeps + smooth plots" 2>/dev/null || true
for i in 1 2 3 4; do
  git push origin claude/rarar-without-nt-I8tiz 2>&1 && break
  sleep $((2 ** i))
done
echo "[push done $(date)]" >> python/run_overnight.log
