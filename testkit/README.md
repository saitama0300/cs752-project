### Workflow

```bash
# generate NCU reports for model.py
(casio-torch) ➜  testkit git:(main) ✗ ./runall.sh

# use utility to generate reports
(casio-torch) ➜  cs752-project git:(main) ✗ python scripts/proc_kernels.py --input output/v100/test_model/ncu-10th-test_model-train-b1-raw.txt --rep-output rep.csv --all-output all.csv
```

