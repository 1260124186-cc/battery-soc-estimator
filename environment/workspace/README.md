# Battery telemetry estimator

The package in `battery_estimator/` reads battery telemetry and estimates state of charge and polarization voltage. Run it with:

```bash
python -m battery_estimator.cli data/sample.csv output.csv --summary summary.json
```
