# Reference approach

The solution covers four independently checked areas:

1. `model_checks`: piecewise-linear OCV and analytic slope, exponential temperature scaling for both resistances, exact variable-step RC propagation, and the terminal-voltage sign convention.
2. `prediction_checks`: seconds-to-Ah conversion, charge efficiency, actual time intervals, and covariance propagation through `diag(1, a)`.
3. `update_checks`: the two-state voltage Jacobian, coupled Kalman gain, SOC bounds, and Joseph covariance update.
4. `cli_checks`: first-row handling, strict input validation, output schema, full-precision summary metrics, and end-to-end irregular telemetry.

All four groups pass with the reference files installed, producing a total score of 1.0.
