# Joint estimator (raw) — T1-T3 + HJ1

## T1 recovery (truth: theta1=0.05, theta2=1.5, RC=10, kappa=0.2)
theta1=0.0536, theta2=1.264, rc=9.987,
kappa=0.138; converged=True;
kappa profile argmax on grid: 0.144.

## T2 corner (truth kappa = 0)
kappa_hat=0.006, theta2=1.496;
kappa-profile range 3803.6 log-lik units over [0.02, 0.6].

## T3 cost
150 likelihood evaluations, 328s total
(2.18 s/eval).

## HJ1 — coherent deep vs as-if (E3 benchmark)
Deep wins in 5/5 markets.

| s | as-if RMSE (E3) | deep RMSE (joint) | m1 true | m1 hat |
|---|---|---|---|---|
| 0.5 | 0.02483 | 0.00282 | 0.275 | 0.325 |
| 0.75 | 0.01188 | 0.00327 | 0.425 | 0.500 |
| 1.0 | 0.00720 | 0.00542 | 0.500 | 0.500 |
| 1.5 | 0.01583 | 0.00318 | 0.550 | 0.625 |
| 2.0 | 0.02231 | 0.00320 | 0.675 | 0.750 |
