### Primary endpoint and per-bucket accuracy

| group | params | steps | PRIMARY (D1-D3) | sd | D0 | D1 | D2 | D3 | D4 (alarm) |
|---|---|---|---|---|---|---|---|---|---|
| **A FIXED SMALL** | 81k | 8,854 | **0.7960** | 0.0528 | 0.988 | 0.997 | 0.998 | 0.393 | 0.500 |
| **B FIXED LARGE** | 178k | 3,909 | **0.8335** | 0.0143 | 0.980 | 0.993 | 0.995 | 0.513 | 0.530 |
| **C GROWTH ONLY** | 178k | 5,864 | **0.7934** | 0.0495 | 0.984 | 0.996 | 0.998 | 0.387 | 0.495 |
| **D DEVELOPMENTAL** | 178k | 5,236 | **0.7944** | 0.0514 | 0.982 | 0.994 | 0.999 | 0.390 | 0.515 |
| **D-RANDPRUNE** | 191k | 5,157 | **0.7897** | 0.0538 | 0.983 | 0.995 | 0.997 | 0.377 | 0.513 |
| **R RANDOM DEV** | 168k | 5,207 | **0.7843** | 0.0535 | 0.983 | 0.997 | 0.998 | 0.358 | 0.494 |
| _majority baseline_ | | | _0.428_ | | _0.340_ | _0.287_ | _0.501_ | _0.497_ | _0.545_ |

### Compute accounting (all groups at an identical FLOPs budget)

| group | train FLOPs | steps | tokens | wall s | peak RSS MB | inference FLOPs/token | accuracy / PFLOP |
|---|---|---|---|---|---|---|---|
| A FIXED SMALL | 2.000e+12 | 8,854 | 4.53M | 177 | 717 | 147,072 | 397.9 |
| B FIXED LARGE | 2.000e+12 | 3,909 | 2.00M | 162 | 723 | 339,686 | 416.7 |
| C GROWTH ONLY | 2.000e+12 | 5,864 | 3.00M | 176 | 723 | 339,686 | 396.6 |
| D DEVELOPMENTAL | 2.000e+12 | 5,236 | 2.68M | 182 | 737 | 339,686 | 397.2 |
| D-RANDPRUNE | 2.000e+12 | 5,157 | 2.64M | 182 | 737 | 365,901 | 394.8 |
| R RANDOM DEV | 2.000e+12 | 5,207 | 2.67M | 169 | 722 | 320,026 | 392.1 |

### Calibration and novelty

| group | info gain (bits/pred) | Brier | ECE | D4 leakage z | leakage flag |
|---|---|---|---|---|---|
| A FIXED SMALL | 1.405 | 0.2333 | 0.1138 | -1.54 | no |
| B FIXED LARGE | 1.419 | 0.1922 | 0.0872 | -0.59 | no |
| C GROWTH ONLY | 1.400 | 0.2368 | 0.1131 | -1.69 | no |
| D DEVELOPMENTAL | 1.399 | 0.2330 | 0.1096 | -1.08 | no |
| D-RANDPRUNE | 1.397 | 0.2394 | 0.1138 | -1.13 | no |
| R RANDOM DEV | 1.396 | 0.2474 | 0.1184 | -1.72 | no |

### Paired differences in the primary endpoint, across seeds

| comparison | mean diff | 95% CI (paired bootstrap) | Cohen's d | seeds won |
|---|---|---|---|---|
| D - A | -0.0016 | [-0.0074, +0.0048] | -0.21 | 1/5 |
| D - B | -0.0392 | [-0.0856, +0.0072] | -0.64 | 2/5 |
| D - C | +0.0010 | [-0.0009, +0.0038] | +0.31 | 3/5 |
| D - R | +0.0101 | [+0.0052, +0.0159] | +1.42 | 5/5 |
| C - A | -0.0026 | [-0.0088, +0.0039] | -0.31 | 2/5 |
| C - R | +0.0091 | [+0.0044, +0.0156] | +1.26 | 5/5 |
| D - D_RANDPRUNE | +0.0047 | [-0.0010, +0.0093] | +0.71 | 4/5 |
| B - A | +0.0376 | [-0.0109, +0.0831] | +0.61 | 3/5 |

### Per-seed primary endpoint

| group | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| A FIXED SMALL | 0.7359 | 0.8437 | 0.8588 | 0.7751 | 0.7662 |
| B FIXED LARGE | 0.8562 | 0.8321 | 0.8243 | 0.8361 | 0.8191 |
| C GROWTH ONLY | 0.7373 | 0.8336 | 0.8559 | 0.7646 | 0.7756 |
| D DEVELOPMENTAL | 0.7357 | 0.8398 | 0.8560 | 0.7641 | 0.7762 |
| D-RANDPRUNE | 0.7349 | 0.8314 | 0.8609 | 0.7540 | 0.7672 |
| R RANDOM DEV | 0.7260 | 0.8289 | 0.8519 | 0.7598 | 0.7546 |

### Is the grown capacity load-bearing?

| group | growth events | grown share of total ablation effect | worst accuracy drop after growth |
|---|---|---|---|
| A FIXED SMALL | 0.0 | 0.0000% | +0.0000 |
| B FIXED LARGE | 0.0 | 0.0000% | +0.0000 |
| C GROWTH ONLY | 3.8 | 0.2664% | +0.0165 |
| D DEVELOPMENTAL | 3.2 | 0.2205% | +0.0206 |
| D-RANDPRUNE | 3.4 | 0.2641% | +0.0321 |
| R RANDOM DEV | 3.2 | 1.8917% | +0.0305 |
