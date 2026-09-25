# Run report: pilot

- config hash `b872cfcc77dd92a3`  git `1f7161d4e806`
- world-family seal `e50ad237af389d10913d8e0b9f027382baa0c01988225971191b8bd138a36b92`  split {'A': 231, 'B': 77, 'C_dev': 1430, 'C_test': 1430}
- generations 40, seed 0
- Class C-test opened: **False**

## Population trajectory

| metric | first | last | trace |
|---|---|---|---|
| alive | 96 | 96 | `++++++++++++++++++++++++++++++++++++++++` |
| median params | 4910 | 8080 | `         :=#@@@@@@@@@*@@@%*==@*@@@@@@@@@` |
| min params | 863 | 467 | `#=@=====#+++--:::::::..:.:...:.  :.::...` |
| max params | 5.974e+04 | 1.173e+05 | `:::.:::=....... ..:=====@@++**%%%%%%%%%%` |
| mean score | 0.019 | 0.2063 | ` .....:-.:-++=+=*-=+=*+*#%%*%*@#*=%%*@##` |
| max score | 0.1744 | 0.408 | `.  :. -+.-=#--+-*-=#-+-+++*=@=*==-%+-*=+` |
| in-context gain | 0.01259 | 0.02158 | `=.-=.:==.*=:+ *.+*:+:-**@*=####-*+#@===*` |
| species | 15 | 32 | `.  ..::--=====+++*+++==+**++=*#*##@@##*#` |
| archive cells | 7 | 43 | `  .::-==++*****#####%%%%%%%%%%%%%%%@@@@@` |
| growth-bias gene | 0.5 | 0.4334 | `%%%%*+**#%**#%@@%%#*+:=-=:.        ::---` |
| structural-mutation gene | 0.15 | 0.1753 | `..... . .-::=++=+=+*%#*%@*-*#++*+***+*++` |
| temperature gene | 1 | 0.1836 | `@@@@%**+++==--------:...                ` |

## Complexity: is the trend driven or passive?

- selected arm: **-0.00777** nats/generation in mean ln(params)
- neutral arm:  **-0.01115** nats/generation
- **selected minus neutral: +0.00339 nats/generation**

Only the difference is interpretable. A positive selected-arm number on its own is consistent with passive diffusion off the lower size bound.

McShea's driven-trend test asks whether the *minimum* of the distribution moves, not just the mean:

- min params 863 -> 467 (slope -9.60/gen)
- median params 4910 -> 8080 (slope +83.05/gen)

A rising median with a stationary minimum is the signature of **passive diffusion**, not a driven trend.

### Subclade test (displaced founders)

- lineages founded large (>20K): now mean 40723 params
- lineages founded small: now mean 3056 params

Under a **driven** trend, displaced lineages keep growing. Under **passive diffusion** they regress toward the bulk.

## Adaptation: the evolvability decomposition

Slope of adaptation AUC against generation, per class and transplant condition. The hypothesis requires `full` to beat `no_feedback` and `capacity_matched`.

| class | capacity_matched | fixed_hparams | full | no_feedback | reinit_weights |
|---|---|---|---|---|---|
| A | +0.00024 | +0.00072 | +0.00067 | +0.00130 | +0.00011 |
| B | +0.00077 | +0.00139 | +0.00264 | +0.00311 | -0.00036 |
| C_dev | +0.00028 | +0.00111 | +0.00140 | +0.00231 | -0.00047 |

Final-generation AUC by class and condition:

| class | capacity_matched | fixed_hparams | full | no_feedback | reinit_weights |
|---|---|---|---|---|---|
| A | -0.0026 | +0.0377 | +0.0466 | +0.0618 | -0.0075 |
| B | +0.0391 | +0.1078 | +0.1843 | +0.1590 | +0.0034 |
| C_dev | -0.0056 | +0.1187 | +0.1546 | +0.1562 | -0.0012 |

- **A**: feedback-attributable adaptation = `full - no_feedback` = -0.0151  (non-positive: improvement is a reactive prior, not in-context learning)
- **B**: feedback-attributable adaptation = `full - no_feedback` = +0.0253  (positive: the organism is using its own action/reward history)
- **C_dev**: feedback-attributable adaptation = `full - no_feedback` = -0.0016  (non-positive: improvement is a reactive prior, not in-context learning)

## Architecture and mutation

| operator | births | mean score | survived to adulthood |
|---|---|---|---|
| `narrow_head` | 95 | +0.1226 | 53/95 (56%) |
| `contract_ffn` | 83 | +0.1210 | 48/83 (58%) |
| `add_block` | 66 | +0.1124 | 37/66 (56%) |
| `narrow` | 82 | +0.1193 | 31/82 (38%) |
| `expand_ffn` | 53 | +0.1051 | 32/53 (60%) |
| `remove_block` | 52 | +0.1009 | 33/52 (63%) |
| `add_head` | 59 | +0.1273 | 33/59 (56%) |
| `widen` | 48 | +0.1157 | 26/48 (54%) |
| `widen_head` | 52 | +0.1236 | 26/52 (50%) |
| `remove_head` | 35 | +0.0977 | 27/35 (77%) |

- distinct architectures that ever existed: **246**
- organisms recorded: 7193
- extinctions: 3353

Figures written to `results/pilot`.
