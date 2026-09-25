# Run report: pilot

- config hash `48dd5dd1f5331aa4`  git `8e8064590cfd`
- world-family seal `e50ad237af389d10913d8e0b9f027382baa0c01988225971191b8bd138a36b92`  split {'A': 231, 'B': 77, 'C_dev': 1430, 'C_test': 1430}
- generations 40, seed 0
- Class C-test opened: **False**

## Population trajectory

| metric | first | last | trace |
|---|---|---|---|
| alive | 96 | 96 | `++++++++++++++++++++++++++++++++++++++++` |
| median params | 4910 | 1055 | `@@@@@@@@@@###---------:........         ` |
| min params | 863 | 735 | `+.@@%++.%===+@@@@@###@....%%%+= - ::==::` |
| max params | 4.489e+04 | 1.438e+04 | `###@@####@@@@@**#*###*::::::::.    ....:` |
| mean score | 0.0214 | 0.1918 | ` .... ....:=++--------=+****#+**###@@@%%` |
| max score | 0.1934 | 0.3397 | `::. .   . -#@@=---=====***#*#***+*+*****` |
| in-context gain | 0.002629 | 0.01112 | `-:=  ..=.:--+*:-:---****++*@%###-=-%%#*+` |
| species | 12 | 20 | ` : .---=-+=+*++=+*#@##+*##+++#+*++==-:--` |
| archive cells | 7 | 47 | ` ..:::::-====+****###%%%%@@@@@@@@@@@@@@@` |
| growth-bias gene | 0.5 | 0.4352 | `@@@@%@###%##=. .....+++#++++*#==+++==---` |
| structural-mutation gene | 0.15 | 0.06191 | `%%##%%@@%###*#%*+%%%==::..:::.          ` |
| temperature gene | 1 | 0.2857 | `@@@@%%#**+--:....::::.....::.......     ` |
| stack activity (0 = inert) | 0.01769 | 0.1103 | `  :.:++#@@:====+=::::-:.:.#*#...%..*++**` |

### Health check: is the transformer stack doing anything?

- champion stack activity 0.0177 -> 0.1103 (total variation from ablating every block output path)

## Complexity: is the trend driven or passive?

- selected arm: **-0.04836** nats/generation in mean ln(params)
- neutral arm:  **-0.01115** nats/generation
- **selected minus neutral: -0.03720 nats/generation**

Only the difference is interpretable. A positive selected-arm number on its own is consistent with passive diffusion off the lower size bound.

McShea's driven-trend test asks whether the *minimum* of the distribution moves, not just the mean: a driven trend moves the whole distribution, passive diffusion off a lower bound moves only its upper part.

- min params 863 -> 735 (slope -4.93/gen)
- median params 4910 -> 1055 (slope -124.05/gen)

The median falls while the minimum is roughly stationary: the distribution is **compressing from above** rather than shifting wholesale.

### Effective vs raw parameters

- effective parameters 2210 -> 847 (slope -18.4/gen)
- effective fraction 45.0% -> 86.2% ` ## :  @`

Raw parameters fall while the effective fraction *rises*: organisms are becoming **denser**, shedding capacity that was doing nothing. This is the opposite of bloat.

### Subclade test (displaced founders)

- lineages founded large (>20K): now mean 39416 params
- lineages founded small: now mean 3028 params

Under a **driven** trend, displaced lineages keep growing. Under **passive diffusion** they regress toward the bulk.

## Adaptation: the evolvability decomposition

Slope of adaptation AUC against generation, per class and transplant condition. The hypothesis requires `full` to beat `no_feedback` and `capacity_matched`.

| class | capacity_matched | fixed_hparams | full | no_feedback | reinit_weights |
|---|---|---|---|---|---|
| A | -0.00044 | +0.00038 | +0.00031 | -0.00002 | -0.00003 |
| B | +0.00015 | +0.00072 | +0.00287 | +0.00069 | -0.00031 |
| C_dev | -0.00005 | +0.00028 | +0.00146 | +0.00085 | -0.00028 |

Final-generation AUC by class and condition:

| class | capacity_matched | fixed_hparams | full | no_feedback | reinit_weights |
|---|---|---|---|---|---|
| A | -0.0140 | +0.0213 | +0.0319 | +0.0279 | +0.0005 |
| B | +0.0275 | +0.0888 | +0.1521 | +0.1321 | +0.0321 |
| C_dev | -0.0032 | +0.0864 | +0.1412 | +0.1434 | +0.0001 |

- **A**: feedback-attributable adaptation = `full - no_feedback` = +0.0040  (positive: the organism is using its own action/reward history)
- **B**: feedback-attributable adaptation = `full - no_feedback` = +0.0200  (positive: the organism is using its own action/reward history)
- **C_dev**: feedback-attributable adaptation = `full - no_feedback` = -0.0023  (non-positive: improvement is a reactive prior, not in-context learning)

## Architecture and mutation

| operator | births | mean score | survived to adulthood |
|---|---|---|---|
| `contract_ffn` | 146 | +0.1116 | 81/146 (55%) |
| `narrow` | 134 | +0.1067 | 57/134 (43%) |
| `narrow_head` | 56 | +0.0885 | 33/56 (59%) |
| `expand_ffn` | 67 | +0.0626 | 32/67 (48%) |
| `add_block` | 59 | +0.1012 | 31/59 (53%) |
| `widen` | 61 | +0.0902 | 31/61 (51%) |
| `widen_head` | 52 | +0.0884 | 33/52 (63%) |
| `remove_block` | 50 | +0.0765 | 23/50 (46%) |
| `add_head` | 52 | +0.0675 | 26/52 (50%) |
| `remove_head` | 27 | +0.0695 | 16/27 (59%) |

- distinct architectures that ever existed: **228**
- organisms recorded: 7485
- extinctions: 3645

## Pre-registered criteria (PREREGISTRATION.md)

Reported whether or not they are supported. Single seed unless stated; these are machinery checks, not confirmations.

**H1 — adaptation on novel structure improves** (class `C_dev`)

| # | criterion | value | verdict |
|---|---|---|---|
| 1 | slope on `full` > 0 | +0.00146 | met |
| 2 | `full` > `no_feedback` (not a reactive prior) | +0.00146 vs +0.00085 | met |
| 3 | `full` > `capacity_matched` (not capacity) | +0.00146 vs -0.00005 | met |
| 4 | `fixed_hparams` > 0 (not hyperparameter tuning) | +0.00028 | met |

Final-generation level: `full` +0.1412 vs `no_feedback` +0.1434 (difference -0.0023). The slope and the level can disagree; both are reported.

**H3 — architectures grow spontaneously**

| # | criterion | value | verdict |
|---|---|---|---|
| 1 | selected − neutral drift > 0 | -0.03720 nats/gen | **not met** |
| 2 | distribution minimum moves up | slope -4.93/gen | **not met** |
| 3 | displaced founders do not regress | see subclade test | n/a |
| 4 | effective parameters grow | slope -18.4/gen | **not met** |

**H4 — growth-bias gene rises above 0.5**: 0.500 -> 0.435. **not met**. Under drift this gene stays at 0.5 by construction (`tests/test_neutrality.py`), so a departure in either direction is selection, not noise.

Figures written to `results/pilot`.
