# Phase NE-4 - Curated decoder generalization

Acceptance: >=80% verdict-action match on the 10 unseen cycles.

| cycle | curated? | synthetic | full real-neural | curated real-neural | match (curated vs full)? |
|------:|:---------|:----------|:-----------------|:--------------------|:------------------------|
|     0 | curated | PROCEED | PROCEED | PROCEED | OK    |
|     1 |         | PROCEED | PROCEED | PROCEED | OK    |
|     2 |         | PROCEED | PROCEED | PROCEED | OK    |
|     3 |         | REPLAN  | PROCEED | PROCEED | OK    |
|     4 | curated | REPLAN  | PROCEED | REPLAN  | diff  |
|     5 |         | REPLAN  | PROCEED | PROCEED | OK    |
|     6 |         | ABORT   | ABORT   | ABORT   | OK    |
|     7 | curated | ABORT   | ABORT   | ABORT   | OK    |
|     8 |         | ABORT   | ABORT   | ABORT   | OK    |
|     9 |         | ABORT   | ABORT   | ABORT   | OK    |
|    10 |         | ABORT   | ABORT   | ABORT   | OK    |
|    11 | curated | ABORT   | ABORT   | ABORT   | OK    |
|    12 |         | ABORT   | ABORT   | ABORT   | OK    |
|    13 |         | ABORT   | ABORT   | ABORT   | OK    |
|    14 | curated | PROCEED | PROCEED | PROCEED | OK    |

**Curated cycles (5):** curated == full on 4/5
**Unseen cycles (10):** curated == full on 10/10 = **100%**

[ACCEPT] curated decoder generalizes (100% >= 80%).