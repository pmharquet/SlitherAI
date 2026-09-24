# G10 validation and population audit

Run: `C:\Docker\SlitherAI\runs\20260924-163112-852097`. Validation files are indexed zero-based: G1=`generation-0000`, G5=`generation-0004`, G10=`generation-0009`. All use seed 938271, 32 maps, and 90 seconds. Deltas are paired in stored map order; confidence intervals are approximate 95% normal intervals. The last column in each paired result is G5>G1 / tie / G5<G1 (and likewise G10 vs G5); fewer boosts is descriptive and is not automatically a better outcome.

| Validation | Fitness | Food gain | Survival | Boost spent | SE fitness | SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| G1 | 9.451 | 21.843 | 21.88% | 1.266 | 2.533 | `c76fcc1c61e4c090…` |
| G5 | 14.069 | 25.783 | 40.62% | 0.150 | 3.268 | `5ec6d3ff64d6493f…` |
| G10 | 14.069 | 25.783 | 40.62% | 0.150 | 3.268 | `5ec6d3ff64d6493f…` |

| Metric | G1 | G5 | G10 | G5−G1: mean delta [95% CI]; >/=/< | G10−G5: mean delta [95% CI]; >/=/< |
|---|---:|---:|---:|---:|---:|
| fitness | 9.451 | 14.069 | 14.069 | 4.618 [-2.243, 11.479]; 20/0/12 | 0.000 [0.000, 0.000]; 0/32/0 |
| food_gain | 21.843 | 25.783 | 25.783 | 3.939 [-3.817, 11.696]; 17/0/15 | 0.000 [0.000, 0.000]; 0/32/0 |
| alive | 21.88% | 40.62% | 40.62% | 18.75% [2.43%, 35.07%]; 7/24/1 | 0.00% [0.00%, 0.00%]; 0/32/0 |
| boost_spent | 1.266 | 0.150 | 0.150 | -1.116 [-1.668, -0.564]; 4/10/18 | 0.000 [0.000, 0.000]; 0/32/0 |
| age | 54.736 | 56.219 | 56.219 | 1.483 [-12.618, 15.585]; 19/6/7 | 0.000 [0.000, 0.000]; 0/32/0 |
| kills | 0.188 | 0.094 | 0.094 | -0.094 [-0.229, 0.041]; 1/27/4 | 0.000 [0.000, 0.000]; 0/32/0 |

The G5 and G10 JSON artifacts are byte-identical: `5ec6d3ff64d6493f78dba46ffb9b85cb2fb6212de773dfa310cad20fd562b7ad`. Their per-map metrics match exactly, so this is the same validated policy outcome, not independent evidence of another gain.

G1→G5 paired fitness change is 4.618 (approximate 95% CI -2.243 to 11.479; 20 of 32 maps improved). The interval includes zero, so this score gain remains uncertain on this validation set. G5→G10 changes are exactly zero because the validation artifacts and champion genotype are unchanged.

## Population and speciation curve

`mean` and `best` are the NEAT population fitness fields; `eval fitness` is the separately logged population-wide mean episode metric. Node counts include the two output nodes; connection counts are enabled links.

| Generation | Mean fitness | Best fitness | Eval fitness | Food | Survival | Boost | Mean nodes | Enabled links | Species | Effective species | Largest share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -17.930 | 10.468 | -17.690 | 7.397 | 0.31% | 14.207 | 2.000 | 106.000 | 8 | 7.40 | 20.31% |
| 2 | -5.307 | 13.459 | -4.935 | 9.420 | 0.31% | 2.487 | 2.121 | 105.977 | 30 | 15.74 | 14.84% |
| 3 | -4.510 | 18.741 | -4.097 | 10.782 | 0.31% | 2.844 | 2.434 | 106.098 | 36 | 30.02 | 7.42% |
| 4 | -2.711 | 22.772 | -2.363 | 11.395 | 0.78% | 1.440 | 2.551 | 105.828 | 38 | 34.27 | 5.86% |
| 5 | -2.357 | 26.586 | -2.017 | 12.131 | 1.25% | 1.738 | 2.688 | 105.645 | 38 | 36.82 | 4.30% |
| 6 | 0.662 | 34.822 | 1.047 | 15.216 | 2.34% | 1.276 | 2.680 | 105.523 | 38 | 36.20 | 5.08% |
| 7 | 1.690 | 26.612 | 2.087 | 15.934 | 2.73% | 0.777 | 2.773 | 105.477 | 38 | 36.96 | 4.30% |
| 8 | 2.323 | 33.403 | 2.664 | 17.026 | 3.44% | 1.348 | 2.887 | 105.422 | 38 | 37.11 | 3.91% |
| 9 | 0.690 | 30.863 | 0.957 | 15.438 | 5.00% | 1.946 | 2.898 | 105.418 | 38 | 37.31 | 3.91% |
| 10 | 1.965 | 33.749 | 2.296 | 16.915 | 5.23% | 1.789 | 2.898 | 105.320 | 38 | 37.14 | 4.30% |

| Validation generation | Checkpoint | Mean hidden nodes | Mean enabled links | Distinct active topologies |
|---:|---|---:|---:|---:|
| G1 | `checkpoint-0` | 0.000 | 106.000 | 256 |
| G5 | `checkpoint-4` | 0.688 | 105.645 | 224 |
| G10 | `checkpoint-9` | 0.898 | 105.320 | 217 |

Active topology here is node IDs plus enabled connection endpoints; it is a structural diversity proxy, not a count of behaviorally distinct policies.

Across G1→G5→G10, population mean fitness was -17.930→-2.357→1.965; mean episode food gain was 7.397→12.131→16.915; survival was 0.31%→1.25%→5.23%. Species count/effective count was 8/7.40→38/36.82→38/37.14.

## Persistent validation champion

`best-validation.pkl` identifies genome 1065 from zero-based generation index 4 (G5); its topology has 1 hidden node(s) and 108 connection genes. The checkpoint sequence 4–10 contains this key with the same complete gene fingerprint: **True**. `exact/topology` gives the number of exact copies and matching active graph topologies in that checkpoint population.

| Checkpoint | Genome present | Hidden nodes | Connection genes | Species ID | Species size | Exact/topology copies |
|---:|---|---:|---:|---:|---:|---:|
| 4 | True | 1 | 108 | 14 | 10 | 1/1 |
| 5 | True | 1 | 108 | 14 | 10 | 1/1 |
| 6 | True | 1 | 108 | 14 | 11 | 1/1 |
| 7 | True | 1 | 108 | 14 | 10 | 1/2 |
| 8 | True | 1 | 108 | 14 | 10 | 1/2 |
| 9 | True | 1 | 108 | 14 | 11 | 1/1 |
| 10 | True | 1 | 108 | 14 | 11 | 1/1 |

| Generation | Rank / 256 | Selection fitness | Fixed-anchor score | Derived rotating score* |
|---:|---:|---:|---:|---:|
| 1 | absent | — | — | — |
| 2 | absent | — | — | — |
| 3 | absent | — | — | — |
| 4 | absent | — | — | — |
| 5 | 1 | 26.586 | 35.435 | -8.811 |
| 6 | 1 | 34.822 | 35.435 | 32.372 |
| 7 | 1 | 26.612 | 35.435 | -8.678 |
| 8 | 1 | 33.403 | 35.435 | 25.275 |
| 9 | 2 | 27.885 | 35.435 | -2.313 |
| 10 | 1 | 33.749 | 35.435 | 27.005 |

*Derived from the documented selection blend `(fitness − 0.8 × anchor_score) / 0.2`; it is the rotating-scenario contribution for this genome, not an independent validation score. The champion's fixed-anchor score remains 35.435 while its rotating score changes substantially. The validation score uses a different 32-map suite, so those raw scores are not directly comparable.

The G5 and G10 validation plateau is attributable to the same unchanged elite genotype, while population mean fitness, food gain, survival, topology, and species occupancy continue to move. The saved history does not record parent IDs, so descendant counts cannot be reconstructed; species occupancy and genotype persistence above are the reproducible concentration evidence. Ten generations are insufficient to infer global stagnation. Keep the present threshold, reward, and scenario mix for now; collect later common-validation points before proposing a training intervention.

## Reproduction

From the repository root, run `.venv/Scripts/python.exe docs/overnight/analyze_g10_audit.py`. The script reads saved artifacts only. It performs no simulation and writes nothing; `--run PATH` selects another completed run.
