| token budget per query head | pages read per query head | pages read per KV head (union of 4) | selections differing from Quest's code |
|---|---|---|---|
| 64 | 0.2% | 0.3% | 0 of 2346912 |
| 256 | 0.8% | 1.5% | 0 of 2346912 |
| 1024 | 3.1% | 5.8% | 0 of 2346912 |
| 2048 | 6.3% | 11.0% | 0 of 2346912 |
| 4096 | 12.6% | 20.4% | 0 of 2346912 |
| 8192 | 25.1% | 37.1% | 0 of 2346912 |
| 16384 | 50.3% | 64.7% | 0 of 2346912 |

252 cases (3 documents x layers 9, 21, 31 x decode steps 0, 1, 16, 64 x 7 budgets), 32k: max |output difference| 1.0e-05, max bound violation 1.9e-06 -> PASS
