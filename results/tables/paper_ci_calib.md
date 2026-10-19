|                                                                                      |   point |   ci_lo |   ci_hi |   worst |   docs |   bytes |
|:-------------------------------------------------------------------------------------|--------:|--------:|--------:|--------:|-------:|--------:|
| ('8b', 'held-out (a4) 32k', 'trigger alone (no list, delta 0.05)')                   |   4.69  |   1.42  |   11.1  |   11.1  |      3 |   0.632 |
| ('8b', 'held-out (a4) 32k', 'peak-hold correction, no warm-up')                      |   1.39  |   1.27  |    1.46 |    1.46 |      3 |   0.643 |
| ('8b', 'held-out (a4) 32k', 'peak-hold correction + warm-up')                        |   1.63  |   1.09  |    2.42 |    2.42 |      3 |   0.645 |
| ('8b', 'held-out (a4) 32k', 'peak-hold + warm-up + auto token index (stage 4)')      |   1.65  |   1.01  |    2.44 |    2.44 |      3 |   0.645 |
| ('8b', 'held-out (a4) 32k', 'ratio correction, EMA 0.3 (kickoff)')                   |   1.66  |   1.27  |    2.24 |    2.24 |      3 |   0.634 |
| ('8b', 'held-out (a4) 32k', 'ratio correction + warm-up')                            |   1.5   |   1.27  |    1.92 |    1.92 |      3 |   0.633 |
| ('8b', 'held-out (a4) 32k', 'additive correction (EMA) + warm-up')                   |   1.56  |   1.28  |    1.92 |    1.92 |      3 |   0.632 |
| ('8b', 'held-out (a4) 32k', '+ 8 always-exact heads (stage 3, Llama)')               |   1.3   |   1.14  |    1.54 |    1.54 |      3 |   0.644 |
| ('8b', 'held-out (s3) 32k', 'trigger alone (no list, delta 0.05)')                   |   1.13  |   0.953 |    1.26 |    1.26 |      3 |   0.617 |
| ('8b', 'held-out (s3) 32k', 'peak-hold correction + warm-up')                        |   1.19  |   1.11  |    1.24 |    1.24 |      3 |   0.626 |
| ('8b', 'held-out (s3) 32k', 'peak-hold + warm-up + auto token index (stage 4)')      |   1.14  |   0.995 |    1.24 |    1.24 |      3 |   0.626 |
| ('8b', 'held-out (s3) 32k', 'the same, warm-up 64 steps (every 4)')                  |   1.07  |   0.826 |    1.21 |    1.21 |      3 |   0.653 |
| ('8b', 'held-out (s3) 32k', '+ 8 always-exact heads (stage 3, Llama)')               |   0.921 |   0.83  |    1.09 |    1.09 |      3 |   0.628 |
| ('8b', 'stage-1 docs 16k', 'trigger alone (no list, delta 0.05)')                    | 109     |   1.36  |  319    |  319    |      3 |   0.73  |
| ('8b', 'stage-1 docs 16k', 'peak-hold correction, no warm-up')                       |  49.1   |   1.3   |  121    |  121    |      3 |   0.738 |
| ('8b', 'stage-1 docs 16k', 'peak-hold correction + warm-up')                         |   1.07  |   0.725 |    1.51 |    1.51 |      3 |   0.741 |
| ('8b', 'stage-1 docs 16k', 'peak-hold + warm-up + auto token index (stage 4)')       |   1.06  |   0.728 |    1.46 |    1.46 |      3 |   0.741 |
| ('8b', 'stage-1 docs 16k', 'ratio correction, EMA 0.3 (kickoff)')                    |  43.8   |   1.39  |  125    |  125    |      3 |   0.733 |
| ('8b', 'stage-1 docs 16k', 'ratio correction + warm-up')                             |  11.6   |   1.32  |   32    |   32    |      3 |   0.732 |
| ('8b', 'stage-1 docs 16k', 'additive correction (EMA) + warm-up')                    |   2.36  |   1.42  |    3.72 |    3.72 |      3 |   0.73  |
| ('8b', 'stage-1 docs 16k', '+ 8 always-exact heads (stage 3, Llama)')                |   1.27  |   0.979 |    1.67 |    1.67 |      3 |   0.741 |
| ('8b', 'stage-1 docs 32k', 'trigger alone (no list, delta 0.05)')                    |   1.87  |   1.45  |    2.61 |    2.61 |      3 |   0.627 |
| ('8b', 'stage-1 docs 32k', 'peak-hold correction, no warm-up')                       |   2.13  |   1.49  |    3.01 |    3.01 |      3 |   0.635 |
| ('8b', 'stage-1 docs 32k', 'peak-hold correction + warm-up')                         |   1.63  |   1.24  |    2.27 |    2.27 |      3 |   0.636 |
| ('8b', 'stage-1 docs 32k', 'peak-hold + warm-up + auto token index (stage 4)')       |   1.54  |   1.12  |    1.84 |    1.84 |      3 |   0.635 |
| ('8b', 'stage-1 docs 32k', 'ratio correction, EMA 0.3 (kickoff)')                    |   2.05  |   1.68  |    2.78 |    2.78 |      3 |   0.628 |
| ('8b', 'stage-1 docs 32k', 'ratio correction + warm-up')                             |   1.41  |   0.994 |    1.63 |    1.63 |      3 |   0.624 |
| ('8b', 'stage-1 docs 32k', 'additive correction (EMA) + warm-up')                    |   1.36  |   0.988 |    1.8  |    1.8  |      3 |   0.623 |
| ('8b', 'stage-1 docs 32k', '+ 8 always-exact heads (stage 3, Llama)')                |   1.35  |   1.17  |    1.51 |    1.51 |      3 |   0.638 |
| ('qwen3-8b', 'stage-1 docs 16k', 'trigger alone, delta 0.02 (stage 3, Qwen3)')       |   1.3   |   0.817 |    1.93 |    1.93 |      3 |   0.761 |
| ('qwen3-8b', 'stage-1 docs 16k', 'trigger alone (no list, delta 0.05)')              |   1.31  |   0.987 |    1.52 |    1.52 |      3 |   0.712 |
| ('qwen3-8b', 'stage-1 docs 16k', 'peak-hold correction, no warm-up')                 |   1.45  |   1.1   |    1.79 |    1.79 |      3 |   0.725 |
| ('qwen3-8b', 'stage-1 docs 16k', 'peak-hold correction + warm-up')                   |   1.47  |   1.18  |    1.63 |    1.63 |      3 |   0.728 |
| ('qwen3-8b', 'stage-1 docs 16k', 'peak-hold + warm-up + auto token index (stage 4)') |   1.21  |   1.14  |    1.28 |    1.28 |      3 |   0.729 |
| ('qwen3-8b', 'stage-1 docs 16k', 'ratio correction + warm-up')                       |   1.25  |   0.775 |    1.52 |    1.52 |      3 |   0.712 |
| ('qwen3-8b', 'stage-1 docs 16k', 'additive correction (EMA) + warm-up')              |   1.26  |   0.839 |    1.68 |    1.68 |      3 |   0.711 |
| ('qwen3-8b', 'stage-1 docs 32k', 'trigger alone, delta 0.02 (stage 3, Qwen3)')       |   1.24  |   1.01  |    1.64 |    1.64 |      3 |   0.722 |
| ('qwen3-8b', 'stage-1 docs 32k', 'trigger alone (no list, delta 0.05)')              |   1.52  |   1.11  |    2.23 |    2.23 |      3 |   0.664 |
| ('qwen3-8b', 'stage-1 docs 32k', 'peak-hold correction, no warm-up')                 |   1.25  |   0.95  |    1.67 |    1.67 |      3 |   0.683 |
| ('qwen3-8b', 'stage-1 docs 32k', 'peak-hold correction + warm-up')                   |   1.28  |   1.07  |    1.39 |    1.39 |      3 |   0.687 |
| ('qwen3-8b', 'stage-1 docs 32k', 'peak-hold + warm-up + auto token index (stage 4)') |   1.03  |   0.919 |    1.1  |    1.1  |      3 |   0.688 |
| ('qwen3-8b', 'stage-1 docs 32k', 'ratio correction + warm-up')                       |   1.33  |   1.05  |    1.83 |    1.83 |      3 |   0.664 |
| ('qwen3-8b', 'stage-1 docs 32k', 'additive correction (EMA) + warm-up')              |   1.28  |   1.07  |    1.56 |    1.56 |      3 |   0.664 |