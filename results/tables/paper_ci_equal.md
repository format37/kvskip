| method                                  |   bytes |   point |   ci_lo |   ci_hi |      worst |   docs |
|:----------------------------------------|--------:|--------:|--------:|--------:|-----------:|-------:|
| refresh + calibration (ours)            |   0.64  |    1.53 |   1.22  |    1.94 |   2.44     |      6 |
| refresh + 8 hand-picked exact heads     |   0.641 |    1.32 |   1.2   |    1.45 |   1.54     |      6 |
| per-step oracle                         |   0.64  |    1.32 |   1.06  |    1.58 |   1.75     |      6 |
| Quest, pages per KV head                |   0.64  |    4.7  |   1.82  |   14.5  |  50.5      |      6 |
| Quest, 16-token bounds, 64-token loads  |   0.64  |   12.3  |   3.85  |   37.9  |  83.5      |      6 |
| Quest as published (12288 tokens)       |   0.602 |   55.1  |  38.9   |   77.4  |  88.9      |      6 |
| static                                  |   0.641 |    6.98 |   3.41  |   14.2  |  27.8      |      6 |
| single-pass V-skip                      |   0.77  |    1.19 |   0.907 |    1.52 |   1.75     |      6 |
| refresh + calibration at top-p 0.2      |   0.202 |  168    | 109     |  255    | 304        |      6 |
| Quest per KV head at Quest-2048's bytes |   0.225 |   47.2  |  14.1   |  156    | 285        |      6 |
| Quest as published (2048 tokens)        |   0.225 |  458    | 304     |  797    |   1.54e+03 |      6 |
| per-step oracle at Quest-2048's bytes   |   0.225 |    7    |   3.91  |   12.8  |  19.7      |      6 |