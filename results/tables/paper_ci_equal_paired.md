| method                                 |   docs_worse_than_ours |   ratio_point |   ratio_ci_lo |   ratio_ci_hi |   ratio_docs |
|:---------------------------------------|-----------------------:|--------------:|--------------:|--------------:|-------------:|
| refresh + 8 hand-picked exact heads    |                      2 |         0.863 |         0.702 |          1.06 |            6 |
| per-step oracle                        |                      2 |         0.867 |         0.693 |          1.12 |            6 |
| Quest, pages per KV head               |                      5 |         3.08  |         1.31  |          7.98 |            6 |
| Quest, 16-token bounds, 64-token loads |                      6 |         8.05  |         2.78  |         21.8  |            6 |
| Quest as published (12288 tokens)      |                      6 |        36.1   |        27.2   |         46.8  |            6 |
| static                                 |                      6 |         4.57  |         2.54  |          7.86 |            6 |
| single-pass V-skip                     |                      2 |         0.78  |         0.563 |          1.08 |            6 |