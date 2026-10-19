| condition                         |   niah_multikey_2 |   niah_multivalue |    vt |   fwe |   qa_1 |
|:----------------------------------|------------------:|------------------:|------:|------:|-------:|
| refresh + calibration (ours)      |             0.853 |             0.749 | 0.76  | 0.784 |  0.777 |
| + head (2,5) latched to the index |             0.853 |             0.748 | 0.76  | 0.785 |  0.777 |
| refresh + calibration, top-p 0.2  |             0.315 |             0.241 | 0.196 | 0.199 |  0.282 |
| Quest per KV head, our bytes      |             0.853 |             0.749 | 0.76  | 0.784 |  0.777 |
| Quest per KV head, Quest's bytes  |             0.239 |             0.228 | 0.227 | 0.227 |  0.221 |
| Quest as published, 2048 tokens   |             0.239 |             0.228 | 0.227 | 0.227 |  0.221 |
| exact (fp32)                      |             1     |             1     | 1     | 1     |  1     |
| stock bf16 SDPA                   |             1     |             1     | 1     | 1     |  1     |
| per-step oracle, our bytes        |             0.852 |             0.749 | 0.76  | 0.784 |  0.777 |
| per-step oracle, Quest's bytes    |             0.239 |             0.228 | 0.227 | 0.227 |  0.221 |