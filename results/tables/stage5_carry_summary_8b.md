| variant                                      |   pairs |   kl_mean |   kl_wo_worst3 |   pairs_step_gt_50 |   better |   worse |   d_bytes_pp |   d_bytes_pp_max |
|:---------------------------------------------|--------:|----------:|---------------:|-------------------:|---------:|--------:|-------------:|-----------------:|
| stage 4: every document from zero            |      16 |      1.45 |          0.971 |                  5 |        0 |       0 |        0     |            0     |
| carry c x 0.5                                |      16 |      1.48 |          0.93  |                  6 |        5 |       7 |        0.325 |            0.722 |
| carry c x 0.8                                |      16 |      1.38 |          0.889 |                  4 |        8 |       4 |        0.596 |            1.17  |
| carry c x 1                                  |      16 |      1.34 |          0.947 |                  2 |        7 |       4 |        0.847 |            1.6   |
| carry c x 1, no warm-up once carried         |      16 |      1.31 |          0.882 |                  2 |        7 |       5 |        1.19  |            2.01  |
| carry the index latch only                   |      16 |      1.25 |          0.899 |                  3 |        8 |       1 |        0.794 |            2.5   |
| floor: worst miss so far, x 0.8 per document |      16 |      1.18 |          0.909 |                  0 |       10 |       3 |        8.15  |           15.9   |