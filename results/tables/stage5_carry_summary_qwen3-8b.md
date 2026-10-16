| variant                                      |   pairs |   kl_mean |   kl_wo_worst3 |   pairs_step_gt_50 |   better |   worse |   d_bytes_pp |   d_bytes_pp_max |
|:---------------------------------------------|--------:|----------:|---------------:|-------------------:|---------:|--------:|-------------:|-----------------:|
| stage 4: every document from zero            |      10 |      1.2  |          0.8   |                  2 |        0 |       0 |        0     |            0     |
| carry c x 0.5                                |      10 |      1.36 |          0.901 |                  3 |        2 |       7 |        0.328 |            0.507 |
| carry c x 0.8                                |      10 |      1.23 |          0.786 |                  4 |        3 |       5 |        0.689 |            1.01  |
| carry c x 1                                  |      10 |      1.31 |          0.866 |                  1 |        2 |       7 |        0.924 |            1.38  |
| carry c x 1, no warm-up once carried         |      10 |      1.33 |          0.845 |                  4 |        3 |       5 |        0.702 |            1.23  |
| carry the index latch only                   |      10 |      1.23 |          0.849 |                  2 |        3 |       5 |        0.448 |            1.06  |
| floor: worst miss so far, x 0.8 per document |      10 |      1.09 |          0.778 |                  0 |        3 |       6 |       13.4   |           22.5   |