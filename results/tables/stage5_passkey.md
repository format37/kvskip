| condition                                              |   retrieved |   depths |   kl_x_noise |   kl_worst |   answer_kl_x_noise |   bytes_total |   answer_as_reference |
|:-------------------------------------------------------|------------:|---------:|-------------:|-----------:|--------------------:|--------------:|----------------------:|
| exact fp32 (reference)                                 |          10 |       10 |   nan        | nan        |              nan    |         1     |                   nan |
| Quest as published, 256 tokens / query head            |          10 |       10 |     1.18e+03 |   2.92e+03 |              400    |         0.135 |                     4 |
| Quest as published, 512 tokens / query head            |          10 |       10 |   652        |   1.31e+03 |              244    |         0.149 |                     5 |
| Quest as published, 1024 tokens / query head           |          10 |       10 |   398        | 780        |              164    |         0.176 |                     7 |
| Quest as published, 2048 tokens / query head           |          10 |       10 |   208        | 660        |              106    |         0.225 |                     7 |
| Quest as published, 4096 tokens / query head           |          10 |       10 |    89.1      | 530        |               63.3  |         0.316 |                     7 |
| refresh + calibration (stage 4), top-p 0.99            |          10 |       10 |     2.93     |  51        |                4.08 |         0.764 |                    10 |
| refresh + calibration, top-p 0.4                       |          10 |       10 |   121        | 213        |               79.7  |         0.456 |                     5 |
| refresh + calibration, top-p 0.2                       |          10 |       10 |   230        | 596        |              184    |         0.39  |                     5 |
| per-step oracle at the bytes of questp2048             |          10 |       10 |    16.1      |  33.7      |               11.1  |         0.225 |                     8 |
| static at the bytes of questp2048                      |          10 |       10 |    59.9      | 352        |               70.3  |         0.226 |                     6 |
| Quest, stage-4 GQA variant, at the bytes of questp2048 |          10 |       10 |    29.7      |  49        |               16.5  |         0.225 |                     7 |