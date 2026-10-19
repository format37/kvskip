|                                               |   kl_x_noise |   worst |   bytes |
|:----------------------------------------------|-------------:|--------:|--------:|
| (16384, 'event trigger + 8-head list')        |        1.27  |    1.7  |   0.741 |
| (16384, 'event trigger + calibration (ours)') |        1.06  |    1.5  |   0.741 |
| (16384, 'event trigger alone')                |      109     |  319    |   0.73  |
| (16384, 'oracle trigger (coverage < 0.98)')   |        0.873 |    1.18 |   0.761 |
| (16384, 'query-drift trigger (per head)')     |        5     |    7.08 |   0.644 |
| (16384, 'timer, every 16 steps')              |       17.1   |   25.3  |   0.643 |
| (16384, 'timer, every 32 steps')              |       28.9   |   43.8  |   0.627 |
| (16384, 'timer, every 4 steps')               |        1.76  |    2.48 |   0.717 |
| (16384, 'timer, every 8 steps')               |       19.9   |   42.8  |   0.664 |
| (32768, 'event trigger + 8-head list')        |        1.35  |    1.5  |   0.638 |
| (32768, 'event trigger + calibration (ours)') |        1.54  |    1.8  |   0.635 |
| (32768, 'event trigger alone')                |        1.87  |    2.6  |   0.627 |
| (32768, 'oracle trigger (coverage < 0.98)')   |        1.24  |    1.58 |   0.657 |
| (32768, 'query-drift trigger (per head)')     |        3.24  |    4.29 |   0.532 |
| (32768, 'timer, every 16 steps')              |       14.7   |   35    |   0.529 |
| (32768, 'timer, every 32 steps')              |       10.4   |   20.9  |   0.505 |
| (32768, 'timer, every 4 steps')               |        1.99  |    2.26 |   0.627 |
| (32768, 'timer, every 8 steps')               |        3.83  |    6.96 |   0.561 |