|                                                         |   kl_x_noise |   kl_p95 |   top1 |   kv_read |   refresh_rate |   ms_per_step |
|:--------------------------------------------------------|-------------:|---------:|-------:|----------:|---------------:|--------------:|
| (32768, 'kvskip_dense', 'code')                         |        1.23  | 0.00237  |  1     |   nan     |       nan      |          25.4 |
| (32768, 'kvskip_dense', 'dialog')                       |        1.43  | 0.00201  |  0.992 |   nan     |       nan      |          25.3 |
| (32768, 'kvskip_dense', 'prose')                        |        0.84  | 0.000871 |  1     |   nan     |       nan      |          27.4 |
| (32768, 'kvskip_refresh_delta0.05_age32_dh8', 'code')   |        1.28  | 0.00212  |  0.992 |     0.711 |         0.0841 |          61.9 |
| (32768, 'kvskip_refresh_delta0.05_age32_dh8', 'dialog') |        1.49  | 0.00194  |  0.992 |     0.48  |         0.0747 |          59.2 |
| (32768, 'kvskip_refresh_delta0.05_age32_dh8', 'prose')  |        1.69  | 0.00178  |  1     |     0.73  |         0.0811 |          67.1 |
| (32768, 'kvskip_vskip1_prev_tau0.0001', 'code')         |        1.04  | 0.0019   |  1     |   nan     |       nan      |          24.9 |
| (32768, 'kvskip_vskip1_prev_tau0.0001', 'dialog')       |        1.27  | 0.00207  |  1     |   nan     |       nan      |          25   |
| (32768, 'kvskip_vskip1_prev_tau0.0001', 'prose')        |        1.44  | 0.00182  |  0.984 |   nan     |       nan      |          25.3 |
| (32768, 'kvskip_vskip1_sink_tau0.0001', 'code')         |        0.843 | 0.00195  |  1     |   nan     |       nan      |          25.5 |
| (32768, 'kvskip_vskip1_sink_tau0.0001', 'dialog')       |        1.37  | 0.00195  |  0.984 |   nan     |       nan      |          24.8 |
| (32768, 'kvskip_vskip1_sink_tau0.0001', 'prose')        |        1.01  | 0.00107  |  0.984 |   nan     |       nan      |          26.8 |
| (32768, 'kvskip_vskip2_topp0.99', 'code')               |        1.13  | 0.00193  |  1     |   nan     |       nan      |          44.3 |
| (32768, 'kvskip_vskip2_topp0.99', 'dialog')             |        1.23  | 0.00194  |  0.984 |   nan     |       nan      |          43.9 |
| (32768, 'kvskip_vskip2_topp0.99', 'prose')              |        1.22  | 0.000951 |  1     |   nan     |       nan      |          48.5 |
| (32768, 'stock_sdpa_bf16', 'code')                      |        1     | 0.00195  |  0.984 |   nan     |       nan      |          25.6 |
| (32768, 'stock_sdpa_bf16', 'dialog')                    |        1     | 0.0019   |  1     |   nan     |       nan      |          25.4 |
| (32768, 'stock_sdpa_bf16', 'prose')                     |        1     | 0.00102  |  1     |   nan     |       nan      |          26.8 |