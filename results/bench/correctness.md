| check | cases | max rel. error | mean rel. error |
|---|---|---|---|
| kvskip_bf16_splits_auto | 48 | 1.83e-03 | 1.63e-03 |
| kvskip_f32_splits_auto | 48 | 1.09e-06 | 4.72e-07 |
| kvskip_bf16_splits_1 | 48 | 1.83e-03 | 1.63e-03 |
| kvskip_f32_splits_1 | 48 | 9.03e-06 | 4.06e-06 |
| kvskip_bf16_splits_7 | 48 | 1.83e-03 | 1.63e-03 |
| kvskip_f32_splits_7 | 48 | 2.26e-06 | 9.43e-07 |
| sdpa_bf16 | 48 | 1.89e-03 | 1.68e-03 |
| fa2_bf16 | 48 | 1.89e-03 | 1.68e-03 |
| kvskip_bf16_bshd_layout | 48 | 1.83e-03 | 1.63e-03 |
| kvskip_bf16_with_stats | 48 | 1.83e-03 | 1.63e-03 |
| kvskip_masked_bf16_splits_auto | 48 | 1.82e-03 | 1.62e-03 |
| kvskip_masked_f32_splits_auto | 48 | 5.12e-07 | 3.00e-07 |
| kvskip_masked_bf16_splits_1 | 48 | 1.82e-03 | 1.62e-03 |
| kvskip_masked_f32_splits_1 | 48 | 2.64e-06 | 1.44e-06 |
| kvskip_masked_bf16_splits_5 | 48 | 1.82e-03 | 1.62e-03 |
| kvskip_masked_f32_splits_5 | 48 | 1.11e-06 | 4.96e-07 |
| vskip2_topp_bf16 | 48 | 1.82e-03 | 1.62e-03 |
| vskip2_topp_f32 | 48 | 8.05e-05 | 4.57e-05 |
| vskip2_topp_kept_frac | 48 | 9.03e-01 | 6.42e-01 |
| vskip2_tau_bf16 | 48 | 1.81e-03 | 1.61e-03 |
| vskip2_tau_f32 | 48 | 8.05e-05 | 4.57e-05 |
| vskip2_tau_kept_frac | 48 | 9.52e-01 | 6.68e-01 |
| vskip1_tau0.0001_bf16 | 48 | 1.82e-03 | 1.62e-03 |
| vskip1_tau0.0001_f32 | 48 | 7.73e-07 | 3.69e-07 |
| vskip1_tau0.0001_kept_frac | 48 | 1.00e+00 | 8.16e-01 |
| vskip1_tau0.0001_kept_frac_ref | 48 | 1.00e+00 | 8.16e-01 |
| vskip1_lse_tau0.0001_f32 | 48 | 5.69e-07 | 2.85e-07 |
| vskip1_tau0.001_bf16 | 48 | 1.83e-03 | 1.62e-03 |
| vskip1_tau0.001_f32 | 48 | 7.25e-07 | 3.36e-07 |
| vskip1_tau0.001_kept_frac | 48 | 1.00e+00 | 5.04e-01 |
| vskip1_tau0.001_kept_frac_ref | 48 | 1.00e+00 | 5.04e-01 |
| vskip1_lse_tau0.001_f32 | 48 | 4.66e-07 | 2.59e-07 |
| kvskip_bf16_batch12 | 1 | 1.67e-03 | 1.67e-03 |
| synthetic_D64_T1000_splitsNone | 1 | 1.46e-03 | 1.46e-03 |
| synthetic_D64_T33_splits3 | 1 | 1.25e-03 | 1.25e-03 |
| synthetic_D128_T64_splits4 | 1 | 1.22e-03 | 1.22e-03 |
| synthetic_D128_T65_splits9 | 1 | 1.57e-03 | 1.57e-03 |
| synthetic_D128_T3000_splits64 | 1 | 1.42e-03 | 1.42e-03 |

Block stats vs fp64: max |Δ block max| = 8.78e-06 nats, max rel. Δ block sumexp = 7.44e-06, max |Δ block weight| = 4.75e-07.
