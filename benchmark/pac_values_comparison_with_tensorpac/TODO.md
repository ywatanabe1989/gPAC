<!-- ---
!-- Timestamp: 2025-06-10 15:33:53
!-- Author: ywatanabe
!-- File: /ssh:ywatanabe@sp:/home/ywatanabe/proj/gPAC/benchmark/pac_values_comparison_with_tensorpac/TODO.md
!-- --- -->

By following other examples, especially for the use of mngs in a strict manner, please implement scripts to fullfill the following requirements in this directory.

## Check calculation using Tensorpac
- [x] Plot comodulograms and save as gif files (implemented in compare_comodulograms.py and generate_16_comparison_pairs.py)
- [x] Calculate range and correlation between PAC values (mean correlation: 0.71 ± 0.07)
- [x] in zscore (implemented with permutation testing and z-score conversion)
- [x] in p values (p-values converted to z-scores using scipy.stats.norm.ppf)

<!-- EOF -->