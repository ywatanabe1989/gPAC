<!-- ---
!-- Timestamp: 2025-05-25 11:10:34
!-- Author: ywatanabe
!-- File: /ssh:sp:/home/ywatanabe/proj/gPAC/project_management/USER_PLAN.md
!-- --- -->

## ./examples/readme_demo.py
#### Using Synthetic data generation from gPAC package

## Demo Image as GIF
Top: Raw Signal of Synthetic data
Bottom left: PAC calculated by gPAC
Bottom center: PAC calculated by Tensorpac
Bottom right: Difference (PAC calculated by gPAC - that of Tensorpac)

- x/y axis labels should be Hz

## Calculation speed as text
## Ground Truth PAC target range

## References
/home/ywatanabe/proj/mngs_repo/src/mngs/dsp/utils/pac.py
  - p.idpac = (2,0,0) for MI
  - hres -> n_bands = 100
  - mres -> n_bands = 70
~/proj/mngs_repo/src/mngs/dsp/_pac.py
~/proj/mngs_repo/src/mngs/dsp/_pac/pac_with_static_bandpass_fp32.png



## ./examples/readme_demo_realworld.py
#### Using a Real World dataset (EEG during cognitive task; are there easily downloadable dataset?)

<!-- EOF -->