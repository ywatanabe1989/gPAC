<!-- ---
!-- title: ./torchPAC/scripts/PackageHandlers/README.md
!-- author: ywatanabe
!-- date: 2024-11-04 22:13:45
!-- --- -->


## Variables which impacts on calculation
- Package-inspecific variables

| Variable      | Tensorpac    | MNGS (torchPAC) | Impact                                               | Tensorpac Ref                                                   | MNGS Ref                                             |
|---------------|--------------|-----------------|------------------------------------------------------|-----------------------------------------------------------------|------------------------------------------------------|
| `batch_size`  | ✓            | ✓               | # of Signal Samples                                  | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)             |
| `n_chs`       | ✓            | ✓               | # of Signal Channels                                 | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)             |
| `n_segments`  | ✓            | ✓               | # of Signal segments                                 | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)             |
| `t_sec`       | ✓            | ✓               | Signal Length for Batch Samples [s]                  |                                                                 |                                                      |
| `fs`          | ✓            | ✓               | Sampling frequency [1/Hz]                            | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)             |
| `pha_n_bands` | ✓            | ✓               | # of Frequency Bands for Phase                       | [_BaseHandler.py#L106](_BaseHandler.py#L39)                     | [_BaseHandler.py#L43-L46](BaseHandler.py#L43-L46)    |
| `amp_n_bands` | ✓            | ✓               | # of Frequency Bands for Amplitude                   | [_BaseHandler.py#L106](_BaseHandler.py#L39)                     | [_BaseHandler.py#L43-L46](BaseHandler.py#L43-L46)    |
| `n_perm`      | ✓            | ✓               | # of permutations to calculate z score of PAC values | [TensorpacHandler.py#L41](TensorpacHandler.py#L41)              | [MNGSHandler.py#L47](MNGSHandler.py#L47)             |
| `fp16`        | ✓            | ✓               | Use float16 precision (default float32)              | [prepare_signal.py#L18-L21](../utils/prepare_signal.py#L18-L21) | [MNGSHandler.py#L62](MNGSHandler.py#L62)             |
| `chunk_size`  | ✓ (✘ ^1, ^2) | ✓               | # of Signal Samples calculated at one itereation     | [TensorpacHandler.py#L118-L121](TensorpacHandler.py#L118-L121)  | [MNGSHandler.py#L143-L169](MNGSHandler.py#L143-L169) |



- mngs-specific variables

| Variable    | Tensorpac | MNGS (torchPAC) | Impact                                            | Tensorpac Ref | MNGS Ref                                       |
|-------------|-----------|-----------------|---------------------------------------------------|---------------|------------------------------------------------|
| `no_grad`   | ✓         | ✘               | Whether to calculate gradiation (for AI training) | -             | [MNGSHandler.py#L48](MNGSHandler.py#L48) fixme |
| `in_place`  | ✓         | ✘               | Whether to calculate in the in_place manner       | -             | [MNGSHandler.py#L48](MNGSHandler.py#L96)       |
| `trainable` | ✓         | ✘               | Whether to use trainable model                    | -             | [MNGSHandler.py#L49](MNGSHandler.py#L97)       |
| `device`    | ✓         | ✘               | Hardware selection (CPU or GPU)                   | -             | [MNGSHandler.py#L50](MNGSHandler.py#L98)       |


- tensorpac-specific variables

| Variable      | Tensorpac | MNGS (torchPAC) | Impact              | Tensorpac Ref                                      | MNGS Ref |
|---------------|-----------|-----------------|---------------------|----------------------------------------------------|----------|
| `use_threads` | ✘         | ✓               | CPU parallelization | [TensorpacHandler.py#L66](TensorpacHandler.py#L66) | -        |

## Batch_size vs. chunk_size
- <sup>1</sup> n_chunks = math.ceil(batch_size / chunk_size)
- <sup>2</sup> chunk size does not affect the use_threads mode for Tensorpac calculation
