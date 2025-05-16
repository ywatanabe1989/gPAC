# Bug Report: Shape Mismatch in PAC Classifier

## Issue Description
There is a matrix shape mismatch error when running the `classify_demo_signals_using_gPAC_module.py` script. The error occurs during the forward pass of the `PACClassifier` when attempting to process the PAC values through the classifier.

**Error message:**
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x300 and 100x64)
```

This happens in the forward pass of the `PACClassifier` class when processing the output from the PAC module.

## Root Cause
The error occurs because there's a shape mismatch between the output of the PAC module and what the classifier expects.

1. In the `PAC` module, the output shape of `observed_pac` (line 248) or `pac_z` (line 246) is `(B, C, F_pha, F_amp)` where:
   - `F_pha` is the number of phase frequency bands (`pha_n_bands`)
   - `F_amp` is the number of amplitude frequency bands (`amp_n_bands`)

2. In the `PACClassifier` class, the code assumes that the output from the PAC module will be flattened to have `n_features = self.pha_n_bands * self.amp_n_bands` features.

3. But when the data is passed to the linear layer in the classifier, it's not being properly reshaped. The error suggests:
   - The PAC output has a shape that, when flattened incorrectly, gives 300 features (32x300)
   - The classifier's linear layer expects 100 features based on its weight matrix shape (100x64)

## Steps to Reproduce
Run the script `./scripts/learnability/classify_demo_signals_using_gPAC_module.py`, which will fail with the matrix shape mismatch error when it tries to train the PAC classifier.

## Impact
The classification model cannot be trained or used due to this shape mismatch, preventing the evaluation of trainable vs. fixed PAC band implementations.

## Fix Required
The `forward` method in the `PACClassifier` class needs to properly reshape the PAC values before passing them to the classification heads. The PAC module output needs to be reshaped to correctly match the expected input dimensions of the classifier's linear layers.

## Applied Fix
The issue has been fixed by modifying the `forward` method in the `PACClassifier` class to properly handle the channel dimension from the PAC module output:

1. Added logic to detect and handle the channel dimension (dimension 1) in the PAC values tensor
2. If multiple channels are present, we average across channels to get a single representation
3. If only one channel is present, we simply remove the channel dimension
4. After handling the channel dimension, the tensor is then flattened to the expected shape for the classifier

This fix allows the PAC values to be properly reshaped from `(B, C, F_pha, F_amp)` to `(B, F_pha*F_amp)` for classification.