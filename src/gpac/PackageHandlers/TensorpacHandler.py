#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-05 00:02:38 (ywatanabe)"
# File: ./torchPAC/scripts/PackageHandlers/TensorpacHandler.py

"""
Functionality:
    - Implements TensorpacHandler for phase-amplitude coupling calculations
    - Provides both thread-based and chunk-based processing options
Input:
    - Neural time series data (EEG/iEEG)
    - Configuration parameters for PAC calculation
Output:
    - Phase-amplitude coupling matrices
Prerequisites:
    - tensorpac
    - numpy
    - mngs package
"""

import math
from concurrent.futures import ThreadPoolExecutor

import mngs
import numpy as np
import tensorpac
from scripts.PackageHandlers import BaseHandler

TIMEOUT_SEC = int(10 * 60)


class TensorpacHandler(BaseHandler):
    def __init__(
        self,
        seq_len: int,
        fs: float,
        pha_n_bands: int,
        pha_min_hz: float,
        pha_max_hz: float,
        amp_n_bands: int,
        amp_min_hz: float,
        amp_max_hz: float,
        n_perm: int,
        chunk_size: int,
        fp16: bool,
        in_place: bool,
        trainable: bool,
        device: str,
        use_threads: bool,
        ts: mngs.gen.TimeStamper,
    ):
        super().__init__(
            seq_len,
            fs,
            pha_n_bands,
            pha_min_hz,
            pha_max_hz,
            amp_n_bands,
            amp_min_hz,
            amp_max_hz,
            n_perm,
            chunk_size,
            fp16,
            in_place,
            trainable,
            device,
            use_threads,
            ts,
        )

        del self.in_place, self.trainable
        self.ts(self.init_start_str)
        self.model = self.init_model()
        self.ts(self.init_end_str)

    def init_model(self) -> tensorpac.Pac:
        resolution_dict = {
            10: "lres",
            30: "mres",
            50: "hres",
            70: "demon",
            100: "hulk",
        }
        model = tensorpac.Pac(
            f_pha=resolution_dict[self.pha_n_bands],
            f_amp=resolution_dict[self.pha_n_bands],
            dcomplex="wavelet",
        )
        model.idpac = (2, 0, 0)
        return model

    def calc_pac(self, xx: np.ndarray) -> np.ndarray:
        """
        Calculate phase-amplitude coupling using either threads or chunks.

        Parameters
        ----------
        xx : np.ndarray
            Input array of shape (batch_size, n_chs, n_segments, seq_len)

        Returns
        -------
        np.ndarray
            Calculated PAC values
        """
        return (
            self._calc_pac_unfair_using_threads(xx)
            if self.use_threads
            else self._calc_pac_fair_chunk(xx)
        )

    def _calc_pac(self, xs: np.ndarray) -> np.ndarray:
        pha = self.model.filter(self.fs, xs, ftype="phase", n_jobs=-1)
        amp = self.model.filter(self.fs, xs, ftype="amplitude", n_jobs=-1)
        xpac = self.model.fit(pha, amp, n_jobs=-1, verbose=False)
        return xpac.transpose(2, 1, 0)

    def _calc_pac_fair_chunk(self, xx: np.ndarray) -> np.ndarray:
        """
        Calculate PAC using chunk-based processing.

        Parameters
        ----------
        xx : np.ndarray
            Input array of shape (batch_size, n_chs, n_segments, seq_len)

        Returns
        -------
        np.ndarray
            Calculated PAC values
        """
        assert xx.ndim == 4
        assert xx.dtype == np.float16 if self.fp16 else np.float32

        xx = self.dim_handler.fit(xx, keepdims=[-1])
        n_chunks = math.ceil(len(xx) / self.chunk_size)

        xpac = np.vstack(
            [
                self._calc_pac(
                    xx[i_batch * self.chunk_size : (i_batch + 1) * self.chunk_size]
                )
                for i_batch in range(n_chunks)
            ]
        )

        return self.dim_handler.unfit(xpac)

    def _calc_pac_unfair_using_threads(
        self, xx: np.ndarray, n_jobs: int = -1
    ) -> np.ndarray:
        """
        Calculate PAC using thread-based processing.

        Parameters
        ----------
        xx : np.ndarray
            Input array of shape (batch_size, n_chs, n_segments, seq_len)
        n_jobs : int, optional
            Number of parallel jobs, by default -1 (all cores)

        Returns
        -------
        np.ndarray
            Calculated PAC values
        """
        assert xx.ndim == 4
        assert xx.dtype == np.float16 if self.fp16 else np.float32

        xx = self.dim_handler.fit(xx, keepdims=np.array([-2, -1]))
        batch_size_by_n_chs = len(xx)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._calc_pac, xx[i_batch])
                for i_batch in range(batch_size_by_n_chs)
            ]
            xpac = [future.result() for future in futures]

        xpac = np.array(xpac)
        return self.dim_handler.unfit(xpac)

    @property
    def freqs_amp(self) -> np.ndarray:
        return self.model.f_amp.mean(axis=-1)

    @property
    def freqs_pha(self) -> np.ndarray:
        return self.model.f_pha.mean(axis=-1)

    def __str__(self) -> str:
        return "Tensorpac"


# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-04 15:05:23 (ywatanabe)"
# # File: ./torchPAC/scripts/PackageHandlers/TensorpacHandler.py

# """
# This script defines TensorpacHandler.
# """

# # Imports
# import math
# from concurrent.futures import ThreadPoolExecutor

# import mngs
# import numpy as np
# import tensorpac
# from scripts.PackageHandlers import BaseHandler

# TIMEOUT_SEC = int(10 * 60)


# # Functions
# class TensorpacHandler(BaseHandler):
#     def __init__(
#         self,
#         seq_len: int,
#         fs: float,
#         pha_n_bands: int,
#         pha_min_hz: float,
#         pha_max_hz: float,
#         amp_n_bands: int,
#         amp_min_hz: float,
#         amp_max_hz: float,
#         n_perm: int,
#         chunk_size: int,
#         fp16: bool,
#         in_place: bool,
#         trainable: bool,
#         device: str,
#         use_threads: bool,
#         ts: mngs.gen.TimeStamper,
#     ):
#         # Maintains parameters as attributes by following the BaseHandler's requirements
#         super().__init__(
#             seq_len,
#             fs,
#             pha_n_bands,
#             pha_min_hz,
#             pha_max_hz,
#             amp_n_bands,
#             amp_min_hz,
#             amp_max_hz,
#             n_perm,
#             chunk_size,
#             fp16,
#             in_place,
#             trainable,
#             device,
#             use_threads,
#             ts,
#         )

#         # Explicitly disables unneccessary variables for this class.
#         # Since parameters are passed using the grid search method, the above parameters should be accepted.
#         del self.in_place, self.trainable#, self.device
#         self.ts(self.init_start_str)
#         self.model = self.init_model()
#         self.ts(self.init_end_str)

#     def init_model(self) -> tensorpac.Pac:
#         resolution_dict = {10: "lres", 30: "mres", 50: "hres", 70: "demon", 100: "hulk"}
#         model = tensorpac.Pac(
#             f_pha=resolution_dict[self.pha_n_bands],
#             f_amp=resolution_dict[self.pha_n_bands],
#             dcomplex="wavelet",
#         )
#         model.idpac = (2, 0, 0)
#         return model

#     def calc_pac(self, xx: np.ndarray) -> np.ndarray:
#         return self._calc_pac_unfair_using_threads(xx) if self.use_threads else self._calc_pac_fair_chunk(xx)

#     def _calc_pac(self, xs: np.ndarray) -> np.ndarray:
#         pha = self.model.filter(self.fs, xs, ftype="phase", n_jobs=-1)
#         amp = self.model.filter(self.fs, xs, ftype="amplitude", n_jobs=-1)
#         xpac = self.model.fit(pha, amp, n_jobs=-1, verbose=False)
#         return xpac.transpose(2, 1, 0)

#     def _calc_pac_fair_chunk(self, xx: np.ndarray) -> np.ndarray:
#         """
#         xx.shape: (batch_size, n_chs, n_segments, seq_len).
#         This is suitable for fair comparisons between the MNGS's PAC calculation.
#         """

#         # Ensures the input properties
#         assert xx.ndim == 4
#         assert xx.dtype == np.float16 if self.fp16 else np.float32

#         xx = self.dim_handler.fit(xx, keepdims=[-1])  # fixme

#         # Batch processing to limit RAM/VRAM usage
#         n_chunks = math.ceil(len(xx) / self.chunk_size)
#         if self.chunk_size == 1:
#             # (Do not need to add the first dimension for consistency in numpy)
#             xpac = np.vstack(
#                 [
#                     self._calc_pac(
#                         xx[
#                             i_batch
#                             * self.chunk_size : (i_batch + 1)
#                             * self.chunk_size
#                         ]
#                     )
#                     for i_batch in range(n_chunks)
#                 ]
#             )
#         else:
#             xpac = np.vstack(
#                 [
#                     self._calc_pac(
#                         xx[
#                             i_batch
#                             * self.chunk_size : (i_batch + 1)
#                             * self.chunk_size
#                         ]
#                     )
#                     for i_batch in range(n_chunks)
#                 ]
#             )

#         xpac = self.dim_handler.unfit(xpac)

#         return xpac

#     def _calc_pac_unfair_using_threads(self, xx, n_jobs=-1) -> np.ndarray:
#         """
#         xx.shape: (batch_size, n_chs, n_segments, seq_len)
#         This is not suitable for fair comparisons between the MNGS's PAC calculation.
#         """

#         # Ensures the input properties
#         assert xx.ndim == 4
#         assert xx.dtype == np.float16 if self.fp16 else np.float32

#         xx = self.dim_handler.fit(xx, keepdims=np.array([-2, -1]))

#         # Processing using threads
#         batch_size_by_n_chs, n_segments, _ = xx.shape
#         xpac = []
#         with ThreadPoolExecutor(
#             max_workers=None
#         ) as executor:  # Using system's max processors
#             futures = []
#             for i_batch in range(batch_size_by_n_chs):
#                 futures.append(
#                     executor.submit(
#                         self._calc_pac,
#                         xx[i_batch],
#                     )
#                 )
#             for future in futures:
#                 xpac.append(future.result())

#         # Collect calculated xpacs
#         xpac = np.array(xpac)
#         xpac = self.dim_handler.unfit(xpac)

#         return xpac

#     @property
#     def freqs_amp(self) -> np.ndarray:
#         return self.model.f_amp.mean(axis=-1)

#     @property
#     def freqs_pha(self) -> np.ndarray:
#         return self.model.f_pha.mean(axis=-1)

#     def __str__(self) -> str:
#         return "Tensorpac"


#

# EOF
