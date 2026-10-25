import unittest
from unittest import mock

import aether.config as config
from aether.custom_kernels import conv_kernel


class TestChooseDweightSplitK(unittest.TestCase):
    """The split-K grid fills each resident block slot once, and no more."""

    def _split(self, limits, num_m_tiles, grid_x, grid_y, threads):
        with mock.patch.object(config, "_MP_LIMITS", limits):
            return conv_kernel.choose_dweight_split_k(num_m_tiles, grid_x, grid_y, threads)

    def test_split_fills_one_wave_of_resident_blocks(self):
        # (mp_count, max threads/MP), num_m_tiles, grid_x, grid_y, threads -> split
        cases = [
            # gfx1201: 32 WGPs x (2048 // 512) = 128 slots.
            ((32, 2048), 16384, 1, 1, 512, 128),  # single tile column takes every slot
            ((32, 2048), 16384, 5, 1, 512, 25),   # 125 blocks, not a ragged 2nd wave
            ((32, 2048), 4096, 9, 2, 512, 7),     # 126 blocks
            ((32, 2048), 16384, 5, 1, 256, 51),   # smaller blocks -> 256 slots
            ((80, 1536), 16384, 5, 1, 512, 48),   # 80 MPs x 3 blocks = 240 slots
            ((32, 2048), 16384, 20, 10, 512, 1),  # grid already exceeds the slots
            ((32, 2048), 3, 1, 1, 512, 3),        # can't split finer than M tiles
        ]
        for limits, num_m_tiles, grid_x, grid_y, threads, expected in cases:
            with self.subTest(limits=limits, num_m_tiles=num_m_tiles,
                              grid=(grid_x, grid_y), threads=threads):
                self.assertEqual(
                    self._split(limits, num_m_tiles, grid_x, grid_y, threads), expected
                )


if __name__ == "__main__":
    unittest.main()
