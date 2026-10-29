"""Checks for the remaining course algorithms and the single app slot."""

import unittest
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from main import IP, TOOLS, VISIBLE_CATEGORIES
from single_application import APPLICATION
from photo_fix import (assess_image, find_bright_damage, model_available,
                       run_photo_fix, save_stages)


class AlgorithmTests(unittest.TestCase):
    def test_one_photo_fix_application_is_registered(self):
        self.assertIsNotNone(APPLICATION)
        self.assertEqual(VISIBLE_CATEGORIES, (1, 2, 4, 5, 6, 8))
        self.assertEqual([tool["id"] for tool in TOOLS if tool["cat"] == 8],
                         ["single_application"])
        self.assertIn("median_filter", {tool["id"] for tool in TOOLS})

    def test_median_reduces_salt_and_pepper_noise(self):
        image = np.full((96, 96, 3), 120, np.uint8)
        noisy = image.copy()
        noisy[::5, ::5] = 255
        noisy[2::5, 2::5] = 0
        filtered = IP.median_filter(noisy, k=3)
        self.assertGreater(cv2.PSNR(image, filtered), cv2.PSNR(image, noisy))

    def test_morphology_opening_removes_small_bright_spot(self):
        image = np.zeros((35, 35), np.uint8)
        image[17, 17] = 255
        self.assertEqual(int(IP.opening(image, k=3)[17, 17]), 0)

    def test_sobel_detects_vertical_boundary(self):
        image = np.zeros((64, 64, 3), np.uint8)
        image[:, 32:] = 255
        edges = IP.sobel(image)
        self.assertGreater(float(edges[:, 31:33].mean()), 100)
        self.assertLess(float(edges[:, 5:10].mean()), 5)

    def test_photo_fix_repair_handles_thin_white_scratch(self):
        image = np.full((160, 200, 3), 115, np.uint8)
        cv2.line(image, (70, 30), (70, 110), (255, 255, 255), 2)
        self.assertGreater(int(find_bright_damage(image)[70, 70]), 0)
        result = run_photo_fix(image, mode=0, light=False, contrast=False,
                               denoise=False, sharpen=0)
        self.assertLess(float(result.final[40:100, 70].mean()), 150)
        self.assertIn("repaired", result.stages)
        self.assertFalse(result.colorized)
        with TemporaryDirectory() as directory:
            saved = save_stages(result, directory)
            self.assertTrue(saved["final"].is_file())
            self.assertTrue(saved["original"].is_file())

    def test_combined_mode_does_not_recolor_existing_color_photo(self):
        image = np.full((64, 80, 3), (40, 110, 180), np.uint8)
        result = run_photo_fix(image, mode=2)
        self.assertFalse(result.colorized)
        self.assertNotIn("colorized", result.stages)
        self.assertEqual(result.final.shape, image.shape)

    @unittest.skipUnless(model_available(), "DDColor ONNX chưa được tải")
    def test_ddcolor_produces_color_for_grayscale_photo(self):
        gray = np.tile(np.linspace(35, 220, 96, dtype=np.uint8), (72, 1))
        result = run_photo_fix(gray, mode=1)
        self.assertTrue(result.colorized)
        self.assertEqual(result.final.shape, (72, 96, 3))
        self.assertGreater(float(np.mean(np.abs(result.final[:, :, 0].astype(np.int16) -
                                            result.final[:, :, 2].astype(np.int16)))), 2)

    @unittest.skipUnless(model_available(), "DDColor ONNX chưa được tải")
    def test_combined_mode_recognizes_sepia_and_recolors_it(self):
        luminance = np.tile(np.linspace(35, 220, 96, dtype=np.uint8), (72, 1))
        lab = np.dstack((luminance, np.full_like(luminance, 138),
                         np.full_like(luminance, 150)))
        sepia = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.assertTrue(assess_image(sepia)["sepia"])
        result = run_photo_fix(sepia, mode=2)
        self.assertTrue(result.colorized)
        self.assertIn("colorized", result.stages)
        enhanced = result.stages["enhanced"]
        self.assertTrue(np.array_equal(enhanced[:, :, 0], enhanced[:, :, 2]))


if __name__ == "__main__":
    unittest.main()
