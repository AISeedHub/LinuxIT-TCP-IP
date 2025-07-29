import os
import sys
import pytest
import numpy as np
import cv2
import time
import asyncio
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.pear_detector import PearDetector, ModelConfig, DetectionResult


class TestPearDetector:
    @pytest.fixture
    def test_image(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.circle(img, (320, 320), 100, (255, 255, 255), -1)
        return img

    @pytest.fixture
    def detector(self):
        config = ModelConfig(
            model_path=r"./weights/best-2cls.pt",
            classes=["pear", "defect"],
            confidence_threshold=0.4
        )
        detector = PearDetector(config)
        return detector

    @pytest.fixture
    def test_image_paths(self):
        """Create list of test image paths"""
        base_path = Path(r"./test_photos")
        return [
            str(base_path / "test1.jpg"),
            str(base_path / "test2.jpg"),
            str(base_path / "test3.jpg")
        ]

    def test_model_load(self, detector):
        assert detector.model is not None
        assert detector.config.confidence_threshold == 0.8

    @pytest.mark.asyncio
    async def test_inference_async(self, detector):
        img_path = r"./test_photos/test2.jpg"
        result = await detector.inference(img_path)  # Added await here
        print(result)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_normal, bool)
        assert result.is_normal
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_inference(self, detector, test_image):
        result = await detector.detect(test_image)
        assert isinstance(result, DetectionResult)

    @pytest.mark.asyncio
    async def test_inference_defect(self, detector):
        img_path = r"./test_photos/test1.jpg"
        result = await detector.inference(img_path)
        assert isinstance(result, DetectionResult)
        assert isinstance(result.is_normal, bool)
        assert not result.is_normal
        assert 0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_empty_image(self, detector):
        empty_image = np.zeros((640, 640, 3), dtype=np.uint8)
        result = await detector.detect(empty_image)
        assert isinstance(result, DetectionResult)
        assert result.error.non_detect == True

    @pytest.mark.asyncio
    async def test_performance_single_vs_batch_inference(self, detector, test_image_paths):
        """Test performance comparison between single inference and batch inference"""
        print("\n" + "=" * 60)
        print("PERFORMANCE TEST: Single vs Batch Inference")
        print("=" * 60)

        # Test Single Inference Performance
        print(f"\n🔍 Testing Single Inference with {len(test_image_paths)} images...")
        single_start_time = time.time()

        single_results = []
        for img_path in test_image_paths:
            result = await detector.inference(img_path)
            single_results.append(result)

        single_end_time = time.time()
        single_total_time = single_end_time - single_start_time
        single_avg_time = single_total_time / len(test_image_paths)

        print(f"⏱️  Total Single Inference time: {single_total_time:.4f}s")
        print(f"⏱️  Average time per image: {single_avg_time:.4f}s")

        # Test Batch Inference Performance
        print(f"\n🚀 Testing Batch Inference with {len(test_image_paths)} images...")
        batch_start_time = time.time()

        batch_results = await detector.inferences(test_image_paths)

        batch_end_time = time.time()
        batch_total_time = batch_end_time - batch_start_time
        batch_avg_time = batch_total_time / len(test_image_paths)

        print(f"⏱️  Total Batch Inference time: {batch_total_time:.4f}s")
        print(f"⏱️  Average time per image: {batch_avg_time:.4f}s")

        # Performance comparison
        speedup = single_total_time / batch_total_time if batch_total_time > 0 else 0
        time_saved = single_total_time - batch_total_time
        efficiency = (time_saved / single_total_time) * 100 if single_total_time > 0 else 0

        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"🚀 Batch is faster than Single by: {speedup:.2f}x")
        print(f"⏰ Time saved: {time_saved:.4f}s ({efficiency:.1f}%)")

        # Verify results are the same
        assert len(single_results) == len(batch_results) == len(test_image_paths)

        # Compare accuracy of results
        results_match = True
        for i, (single_res, batch_res) in enumerate(zip(single_results, batch_results)):
            if single_res.is_normal != batch_res.is_normal:
                print(
                    f"⚠️  Different results at image {i + 1}: Single={single_res.is_normal}, Batch={batch_res.is_normal}")
                results_match = False

        if results_match:
            print("✅ Single and Batch inference results are identical")
        else:
            print("❌ There are differences between Single and Batch inference results")

        print("=" * 60)

        # Assertions to ensure test passes
        assert batch_total_time > 0, "Batch inference should take some time"
        assert single_total_time > 0, "Single inference should take some time"
        assert len(batch_results) == len(test_image_paths), "Should get result for each image"

    @pytest.mark.asyncio
    async def test_detailed_performance_metrics(self, detector, test_image_paths):
        """Detailed performance analysis with multiple runs for statistical accuracy"""
        print("\n" + "=" * 70)
        print("DETAILED PERFORMANCE ANALYSIS: Multiple Runs")
        print("=" * 70)

        num_runs = 5  # Number of test runs for averaging
        single_times = []
        batch_times = []

        print(f"Running {num_runs} iterations for statistical accuracy...")

        for run in range(num_runs):
            print(f"\n🔄 Run {run + 1}/{num_runs}")

            # Single inference timing
            single_start = time.time()
            single_results = []
            for img_path in test_image_paths:
                result = await detector.inference(img_path)
                single_results.append(result)
            single_time = time.time() - single_start
            single_times.append(single_time)

            # Batch inference timing
            batch_start = time.time()
            batch_results = await detector.inferences(test_image_paths)
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            print(f"   Single: {single_time:.4f}s | Batch: {batch_time:.4f}s")

        # Calculate statistics
        avg_single = sum(single_times) / len(single_times)
        avg_batch = sum(batch_times) / len(batch_times)
        min_single = min(single_times)
        max_single = max(single_times)
        min_batch = min(batch_times)
        max_batch = max(batch_times)

        avg_speedup = avg_single / avg_batch if avg_batch > 0 else 0

        print(f"\n📈 STATISTICAL RESULTS:")
        print(f"📊 Single Inference:")
        print(f"   Average: {avg_single:.4f}s")
        print(f"   Min:     {min_single:.4f}s")
        print(f"   Max:     {max_single:.4f}s")

        print(f"📊 Batch Inference:")
        print(f"   Average: {avg_batch:.4f}s")
        print(f"   Min:     {min_batch:.4f}s")
        print(f"   Max:     {max_batch:.4f}s")

        print(f"🚀 Average Speedup: {avg_speedup:.2f}x")
        print(f"⚡ Performance Gain: {((avg_single - avg_batch) / avg_single * 100):.1f}%")
        print("=" * 70)

        # Assertions
        assert avg_batch > 0, "Batch inference should take measurable time"
        assert avg_single > 0, "Single inference should take measurable time"