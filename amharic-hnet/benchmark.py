#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for Amharic H-Net model.

This script provides comprehensive benchmarking for the Amharic H-Net model,
including inference speed, memory usage, and throughput under various conditions.
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark class for Amharic H-Net model."""
    
    def __init__(
        self,
        model_path: str,
        output_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        sequence_lengths: List[int] = [32, 64, 128, 256, 512],
        warmup_iterations: int = 5,
        test_iterations: int = 20,
        onnx_path: Optional[str] = None,
        fp16: bool = False,
        int8: bool = False,
    ):
        """Initialize the benchmark.
        
        Args:
            model_path: Path to the model directory
            output_dir: Directory to save benchmark results
            device: Device to run the benchmark on (cuda or cpu)
            batch_sizes: List of batch sizes to benchmark
            sequence_lengths: List of sequence lengths to benchmark
            warmup_iterations: Number of warmup iterations
            test_iterations: Number of test iterations
            onnx_path: Path to ONNX model (if available)
            fp16: Whether to use FP16 precision
            int8: Whether to use INT8 precision
        """
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device
        self.batch_sizes = batch_sizes
        self.sequence_lengths = sequence_lengths
        self.warmup_iterations = warmup_iterations
        self.test_iterations = test_iterations
        self.onnx_path = onnx_path
        self.fp16 = fp16
        self.int8 = int8
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer."""
        try:
            from amharic_hnet.model import HNetTransformer
            from transformers import AutoTokenizer
            
            logger.info(f"Loading model from {self.model_path}")
            self.model = HNetTransformer.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Move model to device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Apply precision settings
            if self.fp16 and self.device == "cuda":
                logger.info("Using FP16 precision")
                self.model = self.model.half()
            
            if self.int8 and self.device == "cuda":
                logger.info("Using INT8 precision")
                try:
                    from torch.quantization import quantize_dynamic
                    self.model = quantize_dynamic(
                        self.model,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                except ImportError:
                    logger.warning("INT8 quantization not supported, falling back to default precision")
            
            # Load ONNX model if available
            self.onnx_model = None
            if self.onnx_path is not None:
                try:
                    import onnxruntime as ort
                    
                    logger.info(f"Loading ONNX model from {self.onnx_path}")
                    
                    # Set up ONNX runtime session
                    if self.device == "cuda":
                        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    else:
                        providers = ['CPUExecutionProvider']
                    
                    self.onnx_model = ort.InferenceSession(
                        self.onnx_path,
                        providers=providers
                    )
                except ImportError:
                    logger.warning("ONNX Runtime not available, skipping ONNX benchmarks")
                except Exception as e:
                    logger.warning(f"Failed to load ONNX model: {e}")
        except ImportError:
            logger.error("Failed to import required modules")
            sys.exit(1)
    
    def generate_dummy_input(self, batch_size: int, sequence_length: int):
        """Generate dummy input for benchmarking.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
        
        Returns:
            Tuple of input_ids and attention_mask
        """
        # Generate random input IDs
        input_ids = torch.randint(
            low=0,
            high=self.model.config.vocab_size,
            size=(batch_size, sequence_length),
            device=self.device
        )
        
        # Generate attention mask (all 1s for simplicity)
        attention_mask = torch.ones(
            (batch_size, sequence_length),
            dtype=torch.long,
            device=self.device
        )
        
        return input_ids, attention_mask
    
    def benchmark_forward(self, batch_size: int, sequence_length: int):
        """Benchmark forward pass.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking forward pass with batch_size={batch_size}, sequence_length={sequence_length}")
        
        # Generate dummy input
        input_ids, attention_mask = self.generate_dummy_input(batch_size, sequence_length)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.model(input_ids, attention_mask=attention_mask)
        
        # Benchmark
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        memory_before = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        
        for _ in range(self.test_iterations):
            with torch.no_grad():
                _ = self.model(input_ids, attention_mask=attention_mask)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        memory_after = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        latency = elapsed_time / self.test_iterations
        throughput = batch_size / latency
        memory_usage = (memory_after - memory_before) / 1024 / 1024  # MB
        
        return {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "latency": latency,
            "throughput": throughput,
            "memory_usage": memory_usage,
        }
    
    def benchmark_generate(self, batch_size: int, sequence_length: int, max_new_tokens: int = 32):
        """Benchmark text generation.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            max_new_tokens: Maximum number of new tokens to generate
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info(f"Benchmarking generation with batch_size={batch_size}, sequence_length={sequence_length}, max_new_tokens={max_new_tokens}")
        
        # Generate dummy input
        input_ids, attention_mask = self.generate_dummy_input(batch_size, sequence_length)
        
        # Warmup
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    num_return_sequences=1,
                )
        
        # Benchmark
        torch.cuda.synchronize() if self.device == "cuda" else None
        start_time = time.time()
        memory_before = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        
        for _ in range(self.test_iterations):
            with torch.no_grad():
                _ = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    num_return_sequences=1,
                )
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.time()
        memory_after = torch.cuda.memory_allocated() if self.device == "cuda" else 0
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        latency = elapsed_time / self.test_iterations
        tokens_per_second = (batch_size * max_new_tokens) / latency
        memory_usage = (memory_after - memory_before) / 1024 / 1024  # MB
        
        return {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "max_new_tokens": max_new_tokens,
            "latency": latency,
            "tokens_per_second": tokens_per_second,
            "memory_usage": memory_usage,
        }
    
    def benchmark_onnx(self, batch_size: int, sequence_length: int):
        """Benchmark ONNX model.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
        
        Returns:
            Dictionary with benchmark results or None if ONNX model is not available
        """
        if self.onnx_model is None:
            return None
        
        logger.info(f"Benchmarking ONNX model with batch_size={batch_size}, sequence_length={sequence_length}")
        
        # Generate dummy input
        input_ids, attention_mask = self.generate_dummy_input(batch_size, sequence_length)
        
        # Convert to numpy for ONNX Runtime
        input_ids_np = input_ids.cpu().numpy()
        attention_mask_np = attention_mask.cpu().numpy()
        
        # Get input names
        input_names = [input.name for input in self.onnx_model.get_inputs()]
        
        # Prepare inputs
        onnx_inputs = {}
        for name in input_names:
            if "input_ids" in name:
                onnx_inputs[name] = input_ids_np
            elif "attention_mask" in name:
                onnx_inputs[name] = attention_mask_np
        
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = self.onnx_model.run(None, onnx_inputs)
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(self.test_iterations):
            _ = self.onnx_model.run(None, onnx_inputs)
        
        end_time = time.time()
        
        # Calculate metrics
        elapsed_time = end_time - start_time
        latency = elapsed_time / self.test_iterations
        throughput = batch_size / latency
        
        return {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "latency": latency,
            "throughput": throughput,
        }
    
    def run_benchmarks(self):
        """Run all benchmarks.
        
        Returns:
            Dictionary with all benchmark results
        """
        results = {
            "model_info": {
                "model_path": self.model_path,
                "device": self.device,
                "fp16": self.fp16,
                "int8": self.int8,
                "onnx_path": self.onnx_path,
            },
            "forward": [],
            "generate": [],
            "onnx": [],
        }
        
        # Benchmark forward pass
        for batch_size in self.batch_sizes:
            for sequence_length in self.sequence_lengths:
                try:
                    result = self.benchmark_forward(batch_size, sequence_length)
                    results["forward"].append(result)
                except Exception as e:
                    logger.warning(f"Failed to benchmark forward pass with batch_size={batch_size}, sequence_length={sequence_length}: {e}")
        
        # Benchmark generation
        for batch_size in self.batch_sizes[:3]:  # Use smaller batch sizes for generation
            for sequence_length in self.sequence_lengths[:3]:  # Use smaller sequence lengths for generation
                try:
                    result = self.benchmark_generate(batch_size, sequence_length)
                    results["generate"].append(result)
                except Exception as e:
                    logger.warning(f"Failed to benchmark generation with batch_size={batch_size}, sequence_length={sequence_length}: {e}")
        
        # Benchmark ONNX model
        if self.onnx_model is not None:
            for batch_size in self.batch_sizes:
                for sequence_length in self.sequence_lengths:
                    try:
                        result = self.benchmark_onnx(batch_size, sequence_length)
                        if result is not None:
                            results["onnx"].append(result)
                    except Exception as e:
                        logger.warning(f"Failed to benchmark ONNX model with batch_size={batch_size}, sequence_length={sequence_length}: {e}")
        
        # Save results
        output_file = os.path.join(self.output_dir, "benchmark_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_file}")
        
        # Generate plots
        self.generate_plots(results)
        
        return results
    
    def generate_plots(self, results):
        """Generate plots from benchmark results.
        
        Args:
            results: Dictionary with benchmark results
        """
        # Plot forward pass latency vs batch size for different sequence lengths
        plt.figure(figsize=(10, 6))
        for sequence_length in self.sequence_lengths:
            data = [r for r in results["forward"] if r["sequence_length"] == sequence_length]
            if data:
                batch_sizes = [r["batch_size"] for r in data]
                latencies = [r["latency"] * 1000 for r in data]  # Convert to ms
                plt.plot(batch_sizes, latencies, marker="o", label=f"Seq Len = {sequence_length}")
        
        plt.xlabel("Batch Size")
        plt.ylabel("Latency (ms)")
        plt.title("Forward Pass Latency vs Batch Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "forward_latency_vs_batch_size.png"))
        
        # Plot forward pass throughput vs batch size for different sequence lengths
        plt.figure(figsize=(10, 6))
        for sequence_length in self.sequence_lengths:
            data = [r for r in results["forward"] if r["sequence_length"] == sequence_length]
            if data:
                batch_sizes = [r["batch_size"] for r in data]
                throughputs = [r["throughput"] for r in data]
                plt.plot(batch_sizes, throughputs, marker="o", label=f"Seq Len = {sequence_length}")
        
        plt.xlabel("Batch Size")
        plt.ylabel("Throughput (samples/s)")
        plt.title("Forward Pass Throughput vs Batch Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "forward_throughput_vs_batch_size.png"))
        
        # Plot generation tokens per second vs batch size for different sequence lengths
        plt.figure(figsize=(10, 6))
        for sequence_length in self.sequence_lengths[:3]:  # Use smaller sequence lengths for generation
            data = [r for r in results["generate"] if r["sequence_length"] == sequence_length]
            if data:
                batch_sizes = [r["batch_size"] for r in data]
                tokens_per_second = [r["tokens_per_second"] for r in data]
                plt.plot(batch_sizes, tokens_per_second, marker="o", label=f"Seq Len = {sequence_length}")
        
        plt.xlabel("Batch Size")
        plt.ylabel("Tokens per Second")
        plt.title("Generation Tokens per Second vs Batch Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "generation_tokens_per_second_vs_batch_size.png"))
        
        # Plot memory usage vs batch size for different sequence lengths
        plt.figure(figsize=(10, 6))
        for sequence_length in self.sequence_lengths:
            data = [r for r in results["forward"] if r["sequence_length"] == sequence_length]
            if data:
                batch_sizes = [r["batch_size"] for r in data]
                memory_usages = [r["memory_usage"] for r in data]
                plt.plot(batch_sizes, memory_usages, marker="o", label=f"Seq Len = {sequence_length}")
        
        plt.xlabel("Batch Size")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Usage vs Batch Size")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "memory_usage_vs_batch_size.png"))
        
        # Compare PyTorch vs ONNX latency if ONNX results are available
        if results["onnx"]:
            plt.figure(figsize=(10, 6))
            for sequence_length in self.sequence_lengths:
                pytorch_data = [r for r in results["forward"] if r["sequence_length"] == sequence_length]
                onnx_data = [r for r in results["onnx"] if r["sequence_length"] == sequence_length]
                
                if pytorch_data and onnx_data:
                    batch_sizes = [r["batch_size"] for r in pytorch_data]
                    pytorch_latencies = [r["latency"] * 1000 for r in pytorch_data]  # Convert to ms
                    onnx_latencies = [r["latency"] * 1000 for r in onnx_data]  # Convert to ms
                    
                    plt.plot(batch_sizes, pytorch_latencies, marker="o", linestyle="-", label=f"PyTorch, Seq Len = {sequence_length}")
                    plt.plot(batch_sizes, onnx_latencies, marker="s", linestyle="--", label=f"ONNX, Seq Len = {sequence_length}")
            
            plt.xlabel("Batch Size")
            plt.ylabel("Latency (ms)")
            plt.title("PyTorch vs ONNX Latency")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "pytorch_vs_onnx_latency.png"))
        
        logger.info(f"Plots saved to {self.output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Benchmark Amharic H-Net model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the benchmark on (cuda or cpu)",
    )
    parser.add_argument(
        "--batch_sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32],
        help="List of batch sizes to benchmark",
    )
    parser.add_argument(
        "--sequence_lengths",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
        help="List of sequence lengths to benchmark",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--test_iterations",
        type=int,
        default=20,
        help="Number of test iterations",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="Path to ONNX model (if available)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use FP16 precision",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 precision",
    )
    
    args = parser.parse_args()
    
    # Create benchmark
    benchmark = ModelBenchmark(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.sequence_lengths,
        warmup_iterations=args.warmup_iterations,
        test_iterations=args.test_iterations,
        onnx_path=args.onnx_path,
        fp16=args.fp16,
        int8=args.int8,
    )
    
    # Run benchmarks
    benchmark.run_benchmarks()


if __name__ == "__main__":
    main()