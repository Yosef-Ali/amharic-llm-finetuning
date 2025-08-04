#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model optimization script for Amharic H-Net model.

This script provides functions for optimizing the model for deployment,
including ONNX conversion, quantization, and pruning.
"""

import os
import time
import logging
import argparse
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Optimizer for Amharic H-Net model."""
    
    def __init__(self, 
                 model_path: Union[str, Path],
                 output_dir: Optional[Union[str, Path]] = None,
                 device: str = "cpu"):
        """Initialize optimizer.
        
        Args:
            model_path: Path to model directory
            output_dir: Directory to save optimized model
            device: Device to use for optimization
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir) if output_dir else self.model_path / "optimized"
        self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model and tokenizer
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()
    
    def _load_model_and_tokenizer(self) -> None:
        """Load model and tokenizer."""
        try:
            # Try to import HNetTransformer
            from amharic_hnet.model import HNetTransformer
            
            # Load model
            logger.info(f"Loading model from {self.model_path}")
            self.model = HNetTransformer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            logger.info(f"Model and tokenizer loaded successfully")
        except ImportError:
            logger.error(f"Failed to import HNetTransformer. Make sure the amharic_hnet package is installed.")
            raise
        except Exception as e:
            logger.error(f"Failed to load model and tokenizer: {e}")
            raise
    
    def convert_to_onnx(self, 
                        output_path: Optional[Union[str, Path]] = None,
                        opset_version: int = 12,
                        dynamic_axes: bool = True) -> Path:
        """Convert model to ONNX format.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Whether to use dynamic axes
            
        Returns:
            Path to ONNX model
        """
        if self.model is None:
            logger.error("Model not loaded")
            return None
        
        # Set output path
        if output_path is None:
            output_path = self.output_dir / "model.onnx"
        else:
            output_path = Path(output_path)
        
        # Create dummy input
        batch_size = 1
        sequence_length = 64
        dummy_input_ids = torch.ones(batch_size, sequence_length, dtype=torch.long).to(self.device)
        dummy_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long).to(self.device)
        
        # Set dynamic axes
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size", 1: "sequence_length"},
            }
        
        # Export model to ONNX
        try:
            logger.info(f"Converting model to ONNX format")
            torch.onnx.export(
                self.model,
                (dummy_input_ids, dummy_attention_mask),
                output_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["output"],
                dynamic_axes=dynamic_axes_dict,
                opset_version=opset_version,
                do_constant_folding=True,
                export_params=True,
                verbose=False,
            )
            logger.info(f"Model converted to ONNX format and saved to {output_path}")
            
            # Verify ONNX model
            self._verify_onnx_model(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"Failed to convert model to ONNX format: {e}")
            return None
    
    def _verify_onnx_model(self, onnx_path: Union[str, Path]) -> bool:
        """Verify ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            
        Returns:
            Whether verification was successful
        """
        try:
            import onnx
            
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Check model
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"ONNX model verification successful")
            return True
        except ImportError:
            logger.warning(f"ONNX package not installed. Skipping verification.")
            return False
        except Exception as e:
            logger.error(f"ONNX model verification failed: {e}")
            return False
    
    def optimize_onnx_model(self, 
                           onnx_path: Union[str, Path],
                           output_path: Optional[Union[str, Path]] = None) -> Path:
        """Optimize ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save optimized ONNX model
            
        Returns:
            Path to optimized ONNX model
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            from onnxruntime.transformers.onnx_model_bert import BertOptimizationOptions
            
            # Set output path
            if output_path is None:
                output_path = Path(onnx_path).parent / f"{Path(onnx_path).stem}_optimized.onnx"
            else:
                output_path = Path(output_path)
            
            # Configure optimization options
            optimization_options = BertOptimizationOptions('bert')
            optimization_options.enable_embed_layer_norm = False
            
            # Optimize model
            logger.info(f"Optimizing ONNX model")
            optimized_model = optimizer.optimize_model(
                str(onnx_path),
                'bert',
                num_heads=self.model.config.num_attention_heads,
                hidden_size=self.model.config.hidden_size,
                optimization_options=optimization_options
            )
            
            # Save optimized model
            optimized_model.save_model_to_file(str(output_path))
            
            logger.info(f"ONNX model optimized and saved to {output_path}")
            return output_path
        except ImportError:
            logger.warning(f"ONNX Runtime Transformers package not installed. Skipping optimization.")
            return onnx_path
        except Exception as e:
            logger.error(f"Failed to optimize ONNX model: {e}")
            return onnx_path
    
    def quantize_onnx_model(self, 
                           onnx_path: Union[str, Path],
                           output_path: Optional[Union[str, Path]] = None,
                           quantization_type: str = "dynamic") -> Path:
        """Quantize ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Path to save quantized ONNX model
            quantization_type: Type of quantization (dynamic or static)
            
        Returns:
            Path to quantized ONNX model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Set output path
            if output_path is None:
                output_path = Path(onnx_path).parent / f"{Path(onnx_path).stem}_quantized.onnx"
            else:
                output_path = Path(output_path)
            
            # Quantize model
            logger.info(f"Quantizing ONNX model using {quantization_type} quantization")
            if quantization_type == "dynamic":
                quantize_dynamic(
                    str(onnx_path),
                    str(output_path),
                    weight_type=QuantType.QInt8
                )
            else:
                logger.warning(f"Static quantization not implemented yet. Using dynamic quantization.")
                quantize_dynamic(
                    str(onnx_path),
                    str(output_path),
                    weight_type=QuantType.QInt8
                )
            
            logger.info(f"ONNX model quantized and saved to {output_path}")
            return output_path
        except ImportError:
            logger.warning(f"ONNX Runtime Quantization package not installed. Skipping quantization.")
            return onnx_path
        except Exception as e:
            logger.error(f"Failed to quantize ONNX model: {e}")
            return onnx_path
    
    def benchmark_onnx_model(self, 
                            onnx_path: Union[str, Path],
                            num_iterations: int = 100,
                            warmup_iterations: int = 10,
                            sequence_length: int = 64) -> Dict[str, float]:
        """Benchmark ONNX model.
        
        Args:
            onnx_path: Path to ONNX model
            num_iterations: Number of iterations
            warmup_iterations: Number of warmup iterations
            sequence_length: Sequence length
            
        Returns:
            Benchmark results
        """
        try:
            import onnxruntime as ort
            
            # Create ONNX Runtime session
            logger.info(f"Creating ONNX Runtime session")
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(str(onnx_path), session_options)
            
            # Create dummy input
            batch_size = 1
            dummy_input_ids = np.ones((batch_size, sequence_length), dtype=np.int64)
            dummy_attention_mask = np.ones((batch_size, sequence_length), dtype=np.int64)
            
            # Prepare input
            ort_inputs = {
                "input_ids": dummy_input_ids,
                "attention_mask": dummy_attention_mask,
            }
            
            # Warmup
            logger.info(f"Warming up for {warmup_iterations} iterations")
            for _ in range(warmup_iterations):
                session.run(None, ort_inputs)
            
            # Benchmark
            logger.info(f"Benchmarking for {num_iterations} iterations")
            latencies = []
            for _ in range(num_iterations):
                start_time = time.time()
                session.run(None, ort_inputs)
                latencies.append((time.time() - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p90_latency = np.percentile(latencies, 90)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            
            # Log results
            logger.info(f"Benchmark results:")
            logger.info(f"  Average latency: {avg_latency:.2f} ms")
            logger.info(f"  P50 latency: {p50_latency:.2f} ms")
            logger.info(f"  P90 latency: {p90_latency:.2f} ms")
            logger.info(f"  P95 latency: {p95_latency:.2f} ms")
            logger.info(f"  P99 latency: {p99_latency:.2f} ms")
            
            return {
                "avg_latency": avg_latency,
                "p50_latency": p50_latency,
                "p90_latency": p90_latency,
                "p95_latency": p95_latency,
                "p99_latency": p99_latency,
            }
        except ImportError:
            logger.warning(f"ONNX Runtime package not installed. Skipping benchmark.")
            return {}
        except Exception as e:
            logger.error(f"Failed to benchmark ONNX model: {e}")
            return {}
    
    def prune_model(self, 
                   pruning_method: str = "magnitude",
                   pruning_ratio: float = 0.3) -> None:
        """Prune model.
        
        Args:
            pruning_method: Pruning method
            pruning_ratio: Pruning ratio
        """
        try:
            import torch.nn.utils.prune as prune
            
            if self.model is None:
                logger.error("Model not loaded")
                return
            
            # Get prunable parameters
            prunable_modules = []
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prunable_modules.append((name, module))
            
            # Prune model
            logger.info(f"Pruning model using {pruning_method} pruning with ratio {pruning_ratio}")
            for name, module in prunable_modules:
                if pruning_method == "magnitude":
                    prune.l1_unstructured(module, name="weight", amount=pruning_ratio)
                elif pruning_method == "random":
                    prune.random_unstructured(module, name="weight", amount=pruning_ratio)
                else:
                    logger.warning(f"Unknown pruning method: {pruning_method}. Using magnitude pruning.")
                    prune.l1_unstructured(module, name="weight", amount=pruning_ratio)
            
            # Make pruning permanent
            for name, module in prunable_modules:
                prune.remove(module, "weight")
            
            logger.info(f"Model pruned successfully")
            
            # Save pruned model
            pruned_model_path = self.output_dir / "pruned_model"
            self.model.save_pretrained(pruned_model_path)
            self.tokenizer.save_pretrained(pruned_model_path)
            
            logger.info(f"Pruned model saved to {pruned_model_path}")
        except ImportError:
            logger.warning(f"PyTorch pruning package not installed. Skipping pruning.")
        except Exception as e:
            logger.error(f"Failed to prune model: {e}")
    
    def quantize_model(self, 
                      quantization_method: str = "dynamic",
                      quantization_bits: int = 8) -> None:
        """Quantize model.
        
        Args:
            quantization_method: Quantization method
            quantization_bits: Quantization bits
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return
            
            # Quantize model
            logger.info(f"Quantizing model using {quantization_method} quantization with {quantization_bits} bits")
            if quantization_method == "dynamic":
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            elif quantization_method == "static":
                logger.warning(f"Static quantization not implemented yet. Using dynamic quantization.")
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            else:
                logger.warning(f"Unknown quantization method: {quantization_method}. Using dynamic quantization.")
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            
            # Save quantized model
            quantized_model_path = self.output_dir / "quantized_model"
            quantized_model.save_pretrained(quantized_model_path)
            self.tokenizer.save_pretrained(quantized_model_path)
            
            logger.info(f"Quantized model saved to {quantized_model_path}")
        except Exception as e:
            logger.error(f"Failed to quantize model: {e}")
    
    def optimize_for_inference(self) -> None:
        """Optimize model for inference."""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return
            
            # Optimize model for inference
            logger.info(f"Optimizing model for inference")
            
            # Freeze model
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Save optimized model
            optimized_model_path = self.output_dir / "optimized_model"
            self.model.save_pretrained(optimized_model_path)
            self.tokenizer.save_pretrained(optimized_model_path)
            
            logger.info(f"Optimized model saved to {optimized_model_path}")
        except Exception as e:
            logger.error(f"Failed to optimize model for inference: {e}")
    
    def optimize_all(self) -> None:
        """Optimize model using all available methods."""
        # Optimize for inference
        self.optimize_for_inference()
        
        # Convert to ONNX
        onnx_path = self.convert_to_onnx()
        
        if onnx_path:
            # Optimize ONNX model
            optimized_onnx_path = self.optimize_onnx_model(onnx_path)
            
            # Quantize ONNX model
            quantized_onnx_path = self.quantize_onnx_model(optimized_onnx_path)
            
            # Benchmark ONNX model
            self.benchmark_onnx_model(quantized_onnx_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Optimize Amharic H-Net model")
    
    # Input arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--output_dir", type=str, help="Directory to save optimized model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for optimization")
    
    # Optimization arguments
    parser.add_argument("--optimize_all", action="store_true", help="Optimize model using all available methods")
    parser.add_argument("--optimize_for_inference", action="store_true", help="Optimize model for inference")
    parser.add_argument("--convert_to_onnx", action="store_true", help="Convert model to ONNX format")
    parser.add_argument("--optimize_onnx", action="store_true", help="Optimize ONNX model")
    parser.add_argument("--quantize_onnx", action="store_true", help="Quantize ONNX model")
    parser.add_argument("--benchmark_onnx", action="store_true", help="Benchmark ONNX model")
    parser.add_argument("--prune_model", action="store_true", help="Prune model")
    parser.add_argument("--quantize_model", action="store_true", help="Quantize model")
    
    # ONNX arguments
    parser.add_argument("--opset_version", type=int, default=12, help="ONNX opset version")
    parser.add_argument("--dynamic_axes", action="store_true", help="Use dynamic axes for ONNX model")
    parser.add_argument("--onnx_path", type=str, help="Path to ONNX model")
    
    # Pruning arguments
    parser.add_argument("--pruning_method", type=str, default="magnitude", choices=["magnitude", "random"], help="Pruning method")
    parser.add_argument("--pruning_ratio", type=float, default=0.3, help="Pruning ratio")
    
    # Quantization arguments
    parser.add_argument("--quantization_method", type=str, default="dynamic", choices=["dynamic", "static"], help="Quantization method")
    parser.add_argument("--quantization_bits", type=int, default=8, choices=[8, 16], help="Quantization bits")
    
    # Benchmark arguments
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations for benchmark")
    parser.add_argument("--warmup_iterations", type=int, default=10, help="Number of warmup iterations for benchmark")
    parser.add_argument("--sequence_length", type=int, default=64, help="Sequence length for benchmark")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = ModelOptimizer(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Optimize model
    if args.optimize_all:
        optimizer.optimize_all()
    else:
        if args.optimize_for_inference:
            optimizer.optimize_for_inference()
        
        if args.convert_to_onnx:
            onnx_path = optimizer.convert_to_onnx(
                opset_version=args.opset_version,
                dynamic_axes=args.dynamic_axes
            )
            args.onnx_path = onnx_path
        
        if args.optimize_onnx and args.onnx_path:
            optimized_onnx_path = optimizer.optimize_onnx_model(args.onnx_path)
            args.onnx_path = optimized_onnx_path
        
        if args.quantize_onnx and args.onnx_path:
            quantized_onnx_path = optimizer.quantize_onnx_model(
                args.onnx_path,
                quantization_type=args.quantization_method
            )
            args.onnx_path = quantized_onnx_path
        
        if args.benchmark_onnx and args.onnx_path:
            optimizer.benchmark_onnx_model(
                args.onnx_path,
                num_iterations=args.num_iterations,
                warmup_iterations=args.warmup_iterations,
                sequence_length=args.sequence_length
            )
        
        if args.prune_model:
            optimizer.prune_model(
                pruning_method=args.pruning_method,
                pruning_ratio=args.pruning_ratio
            )
        
        if args.quantize_model:
            optimizer.quantize_model(
                quantization_method=args.quantization_method,
                quantization_bits=args.quantization_bits
            )


if __name__ == "__main__":
    main()