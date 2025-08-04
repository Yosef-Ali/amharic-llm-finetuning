"""
Hybrid API Server for Amharic Language AI
Unified endpoints for generation and extraction
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️  Flask not available - API server disabled")

from hybrid.amharic_language_ai import AmharicLanguageAI, LanguageAIConfig, ProcessingMode
from hybrid.hybrid_workflows import WorkflowManager, WorkflowType
from hybrid.schema_aware_generation import SchemaAwareGenerator, GenerationConstraints
from hybrid.content_validator import ContentValidator, ValidationLevel


class HybridAPIServer:
    """API server for hybrid Amharic language processing."""
    
    def __init__(self, config: Optional[LanguageAIConfig] = None):
        self.config = config or LanguageAIConfig()
        self.language_ai = AmharicLanguageAI(self.config)
        self.workflow_manager = WorkflowManager(self.language_ai)
        self.schema_generator = SchemaAwareGenerator()
        self.validator = ContentValidator()
        
        # API statistics
        self.api_stats = {
            "total_requests": 0,
            "generation_requests": 0,
            "extraction_requests": 0,
            "hybrid_requests": 0,
            "workflow_requests": 0,
            "validation_requests": 0,
            "total_processing_time": 0.0,
            "average_response_time": 0.0
        }
        
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            CORS(self.app)  # Enable CORS for all domains
            self._setup_routes()
        else:
            self.app = None
            print("⚠️  Flask not available - API endpoints not registered")
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # Health check
        @self.app.route("/health", methods=["GET"])
        def health_check():
            return jsonify({
                "status": "healthy",
                "service": "Amharic Language AI",
                "version": "1.0.0",
                "components": {
                    "generation": self.language_ai.generator is not None,
                    "extraction": self.language_ai.extractor is not None,
                    "workflows": True,
                    "validation": True
                },
                "timestamp": time.time()
            })
        
        # Generation endpoints
        @self.app.route("/generate", methods=["POST"])
        def generate_text():
            return self._handle_generation_request()
        
        @self.app.route("/generate/schema-aware", methods=["POST"])
        def generate_schema_aware():
            return self._handle_schema_aware_generation()
        
        # Extraction endpoints
        @self.app.route("/extract", methods=["POST"])
        def extract_information():
            return self._handle_extraction_request()
        
        # Hybrid endpoints
        @self.app.route("/hybrid/generate-and-extract", methods=["POST"])
        def generate_and_extract():
            return self._handle_hybrid_request()
        
        @self.app.route("/hybrid/iterative-refinement", methods=["POST"])
        def iterative_refinement():
            return self._handle_refinement_request()
        
        # Workflow endpoints
        @self.app.route("/workflows", methods=["GET"])
        def list_workflows():
            return self._handle_list_workflows()
        
        @self.app.route("/workflows/<workflow_id>/execute", methods=["POST"])
        def execute_workflow(workflow_id):
            return self._handle_workflow_execution(workflow_id)
        
        @self.app.route("/workflows/batch", methods=["POST"])
        def batch_execute_workflow():
            return self._handle_batch_workflow_execution()
        
        # Validation endpoints
        @self.app.route("/validate", methods=["POST"])
        def validate_content():
            return self._handle_validation_request()
        
        # Analytics endpoints
        @self.app.route("/analytics", methods=["GET"])
        def get_analytics():
            return self._handle_analytics_request()
        
        @self.app.route("/analytics/workflows", methods=["GET"])
        def get_workflow_analytics():
            return jsonify(self.workflow_manager.get_workflow_analytics())
        
        # Domain and schema endpoints
        @self.app.route("/domains", methods=["GET"])
        def list_domains():
            from extraction.schemas import AMHARIC_SCHEMAS
            return jsonify({
                "domains": list(AMHARIC_SCHEMAS.keys()),
                "schemas": {
                    domain: {
                        "description": schema.get("description", ""),
                        "entity_types": list(schema.get("entities", {}).keys()),
                        "relationship_types": list(schema.get("relationships", {}).keys())
                    }
                    for domain, schema in AMHARIC_SCHEMAS.items()
                }
            })
        
        @self.app.route("/domains/<domain>/schema", methods=["GET"])
        def get_domain_schema(domain):
            from extraction.schemas import get_schema_by_domain
            try:
                schema = get_schema_by_domain(domain)
                return jsonify({"domain": domain, "schema": schema})
            except Exception as e:
                return jsonify({"error": str(e)}), 400
    
    def _handle_generation_request(self):
        """Handle text generation request."""
        try:
            data = request.get_json()
            if not data or "prompt" not in data:
                return jsonify({"error": "Missing 'prompt' field"}), 400
            
            prompt = data["prompt"]
            domain = data.get("domain", "news")
            
            # Optional parameters
            generation_params = {
                "max_length": data.get("max_length", self.config.max_generation_length),
                "temperature": data.get("temperature", self.config.generation_temperature),
                "generation_strategy": data.get("strategy", self.config.generation_strategy)
            }
            
            # Track request
            start_time = time.time()
            self.api_stats["total_requests"] += 1
            self.api_stats["generation_requests"] += 1
            
            # Generate text
            result = self.language_ai.generate_text(prompt, domain=domain, **generation_params)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.api_stats["total_processing_time"] += processing_time
            self.api_stats["average_response_time"] = self.api_stats["total_processing_time"] / self.api_stats["total_requests"]
            
            return jsonify({
                "status": "success",
                "result": {
                    "prompt": result.original_prompt,
                    "generated_text": result.generated_text,
                    "domain": result.domain,
                    "processing_mode": result.processing_mode.value,
                    "metadata": result.generation_metadata,
                    "processing_time": result.processing_time
                },
                "api_processing_time": processing_time
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def _handle_schema_aware_generation(self):
        """Handle schema-aware text generation."""
        try:
            data = request.get_json()
            if not data or "prompt" not in data:
                return jsonify({"error": "Missing 'prompt' field"}), 400
            
            prompt = data["prompt"]
            domain = data.get("domain", "news")
            
            # Parse constraints
            constraints_data = data.get("constraints", {})
            constraints = GenerationConstraints(
                required_entities=constraints_data.get("required_entities", []),
                minimum_entity_count=constraints_data.get("minimum_entity_count", {}),
                required_relationships=constraints_data.get("required_relationships", []),
                target_text_length=tuple(constraints_data.get("target_text_length", [100, 500])),
                domain_keywords=constraints_data.get("domain_keywords", [])
            )
            
            # Track request
            start_time = time.time()
            self.api_stats["total_requests"] += 1
            self.api_stats["generation_requests"] += 1
            
            # Generate schema-aware text
            result = self.language_ai.schema_guided_generation(
                prompt, 
                domain=domain,
                target_entities=constraints.required_entities
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.api_stats["total_processing_time"] += processing_time
            self.api_stats["average_response_time"] = self.api_stats["total_processing_time"] / self.api_stats["total_requests"]
            
            return jsonify({
                "status": "success",
                "result": {
                    "prompt": result.original_prompt,
                    "generated_text": result.generated_text,
                    "domain": result.domain,
                    "processing_mode": result.processing_mode.value,
                    "constraints": asdict(constraints),
                    "extraction_result": {
                        "entities": result.extraction_result.entities if result.extraction_result else {},
                        "relationships": result.extraction_result.relationships if result.extraction_result else [],
                        "quality_scores": result.quality_scores
                    },
                    "validation_passed": result.validation_passed,
                    "processing_time": result.processing_time
                },
                "api_processing_time": processing_time
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def _handle_extraction_request(self):
        """Handle information extraction request."""
        try:
            data = request.get_json()
            if not data or "text" not in data:
                return jsonify({"error": "Missing 'text' field"}), 400
            
            text = data["text"]
            domain = data.get("domain", "news")
            
            # Track request
            start_time = time.time()
            self.api_stats["total_requests"] += 1
            self.api_stats["extraction_requests"] += 1
            
            # Extract information
            result = self.language_ai.extract_information(text, domain=domain)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.api_stats["total_processing_time"] += processing_time
            self.api_stats["average_response_time"] = self.api_stats["total_processing_time"] / self.api_stats["total_requests"]
            
            return jsonify({
                "status": "success",
                "result": {
                    "text": result.original_prompt,
                    "domain": result.domain,
                    "processing_mode": result.processing_mode.value,
                    "extraction": {
                        "entities": result.extraction_result.entities if result.extraction_result else {},
                        "relationships": result.extraction_result.relationships if result.extraction_result else [],
                        "events": result.extraction_result.events if result.extraction_result else [],
                        "confidence_scores": result.extraction_result.confidence_scores if result.extraction_result else {},
                        "character_spans": result.extraction_result.character_spans if result.extraction_result else {},
                        "metadata": result.extraction_result.metadata if result.extraction_result else {}
                    },
                    "quality_scores": result.quality_scores,
                    "validation_passed": result.validation_passed,
                    "processing_time": result.processing_time
                },
                "api_processing_time": processing_time
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def _handle_hybrid_request(self):
        """Handle hybrid generation+extraction request."""
        try:
            data = request.get_json()
            if not data or "prompt" not in data:
                return jsonify({"error": "Missing 'prompt' field"}), 400
            
            prompt = data["prompt"]
            domain = data.get("domain", "news")
            
            # Track request
            start_time = time.time()
            self.api_stats["total_requests"] += 1
            self.api_stats["hybrid_requests"] += 1
            
            # Perform hybrid processing
            result = self.language_ai.generate_and_extract(prompt, domain=domain)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.api_stats["total_processing_time"] += processing_time
            self.api_stats["average_response_time"] = self.api_stats["total_processing_time"] / self.api_stats["total_requests"]
            
            return jsonify({
                "status": "success",
                "result": {
                    "prompt": result.original_prompt,
                    "domain": result.domain,
                    "processing_mode": result.processing_mode.value,
                    "generation": {
                        "generated_text": result.generated_text,
                        "metadata": result.generation_metadata
                    },
                    "extraction": {
                        "entities": result.extraction_result.entities if result.extraction_result else {},
                        "relationships": result.extraction_result.relationships if result.extraction_result else [],
                        "events": result.extraction_result.events if result.extraction_result else [],
                        "confidence_scores": result.extraction_result.confidence_scores if result.extraction_result else {},
                        "character_spans": result.extraction_result.character_spans if result.extraction_result else {}
                    },
                    "quality_scores": result.quality_scores,
                    "validation_passed": result.validation_passed,
                    "processing_time": result.processing_time
                },
                "api_processing_time": processing_time
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def _handle_refinement_request(self):
        """Handle iterative refinement request."""
        try:
            data = request.get_json()
            if not data or "prompt" not in data:
                return jsonify({"error": "Missing 'prompt' field"}), 400
            
            prompt = data["prompt"]
            domain = data.get("domain", "news")
            quality_target = data.get("quality_target", self.config.quality_threshold)
            
            # Track request
            start_time = time.time()
            self.api_stats["total_requests"] += 1
            self.api_stats["hybrid_requests"] += 1
            
            # Perform iterative refinement
            result = self.language_ai.iterative_refinement(
                prompt, 
                domain=domain,
                quality_target=quality_target
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.api_stats["total_processing_time"] += processing_time
            self.api_stats["average_response_time"] = self.api_stats["total_processing_time"] / self.api_stats["total_requests"]
            
            return jsonify({
                "status": "success",
                "result": {
                    "prompt": result.original_prompt,
                    "domain": result.domain,
                    "processing_mode": result.processing_mode.value,
                    "quality_target": quality_target,
                    "final_result": {
                        "generated_text": result.generated_text,
                        "entities": result.extraction_result.entities if result.extraction_result else {},
                        "relationships": result.extraction_result.relationships if result.extraction_result else []
                    },
                    "quality_scores": result.quality_scores,
                    "validation_passed": result.validation_passed,
                    "iterations_count": result.iterations_count,
                    "refinement_history": result.refinement_history,
                    "processing_time": result.processing_time
                },
                "api_processing_time": processing_time
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def _handle_list_workflows(self):
        """Handle workflow listing request."""
        workflows = {}
        for workflow_id, workflow in self.workflow_manager.workflows.items():
            workflows[workflow_id] = {
                "workflow_id": workflow.workflow_id,
                "workflow_type": workflow.workflow_type.value,
                "description": workflow.description,
                "steps_count": len(workflow.steps),
                "steps": [
                    {
                        "step_id": step.step_id,
                        "step_type": step.step_type,
                        "dependencies": step.dependencies
                    }
                    for step in workflow.steps
                ]
            }
        
        return jsonify({
            "workflows": workflows,
            "count": len(workflows)
        })
    
    def _handle_workflow_execution(self, workflow_id: str):
        """Handle workflow execution request."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "Missing input data"}), 400
            
            # Track request
            start_time = time.time()
            self.api_stats["total_requests"] += 1
            self.api_stats["workflow_requests"] += 1
            
            # Execute workflow
            result = self.workflow_manager.execute_workflow(workflow_id, data)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.api_stats["total_processing_time"] += processing_time
            self.api_stats["average_response_time"] = self.api_stats["total_processing_time"] / self.api_stats["total_requests"]
            
            # Convert results to serializable format
            serialized_results = {}
            for step_id, hybrid_result in result.results.items():
                serialized_results[step_id] = {
                    "processing_mode": hybrid_result.processing_mode.value,
                    "domain": hybrid_result.domain,
                    "generated_text": hybrid_result.generated_text,
                    "quality_scores": hybrid_result.quality_scores,
                    "validation_passed": hybrid_result.validation_passed,
                    "processing_time": hybrid_result.processing_time
                }
            
            return jsonify({
                "status": "success",
                "result": {
                    "workflow_id": result.workflow_id,
                    "workflow_type": result.workflow_type.value,
                    "execution_status": result.status,
                    "steps_completed": result.steps_completed,
                    "total_steps": result.total_steps,
                    "execution_time": result.execution_time,
                    "step_results": serialized_results,
                    "error_messages": result.error_messages
                },
                "api_processing_time": processing_time
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def _handle_batch_workflow_execution(self):
        """Handle batch workflow execution request."""
        try:
            data = request.get_json()
            if not data or "workflow_id" not in data or "input_batch" not in data:
                return jsonify({"error": "Missing 'workflow_id' or 'input_batch' fields"}), 400
            
            workflow_id = data["workflow_id"]
            input_batch = data["input_batch"]
            max_workers = data.get("max_workers", 4)
            
            # Track request
            start_time = time.time()
            self.api_stats["total_requests"] += 1
            self.api_stats["workflow_requests"] += 1
            
            # Execute batch workflow
            results = self.workflow_manager.batch_execute_workflow(
                workflow_id, input_batch, max_workers=max_workers
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.api_stats["total_processing_time"] += processing_time
            self.api_stats["average_response_time"] = self.api_stats["total_processing_time"] / self.api_stats["total_requests"]
            
            # Summarize results
            success_count = sum(1 for r in results if r.status == "success")
            
            return jsonify({
                "status": "success",
                "result": {
                    "workflow_id": workflow_id,
                    "batch_size": len(input_batch),
                    "success_count": success_count,
                    "success_rate": success_count / len(results) if results else 0.0,
                    "total_execution_time": sum(r.execution_time for r in results),
                    "average_execution_time": sum(r.execution_time for r in results) / len(results) if results else 0.0,
                    "individual_results": [
                        {
                            "status": r.status,
                            "steps_completed": r.steps_completed,
                            "execution_time": r.execution_time,
                            "error_messages": r.error_messages
                        }
                        for r in results
                    ]
                },
                "api_processing_time": processing_time
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def _handle_validation_request(self):
        """Handle content validation request."""
        try:
            data = request.get_json()
            if not data or "text" not in data:
                return jsonify({"error": "Missing 'text' field"}), 400
            
            text = data["text"]
            domain = data.get("domain", "news")
            validation_level = data.get("validation_level", "standard")
            
            # Set validation level
            level_mapping = {
                "minimal": ValidationLevel.MINIMAL,
                "standard": ValidationLevel.STANDARD,
                "strict": ValidationLevel.STRICT
            }
            self.validator.validation_level = level_mapping.get(validation_level, ValidationLevel.STANDARD)
            
            # Track request
            start_time = time.time()
            self.api_stats["total_requests"] += 1
            self.api_stats["validation_requests"] += 1
            
            # Validate content
            result = self.validator.validate_content(text, domain=domain)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.api_stats["total_processing_time"] += processing_time
            self.api_stats["average_response_time"] = self.api_stats["total_processing_time"] / self.api_stats["total_requests"]
            
            return jsonify({
                "status": "success",
                "result": {
                    "text": text,
                    "domain": domain,
                    "validation_level": validation_level,
                    "is_valid": result.is_valid,
                    "overall_score": result.overall_score,
                    "detailed_scores": {
                        "length_score": result.length_score,
                        "entity_score": result.entity_score,
                        "relationship_score": result.relationship_score,
                        "domain_score": result.domain_score,
                        "quality_score": result.quality_score,
                        "coherence_score": result.coherence_score
                    },
                    "issues": [
                        {
                            "issue_type": issue.issue_type.value,
                            "severity": issue.severity,
                            "message": issue.message,
                            "suggestion": issue.suggestion
                        }
                        for issue in result.issues
                    ],
                    "suggestions": result.suggestions,
                    "validation_time": result.validation_time
                },
                "api_processing_time": processing_time
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def _handle_analytics_request(self):
        """Handle analytics request."""
        language_ai_stats = self.language_ai.get_statistics()
        
        combined_stats = {
            "api_statistics": self.api_stats,
            "language_ai_statistics": language_ai_stats,
            "system_info": {
                "config": {
                    "generation_model_path": self.config.generation_model_path,
                    "extraction_model": self.config.extraction_model,
                    "default_domain": self.config.default_domain,
                    "quality_threshold": self.config.quality_threshold,
                    "max_refinement_iterations": self.config.max_refinement_iterations
                },
                "components_available": {
                    "generator": self.language_ai.generator is not None,
                    "extractor": self.language_ai.extractor is not None,
                    "workflows": True,
                    "validation": True
                }
            }
        }
        
        return jsonify(combined_stats)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
        """Run the API server."""
        if not self.app:
            print("❌ Flask not available - cannot start server")
            return
        
        print(f"🚀 Starting Amharic Language AI API Server")
        print(f"   Host: {host}")
        print(f"   Port: {port}")
        print(f"   Debug: {debug}")
        print()
        print("📋 Available endpoints:")
        print("   GET  /health - Health check")
        print("   POST /generate - Text generation")
        print("   POST /generate/schema-aware - Schema-aware generation")
        print("   POST /extract - Information extraction")
        print("   POST /hybrid/generate-and-extract - Hybrid processing")
        print("   POST /hybrid/iterative-refinement - Iterative refinement")
        print("   GET  /workflows - List workflows")
        print("   POST /workflows/<id>/execute - Execute workflow")
        print("   POST /workflows/batch - Batch workflow execution")
        print("   POST /validate - Content validation")
        print("   GET  /analytics - System analytics")
        print("   GET  /domains - Available domains")
        print()
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except Exception as e:
            print(f"❌ Failed to start server: {e}")


def create_hybrid_api_server(generation_model_path: str = None,
                           extraction_api_key: str = None,
                           **kwargs) -> HybridAPIServer:
    """Factory function to create hybrid API server."""
    
    config = LanguageAIConfig(
        generation_model_path=generation_model_path,
        extraction_api_key=extraction_api_key,
        **kwargs
    )
    
    return HybridAPIServer(config)


if __name__ == "__main__":
    # Demo server
    server = create_hybrid_api_server()
    server.run(debug=True)