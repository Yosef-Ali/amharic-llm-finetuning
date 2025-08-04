"""
Hybrid Workflows for Amharic Language AI
Advanced workflows combining generation and extraction
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from .amharic_language_ai import AmharicLanguageAI, HybridResult, ProcessingMode


class WorkflowType(Enum):
    """Types of hybrid workflows."""
    CONTENT_CREATION = "content_creation"
    DOCUMENT_ANALYSIS = "document_analysis"
    QUALITY_ASSURANCE = "quality_assurance"
    KNOWLEDGE_EXTRACTION = "knowledge_extraction"
    INTERACTIVE_REFINEMENT = "interactive_refinement"
    BATCH_PROCESSING = "batch_processing"


@dataclass
class WorkflowStep:
    """Individual step in a hybrid workflow."""
    step_id: str
    step_type: str  # "generate", "extract", "validate", "refine"
    parameters: Dict[str, Any]
    dependencies: List[str] = None  # List of step IDs this depends on
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class WorkflowResult:
    """Result from executing a hybrid workflow."""
    workflow_id: str
    workflow_type: WorkflowType
    status: str  # "success", "failed", "partial"
    steps_completed: int
    total_steps: int
    results: Dict[str, HybridResult]
    execution_time: float
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


class HybridWorkflow:
    """Defines a hybrid workflow combining generation and extraction."""
    
    def __init__(self, 
                 workflow_id: str,
                 workflow_type: WorkflowType,
                 steps: List[WorkflowStep],
                 description: str = ""):
        self.workflow_id = workflow_id
        self.workflow_type = workflow_type
        self.steps = steps
        self.description = description
        
        # Build dependency graph
        self.dependency_graph = self._build_dependency_graph()
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build dependency graph for workflow execution order."""
        graph = {}
        
        for step in self.steps:
            graph[step.step_id] = step.dependencies.copy()
        
        return graph
    
    def get_execution_order(self) -> List[str]:
        """Get execution order based on dependencies."""
        executed = set()
        order = []
        
        while len(executed) < len(self.steps):
            for step_id, dependencies in self.dependency_graph.items():
                if step_id not in executed and all(dep in executed for dep in dependencies):
                    order.append(step_id)
                    executed.add(step_id)
                    break
            else:
                # Circular dependency or missing dependency
                remaining = [s for s in self.dependency_graph.keys() if s not in executed]
                raise ValueError(f"Circular dependency or missing steps: {remaining}")
        
        return order


class WorkflowManager:
    """Manages and executes hybrid workflows."""
    
    def __init__(self, language_ai: AmharicLanguageAI):
        self.language_ai = language_ai
        self.workflows = {}
        self.execution_history = []
        
        # Register built-in workflows
        self._register_builtin_workflows()
    
    def _register_builtin_workflows(self):
        """Register built-in workflow templates."""
        
        # Content Creation Workflow
        content_creation = HybridWorkflow(
            workflow_id="content_creation",
            workflow_type=WorkflowType.CONTENT_CREATION,
            description="Generate content then validate through extraction",
            steps=[
                WorkflowStep(
                    step_id="generate",
                    step_type="generate",
                    parameters={"domain": "news", "strategy": "sampling"}
                ),
                WorkflowStep(
                    step_id="extract",
                    step_type="extract",
                    parameters={"validate_entities": True},
                    dependencies=["generate"]
                ),
                WorkflowStep(
                    step_id="validate",
                    step_type="validate",
                    parameters={"quality_threshold": 0.7},
                    dependencies=["extract"]
                )
            ]
        )
        
        # Document Analysis Workflow
        document_analysis = HybridWorkflow(
            workflow_id="document_analysis",
            workflow_type=WorkflowType.DOCUMENT_ANALYSIS,
            description="Extract information then generate summary",
            steps=[
                WorkflowStep(
                    step_id="extract",
                    step_type="extract",
                    parameters={"comprehensive": True}
                ),
                WorkflowStep(
                    step_id="summarize",
                    step_type="generate",
                    parameters={"task": "summary", "max_length": 200},
                    dependencies=["extract"]
                )
            ]
        )
        
        # Quality Assurance Workflow
        quality_assurance = HybridWorkflow(
            workflow_id="quality_assurance",
            workflow_type=WorkflowType.QUALITY_ASSURANCE,
            description="Iterative generation with quality validation",
            steps=[
                WorkflowStep(
                    step_id="initial_generate",
                    step_type="generate",
                    parameters={"domain": "news"}
                ),
                WorkflowStep(
                    step_id="extract_validate",
                    step_type="extract",
                    parameters={"quality_check": True},
                    dependencies=["initial_generate"]
                ),
                WorkflowStep(
                    step_id="refine_generate",
                    step_type="refine",
                    parameters={"iterations": 3, "quality_target": 0.8},
                    dependencies=["extract_validate"]
                )
            ]
        )
        
        # Register workflows
        self.workflows["content_creation"] = content_creation
        self.workflows["document_analysis"] = document_analysis
        self.workflows["quality_assurance"] = quality_assurance
    
    def register_workflow(self, workflow: HybridWorkflow):
        """Register a custom workflow."""
        self.workflows[workflow.workflow_id] = workflow
        print(f"✅ Registered workflow: {workflow.workflow_id}")
    
    def execute_workflow(self, 
                        workflow_id: str, 
                        input_data: Dict[str, Any],
                        **kwargs) -> WorkflowResult:
        """Execute a workflow with given input data."""
        
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        start_time = time.time()
        
        print(f"🚀 Executing workflow: {workflow_id}")
        print(f"   Type: {workflow.workflow_type.value}")
        print(f"   Steps: {len(workflow.steps)}")
        
        result = WorkflowResult(
            workflow_id=workflow_id,
            workflow_type=workflow.workflow_type,
            status="running",
            steps_completed=0,
            total_steps=len(workflow.steps),
            results={},
            execution_time=0.0
        )
        
        try:
            # Get execution order
            execution_order = workflow.get_execution_order()
            step_results = {}
            
            # Execute steps in order
            for step_id in execution_order:
                step = next(s for s in workflow.steps if s.step_id == step_id)
                
                print(f"   📋 Executing step: {step.step_id} ({step.step_type})")
                
                # Execute step based on type
                step_result = self._execute_step(step, input_data, step_results, **kwargs)
                
                if step_result:
                    step_results[step_id] = step_result
                    result.results[step_id] = step_result
                    result.steps_completed += 1
                else:
                    result.error_messages.append(f"Step {step_id} failed")
                    result.status = "partial"
            
            # Determine final status
            if result.steps_completed == result.total_steps:
                result.status = "success"
            elif result.steps_completed == 0:
                result.status = "failed"
            
        except Exception as e:
            result.status = "failed"
            result.error_messages.append(str(e))
            print(f"❌ Workflow execution failed: {e}")
        
        # Finalize result
        result.execution_time = time.time() - start_time
        self.execution_history.append(result)
        
        print(f"✅ Workflow completed: {result.status}")
        print(f"   Steps completed: {result.steps_completed}/{result.total_steps}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        
        return result
    
    def _execute_step(self, 
                     step: WorkflowStep, 
                     input_data: Dict[str, Any],
                     previous_results: Dict[str, HybridResult],
                     **kwargs) -> Optional[HybridResult]:
        """Execute a single workflow step."""
        
        try:
            # Prepare step input
            step_input = self._prepare_step_input(step, input_data, previous_results)
            
            if step.step_type == "generate":
                return self._execute_generate_step(step, step_input, **kwargs)
            elif step.step_type == "extract":
                return self._execute_extract_step(step, step_input, **kwargs)
            elif step.step_type == "validate":
                return self._execute_validate_step(step, step_input, previous_results, **kwargs)
            elif step.step_type == "refine":
                return self._execute_refine_step(step, step_input, **kwargs)
            else:
                print(f"⚠️  Unknown step type: {step.step_type}")
                return None
        
        except Exception as e:
            print(f"❌ Step execution failed: {e}")
            return None
    
    def _prepare_step_input(self, 
                           step: WorkflowStep,
                           input_data: Dict[str, Any],
                           previous_results: Dict[str, HybridResult]) -> Dict[str, Any]:
        """Prepare input data for a step based on dependencies."""
        
        step_input = input_data.copy()
        
        # Add results from dependency steps
        for dep_id in step.dependencies:
            if dep_id in previous_results:
                dep_result = previous_results[dep_id]
                step_input[f"{dep_id}_result"] = dep_result
                
                # Add commonly used fields
                if dep_result.generated_text:
                    step_input["previous_text"] = dep_result.generated_text
                if dep_result.extraction_result:
                    step_input["previous_extraction"] = dep_result.extraction_result
        
        return step_input
    
    def _execute_generate_step(self, 
                              step: WorkflowStep, 
                              step_input: Dict[str, Any],
                              **kwargs) -> HybridResult:
        """Execute a generation step."""
        
        prompt = step_input.get("prompt", step_input.get("text", ""))
        domain = step.parameters.get("domain", "news")
        
        # Handle special generation tasks
        task = step.parameters.get("task", "generation")
        
        if task == "summary" and "previous_extraction" in step_input:
            # Generate summary based on extracted information
            extraction = step_input["previous_extraction"]
            entities = extraction.entities
            summary_prompt = f"በዚህ ጽሑፍ ውስጥ የሚከተሉት ዋና ዋና ነጥቦች አሉ: {entities}. እባክዎ አጭር ማጠቃለያ ይስጡ።"
            prompt = summary_prompt
        
        return self.language_ai.generate_text(prompt, domain=domain, **step.parameters, **kwargs)
    
    def _execute_extract_step(self, 
                             step: WorkflowStep, 
                             step_input: Dict[str, Any],
                             **kwargs) -> HybridResult:
        """Execute an extraction step."""
        
        text = step_input.get("previous_text", step_input.get("text", ""))
        domain = step.parameters.get("domain", "news")
        
        return self.language_ai.extract_information(text, domain=domain, **step.parameters, **kwargs)
    
    def _execute_validate_step(self, 
                              step: WorkflowStep, 
                              step_input: Dict[str, Any],
                              previous_results: Dict[str, HybridResult],
                              **kwargs) -> HybridResult:
        """Execute a validation step."""
        
        quality_threshold = step.parameters.get("quality_threshold", 0.7)
        
        # Get the most recent extraction result
        extraction_result = None
        for dep_id in reversed(step.dependencies):
            if dep_id in previous_results and previous_results[dep_id].extraction_result:
                extraction_result = previous_results[dep_id]
                break
        
        if extraction_result:
            quality = extraction_result.quality_scores.get("overall_quality", 0.0)
            validation_passed = quality >= quality_threshold
            
            # Create validation result
            validation_result = HybridResult(
                original_prompt=f"Validation (threshold: {quality_threshold})",
                processing_mode=ProcessingMode.VALIDATION_LOOP,
                domain=extraction_result.domain,
                quality_scores={"validation_score": quality, "threshold": quality_threshold},
                validation_passed=validation_passed
            )
            
            return validation_result
        
        return None
    
    def _execute_refine_step(self, 
                            step: WorkflowStep, 
                            step_input: Dict[str, Any],
                            **kwargs) -> HybridResult:
        """Execute a refinement step."""
        
        prompt = step_input.get("prompt", step_input.get("text", ""))
        domain = step.parameters.get("domain", "news")
        quality_target = step.parameters.get("quality_target", 0.8)
        
        return self.language_ai.iterative_refinement(
            prompt, 
            domain=domain, 
            quality_target=quality_target, 
            **step.parameters, 
            **kwargs
        )
    
    def create_custom_workflow(self, 
                              workflow_id: str,
                              workflow_type: WorkflowType,
                              steps_config: List[Dict[str, Any]],
                              description: str = "") -> HybridWorkflow:
        """Create a custom workflow from configuration."""
        
        steps = []
        for config in steps_config:
            step = WorkflowStep(
                step_id=config["step_id"],
                step_type=config["step_type"],
                parameters=config.get("parameters", {}),
                dependencies=config.get("dependencies", [])
            )
            steps.append(step)
        
        workflow = HybridWorkflow(workflow_id, workflow_type, steps, description)
        self.register_workflow(workflow)
        
        return workflow
    
    def batch_execute_workflow(self, 
                              workflow_id: str,
                              input_batch: List[Dict[str, Any]],
                              max_workers: int = 4) -> List[WorkflowResult]:
        """Execute workflow on a batch of inputs in parallel."""
        
        print(f"🔄 Batch executing workflow: {workflow_id}")
        print(f"   Batch size: {len(input_batch)}")
        print(f"   Max workers: {max_workers}")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all workflow executions
            future_to_input = {}
            for i, input_data in enumerate(input_batch):
                future = executor.submit(self.execute_workflow, workflow_id, input_data)
                future_to_input[future] = i
            
            # Collect results
            for future in as_completed(future_to_input):
                input_index = future_to_input[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"   ✅ Completed batch item {input_index + 1}/{len(input_batch)}")
                except Exception as e:
                    print(f"   ❌ Failed batch item {input_index + 1}: {e}")
                    # Create failed result
                    failed_result = WorkflowResult(
                        workflow_id=workflow_id,
                        workflow_type=self.workflows[workflow_id].workflow_type,
                        status="failed",
                        steps_completed=0,
                        total_steps=len(self.workflows[workflow_id].steps),
                        results={},
                        execution_time=0.0,
                        error_messages=[str(e)]
                    )
                    results.append(failed_result)
        
        print(f"📊 Batch execution completed:")
        success_count = sum(1 for r in results if r.status == "success")
        print(f"   Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
        
        return results
    
    def get_workflow_analytics(self) -> Dict[str, Any]:
        """Get analytics about workflow executions."""
        
        if not self.execution_history:
            return {"message": "No workflows executed yet"}
        
        analytics = {
            "total_executions": len(self.execution_history),
            "success_rate": 0.0,
            "average_execution_time": 0.0,
            "workflow_types": {},
            "common_errors": {}
        }
        
        success_count = 0
        total_time = 0.0
        
        for result in self.execution_history:
            if result.status == "success":
                success_count += 1
            
            total_time += result.execution_time
            
            # Count workflow types
            workflow_type = result.workflow_type.value
            if workflow_type not in analytics["workflow_types"]:
                analytics["workflow_types"][workflow_type] = 0
            analytics["workflow_types"][workflow_type] += 1
            
            # Count errors
            for error in result.error_messages:
                if error not in analytics["common_errors"]:
                    analytics["common_errors"][error] = 0
                analytics["common_errors"][error] += 1
        
        analytics["success_rate"] = success_count / len(self.execution_history)
        analytics["average_execution_time"] = total_time / len(self.execution_history)
        
        return analytics
    
    def export_workflow_results(self, output_path: str):
        """Export workflow execution history to JSON."""
        
        export_data = {
            "execution_history": [],
            "analytics": self.get_workflow_analytics(),
            "workflows": {wid: {
                "workflow_id": w.workflow_id,
                "workflow_type": w.workflow_type.value,
                "description": w.description,
                "steps": [asdict(step) for step in w.steps]
            } for wid, w in self.workflows.items()}
        }
        
        # Convert results to serializable format
        for result in self.execution_history:
            result_data = asdict(result)
            result_data["workflow_type"] = result.workflow_type.value
            
            # Convert HybridResult objects
            serialized_results = {}
            for step_id, hybrid_result in result.results.items():
                serialized_results[step_id] = {
                    "original_prompt": hybrid_result.original_prompt,
                    "processing_mode": hybrid_result.processing_mode.value,
                    "domain": hybrid_result.domain,
                    "generated_text": hybrid_result.generated_text,
                    "quality_scores": hybrid_result.quality_scores,
                    "validation_passed": hybrid_result.validation_passed,
                    "processing_time": hybrid_result.processing_time
                }
            
            result_data["results"] = serialized_results
            export_data["execution_history"].append(result_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Workflow results exported: {output_path}")


# Built-in workflow templates
def create_content_creation_workflow() -> Dict[str, Any]:
    """Template for content creation workflow."""
    return {
        "workflow_id": "content_creation_custom",
        "workflow_type": WorkflowType.CONTENT_CREATION,
        "description": "Generate high-quality content with validation",
        "steps": [
            {
                "step_id": "generate_initial",
                "step_type": "generate",
                "parameters": {"domain": "news", "max_length": 300}
            },
            {
                "step_id": "extract_check",
                "step_type": "extract",
                "parameters": {"comprehensive": True},
                "dependencies": ["generate_initial"]
            },
            {
                "step_id": "quality_validate",
                "step_type": "validate",
                "parameters": {"quality_threshold": 0.75},
                "dependencies": ["extract_check"]
            }
        ]
    }


def create_document_analysis_workflow() -> Dict[str, Any]:
    """Template for document analysis workflow."""
    return {
        "workflow_id": "document_analysis_custom",
        "workflow_type": WorkflowType.DOCUMENT_ANALYSIS,
        "description": "Comprehensive document analysis and summarization",
        "steps": [
            {
                "step_id": "extract_entities",
                "step_type": "extract",
                "parameters": {"comprehensive": True, "include_relationships": True}
            },
            {
                "step_id": "generate_summary",
                "step_type": "generate",
                "parameters": {"task": "summary", "max_length": 150},
                "dependencies": ["extract_entities"]
            },
            {
                "step_id": "validate_summary",
                "step_type": "extract",
                "parameters": {"domain": "news"},
                "dependencies": ["generate_summary"]
            }
        ]
    }