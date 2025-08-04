#!/usr/bin/env python3
"""
Instruction Processor for Amharic Smart LLM
Handles instruction following and task-specific processing in Amharic
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import re
import json
from pathlib import Path

class InstructionProcessor:
    """Processes and classifies Amharic instructions for appropriate handling"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_instruction_length': 512,
            'confidence_threshold': 0.7,
            'supported_tasks': [
                'translation', 'explanation', 'summarization', 
                'question_answering', 'text_generation', 'conversation'
            ]
        }
        
        # Amharic instruction patterns
        self.instruction_patterns = {
            'translation': {
                'keywords': ['ተርጉም', 'ተርጉምልኝ', 'ወደ', 'ቋንቋ', 'አስተርጉም'],
                'patterns': [
                    r'ወደ\s+(\w+)\s+ተርጉም',
                    r'(\w+)\s+ወደ\s+(\w+)',
                    r'ተርጉምልኝ',
                    r'አስተርጉም'
                ],
                'template': 'ተርጉም: {source} ወደ {target}'
            },
            'explanation': {
                'keywords': ['አስረዳ', 'አስረዳኝ', 'ምንድን', 'ንገረኝ', 'ማብራሪያ'],
                'patterns': [
                    r'አስረዳኝ\s+(.+)',
                    r'ምንድን\s+ነው\s+(.+)',
                    r'ንገረኝ\s+(.+)',
                    r'ማብራሪያ\s+(.+)'
                ],
                'template': 'ማብራሪያ: {topic}'
            },
            'summarization': {
                'keywords': ['አጠቃልል', 'አጠቃላይ', 'ማጠቃለያ', 'ቀንስ'],
                'patterns': [
                    r'አጠቃልል\s+(.+)',
                    r'ማጠቃለያ\s+(.+)',
                    r'ቀንስልኝ\s+(.+)'
                ],
                'template': 'ማጠቃለያ: {content}'
            },
            'question_answering': {
                'keywords': ['ምንድን', 'እንዴት', 'መቼ', 'የት', 'ለምን', 'ማን', 'ምን'],
                'patterns': [
                    r'(ምንድን|ምን)\s+(.+)\?',
                    r'እንዴት\s+(.+)\?',
                    r'መቼ\s+(.+)\?',
                    r'የት\s+(.+)\?',
                    r'ለምን\s+(.+)\?',
                    r'ማን\s+(.+)\?'
                ],
                'template': 'ጥያቄ: {question}'
            },
            'text_generation': {
                'keywords': ['ፃፍ', 'ፃፍልኝ', 'ፍጠር', 'ጻፍ', 'ይፃፉ'],
                'patterns': [
                    r'ፃፍልኝ\s+(.+)',
                    r'ፍጠርልኝ\s+(.+)',
                    r'ጻፍ\s+(.+)'
                ],
                'template': 'ጽሑፍ ፍጠራ: {topic}'
            },
            'conversation': {
                'keywords': ['ሰላም', 'እንዴት ነህ', 'ደህና ነህ', 'ንግግር'],
                'patterns': [
                    r'ሰላም',
                    r'እንዴት\s+ነህ',
                    r'ደህና\s+ነህ'
                ],
                'template': 'ንግግር: {greeting}'
            }
        }
        
        # Task-specific processors
        self.task_processors = {
            'translation': self.process_translation_instruction,
            'explanation': self.process_explanation_instruction,
            'summarization': self.process_summarization_instruction,
            'question_answering': self.process_qa_instruction,
            'text_generation': self.process_generation_instruction,
            'conversation': self.process_conversation_instruction
        }
        
    def classify_instruction(self, instruction: str) -> Dict[str, Any]:
        """Classify the type of instruction and extract parameters"""
        instruction = instruction.strip()
        
        # Initialize result
        result = {
            'task_type': 'conversation',  # Default
            'confidence': 0.0,
            'parameters': {},
            'original_instruction': instruction,
            'processed_instruction': instruction
        }
        
        best_match = None
        best_confidence = 0.0
        
        # Check each instruction type
        for task_type, patterns in self.instruction_patterns.items():
            confidence = self.calculate_pattern_confidence(instruction, patterns)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = task_type
                
        if best_confidence > self.config['confidence_threshold']:
            result['task_type'] = best_match
            result['confidence'] = best_confidence
            
            # Extract parameters using the best matching pattern
            parameters = self.extract_parameters(instruction, self.instruction_patterns[best_match])
            result['parameters'] = parameters
            
        return result
        
    def calculate_pattern_confidence(self, instruction: str, patterns: Dict) -> float:
        """Calculate confidence score for pattern matching"""
        instruction_lower = instruction.lower()
        
        # Keyword matching
        keyword_matches = 0
        for keyword in patterns['keywords']:
            if keyword in instruction_lower:
                keyword_matches += 1
                
        keyword_score = min(keyword_matches / len(patterns['keywords']), 1.0)
        
        # Pattern matching
        pattern_score = 0.0
        for pattern in patterns['patterns']:
            if re.search(pattern, instruction_lower):
                pattern_score = 1.0
                break
                
        # Combined confidence
        confidence = 0.6 * keyword_score + 0.4 * pattern_score
        
        return confidence
        
    def extract_parameters(self, instruction: str, patterns: Dict) -> Dict[str, Any]:
        """Extract parameters from instruction using patterns"""
        parameters = {}
        instruction_lower = instruction.lower()
        
        # Try to extract using regex patterns
        for pattern in patterns['patterns']:
            match = re.search(pattern, instruction_lower)
            if match:
                groups = match.groups()
                if groups:
                    # Store extracted groups
                    for i, group in enumerate(groups):
                        parameters[f'param_{i}'] = group.strip()
                break
                
        return parameters
        
    def process_instruction(self, instruction: str) -> Dict[str, Any]:
        """Main instruction processing pipeline"""
        # Classify instruction
        classification = self.classify_instruction(instruction)
        
        # Process using task-specific processor
        task_type = classification['task_type']
        if task_type in self.task_processors:
            processed_result = self.task_processors[task_type](classification)
        else:
            processed_result = self.process_generic_instruction(classification)
            
        return processed_result
        
    def process_translation_instruction(self, classification: Dict) -> Dict[str, Any]:
        """Process translation instructions"""
        params = classification['parameters']
        
        # Determine source and target languages
        source_lang = 'auto'  # Auto-detect
        target_lang = 'amharic'  # Default target
        
        # Try to extract languages from parameters
        if 'param_0' in params and 'param_1' in params:
            source_lang = params['param_0']
            target_lang = params['param_1']
        elif 'param_0' in params:
            # Single parameter might be target language
            target_lang = params['param_0']
            
        return {
            'task_type': 'translation',
            'action': 'translate_text',
            'parameters': {
                'source_language': source_lang,
                'target_language': target_lang,
                'text_to_translate': params.get('param_0', '')
            },
            'response_template': 'ተርጉሙ: {translated_text}',
            'confidence': classification['confidence']
        }
        
    def process_explanation_instruction(self, classification: Dict) -> Dict[str, Any]:
        """Process explanation instructions"""
        params = classification['parameters']
        topic = params.get('param_0', classification['original_instruction'])
        
        return {
            'task_type': 'explanation',
            'action': 'explain_topic',
            'parameters': {
                'topic': topic,
                'detail_level': 'medium',
                'language': 'amharic'
            },
            'response_template': '{topic} በተመለከተ: {explanation}',
            'confidence': classification['confidence']
        }
        
    def process_summarization_instruction(self, classification: Dict) -> Dict[str, Any]:
        """Process summarization instructions"""
        params = classification['parameters']
        content = params.get('param_0', '')
        
        return {
            'task_type': 'summarization',
            'action': 'summarize_content',
            'parameters': {
                'content': content,
                'summary_length': 'medium',
                'language': 'amharic'
            },
            'response_template': 'ማጠቃለያ: {summary}',
            'confidence': classification['confidence']
        }
        
    def process_qa_instruction(self, classification: Dict) -> Dict[str, Any]:
        """Process question-answering instructions"""
        params = classification['parameters']
        question = params.get('param_0', classification['original_instruction'])
        
        # Determine question type
        question_type = 'general'
        if 'ምንድን' in question or 'ምን' in question:
            question_type = 'what'
        elif 'እንዴት' in question:
            question_type = 'how'
        elif 'መቼ' in question:
            question_type = 'when'
        elif 'የት' in question:
            question_type = 'where'
        elif 'ለምን' in question:
            question_type = 'why'
        elif 'ማን' in question:
            question_type = 'who'
            
        return {
            'task_type': 'question_answering',
            'action': 'answer_question',
            'parameters': {
                'question': question,
                'question_type': question_type,
                'language': 'amharic'
            },
            'response_template': 'መልስ: {answer}',
            'confidence': classification['confidence']
        }
        
    def process_generation_instruction(self, classification: Dict) -> Dict[str, Any]:
        """Process text generation instructions"""
        params = classification['parameters']
        topic = params.get('param_0', 'general topic')
        
        return {
            'task_type': 'text_generation',
            'action': 'generate_text',
            'parameters': {
                'topic': topic,
                'length': 'medium',
                'style': 'informative',
                'language': 'amharic'
            },
            'response_template': '{generated_text}',
            'confidence': classification['confidence']
        }
        
    def process_conversation_instruction(self, classification: Dict) -> Dict[str, Any]:
        """Process conversational instructions"""
        return {
            'task_type': 'conversation',
            'action': 'continue_conversation',
            'parameters': {
                'input': classification['original_instruction'],
                'conversation_type': 'casual',
                'language': 'amharic'
            },
            'response_template': '{response}',
            'confidence': classification['confidence']
        }
        
    def process_generic_instruction(self, classification: Dict) -> Dict[str, Any]:
        """Process generic/unknown instructions"""
        return {
            'task_type': 'generic',
            'action': 'general_response',
            'parameters': {
                'input': classification['original_instruction'],
                'language': 'amharic'
            },
            'response_template': '{response}',
            'confidence': classification['confidence']
        }
        
    def format_response(self, processed_instruction: Dict, generated_content: str) -> str:
        """Format the final response using the template"""
        template = processed_instruction.get('response_template', '{response}')
        
        # Simple template formatting
        if '{translated_text}' in template:
            return template.format(translated_text=generated_content)
        elif '{explanation}' in template:
            topic = processed_instruction['parameters'].get('topic', '')
            return template.format(topic=topic, explanation=generated_content)
        elif '{summary}' in template:
            return template.format(summary=generated_content)
        elif '{answer}' in template:
            return template.format(answer=generated_content)
        elif '{generated_text}' in template:
            return template.format(generated_text=generated_content)
        else:
            return generated_content
            
    def get_instruction_context(self, processed_instruction: Dict) -> Dict[str, Any]:
        """Get context information for instruction processing"""
        return {
            'task_type': processed_instruction['task_type'],
            'confidence': processed_instruction['confidence'],
            'parameters': processed_instruction['parameters'],
            'requires_external_knowledge': processed_instruction['task_type'] in [
                'translation', 'question_answering', 'explanation'
            ],
            'is_creative_task': processed_instruction['task_type'] in [
                'text_generation', 'conversation'
            ]
        }

class ChainOfThoughtReasoning:
    """Implements chain-of-thought reasoning for complex Amharic instructions"""
    
    def __init__(self, instruction_processor: InstructionProcessor):
        self.instruction_processor = instruction_processor
        self.reasoning_steps = []
        
    def break_down_complex_instruction(self, instruction: str) -> List[Dict[str, Any]]:
        """Break down complex instructions into simpler steps"""
        # Simple heuristic: split by conjunctions and punctuation
        separators = ['እና', 'ከዚያ', 'ቀጥሎ', '።', '፣']
        
        steps = [instruction]
        for separator in separators:
            new_steps = []
            for step in steps:
                if separator in step:
                    parts = step.split(separator)
                    new_steps.extend([part.strip() for part in parts if part.strip()])
                else:
                    new_steps.append(step)
            steps = new_steps
            
        # Process each step
        processed_steps = []
        for i, step in enumerate(steps):
            if step:
                processed_step = self.instruction_processor.process_instruction(step)
                processed_step['step_number'] = i + 1
                processed_step['original_step'] = step
                processed_steps.append(processed_step)
                
        return processed_steps
        
    def execute_reasoning_chain(self, instruction: str) -> Dict[str, Any]:
        """Execute chain-of-thought reasoning"""
        # Break down instruction
        steps = self.break_down_complex_instruction(instruction)
        
        # Execute each step
        reasoning_chain = {
            'original_instruction': instruction,
            'steps': steps,
            'reasoning_path': [],
            'final_result': None
        }
        
        for step in steps:
            reasoning_step = {
                'step_number': step['step_number'],
                'instruction': step['original_step'],
                'task_type': step['task_type'],
                'confidence': step['confidence'],
                'reasoning': f"ደረጃ {step['step_number']}: {step['task_type']} ተግባር"
            }
            reasoning_chain['reasoning_path'].append(reasoning_step)
            
        # Determine final result type
        if steps:
            final_step = steps[-1]
            reasoning_chain['final_result'] = {
                'task_type': final_step['task_type'],
                'action': final_step['action'],
                'parameters': final_step['parameters']
            }
            
        return reasoning_chain
        
    def explain_reasoning(self, reasoning_chain: Dict[str, Any]) -> str:
        """Generate explanation of the reasoning process in Amharic"""
        explanation_parts = [
            f"የእኔ አስተሳሰብ ሂደት ለ '{reasoning_chain['original_instruction']}':"
        ]
        
        for step in reasoning_chain['reasoning_path']:
            explanation_parts.append(
                f"ደረጃ {step['step_number']}: {step['reasoning']}"
            )
            
        if reasoning_chain['final_result']:
            final_task = reasoning_chain['final_result']['task_type']
            explanation_parts.append(f"የመጨረሻ ተግባር: {final_task}")
            
        return "\n".join(explanation_parts)

# Example usage and testing
def test_instruction_processor():
    """Test the instruction processor with sample Amharic instructions"""
    processor = InstructionProcessor()
    
    test_instructions = [
        "ወደ እንግሊዝኛ ተርጉምልኝ",
        "ስለ ኢትዮጵያ አስረዳኝ",
        "ይህንን ጽሑፍ አጠቃልልልኝ",
        "ምንድን ነው የአማርኛ ቋንቋ?",
        "ስለ ቡና ፃፍልኝ",
        "ሰላም እንዴት ነህ?"
    ]
    
    print("🧠 Testing Amharic Instruction Processor")
    print("=" * 50)
    
    for instruction in test_instructions:
        result = processor.process_instruction(instruction)
        print(f"\nInstruction: {instruction}")
        print(f"Task Type: {result['task_type']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Action: {result['action']}")
        print(f"Parameters: {result['parameters']}")
        
if __name__ == "__main__":
    test_instruction_processor()