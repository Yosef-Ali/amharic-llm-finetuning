#!/usr/bin/env python3
"""
Enhanced Memory System for Amharic Conversational LLM
Implements sophisticated memory management for multi-turn conversations
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Any
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle

class MemoryEntry:
    """Individual memory entry with metadata"""
    
    def __init__(self, content: str, entry_type: str, importance: float = 1.0, context: Dict = None):
        self.content = content
        self.entry_type = entry_type  # 'user_input', 'assistant_response', 'context', 'fact'
        self.importance = importance
        self.context = context or {}
        self.timestamp = datetime.now()
        self.access_count = 0
        self.last_accessed = datetime.now()
        self.embedding = None  # Will store vector representation
        
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        
    def calculate_relevance_score(self, current_time: datetime = None) -> float:
        """Calculate relevance based on recency, frequency, and importance"""
        if current_time is None:
            current_time = datetime.now()
            
        # Time decay factor (more recent = higher score)
        time_diff = (current_time - self.timestamp).total_seconds() / 3600  # hours
        recency_score = np.exp(-time_diff / 24)  # Decay over 24 hours
        
        # Frequency factor
        frequency_score = min(self.access_count / 10, 1.0)  # Cap at 1.0
        
        # Combined score
        relevance = (0.4 * recency_score + 0.3 * frequency_score + 0.3 * self.importance)
        
        return relevance
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'content': self.content,
            'entry_type': self.entry_type,
            'importance': self.importance,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryEntry':
        """Create from dictionary"""
        entry = cls(
            content=data['content'],
            entry_type=data['entry_type'],
            importance=data['importance'],
            context=data['context']
        )
        entry.timestamp = datetime.fromisoformat(data['timestamp'])
        entry.access_count = data['access_count']
        entry.last_accessed = datetime.fromisoformat(data['last_accessed'])
        return entry

class EnhancedMemory:
    """Enhanced memory system with multiple memory types"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'short_term_capacity': 20,
            'long_term_capacity': 1000,
            'working_memory_capacity': 5,
            'embedding_dim': 256,
            'similarity_threshold': 0.7,
            'importance_threshold': 0.5
        }
        
        # Different memory stores
        self.working_memory = []  # Current conversation context
        self.short_term_memory = []  # Recent conversations
        self.long_term_memory = []  # Important/frequent information
        self.episodic_memory = []  # Specific conversation episodes
        self.semantic_memory = {}  # Facts and knowledge
        
        # Memory indexing for fast retrieval
        self.memory_index = {}
        
        # Simple embedding layer for memory encoding
        self.memory_encoder = nn.Linear(768, self.config['embedding_dim'])
        
    def add_to_working_memory(self, content: str, entry_type: str, context: Dict = None):
        """Add entry to working memory (current conversation)"""
        entry = MemoryEntry(content, entry_type, context=context)
        self.working_memory.append(entry)
        
        # Maintain capacity
        if len(self.working_memory) > self.config['working_memory_capacity']:
            # Move oldest to short-term memory
            oldest = self.working_memory.pop(0)
            self.add_to_short_term_memory(oldest)
            
    def add_to_short_term_memory(self, entry: MemoryEntry):
        """Add entry to short-term memory"""
        self.short_term_memory.append(entry)
        
        # Maintain capacity and promote important memories
        if len(self.short_term_memory) > self.config['short_term_capacity']:
            # Sort by relevance and promote/discard
            self.short_term_memory.sort(key=lambda x: x.calculate_relevance_score(), reverse=True)
            
            # Promote highly relevant memories to long-term
            for memory in self.short_term_memory[:5]:
                if memory.calculate_relevance_score() > self.config['importance_threshold']:
                    self.promote_to_long_term(memory)
                    
            # Keep only the most relevant
            self.short_term_memory = self.short_term_memory[:self.config['short_term_capacity']]
            
    def promote_to_long_term(self, entry: MemoryEntry):
        """Promote entry to long-term memory"""
        # Check if similar memory already exists
        similar_memories = self.find_similar_memories(entry.content, self.long_term_memory)
        
        if similar_memories:
            # Merge with existing memory
            similar_memories[0].importance = max(similar_memories[0].importance, entry.importance)
            similar_memories[0].access_count += entry.access_count
        else:
            # Add as new long-term memory
            entry.importance = max(entry.importance, 0.8)  # Boost importance for long-term
            self.long_term_memory.append(entry)
            
        # Maintain capacity
        if len(self.long_term_memory) > self.config['long_term_capacity']:
            self.long_term_memory.sort(key=lambda x: x.calculate_relevance_score(), reverse=True)
            self.long_term_memory = self.long_term_memory[:self.config['long_term_capacity']]
            
    def find_similar_memories(self, query: str, memory_store: List[MemoryEntry], top_k: int = 5) -> List[MemoryEntry]:
        """Find similar memories using simple text similarity"""
        # Simple similarity based on common words (can be enhanced with embeddings)
        query_words = set(query.lower().split())
        
        similarities = []
        for memory in memory_store:
            memory_words = set(memory.content.lower().split())
            
            if len(query_words) == 0 or len(memory_words) == 0:
                similarity = 0.0
            else:
                intersection = len(query_words.intersection(memory_words))
                union = len(query_words.union(memory_words))
                similarity = intersection / union if union > 0 else 0.0
                
            if similarity > self.config['similarity_threshold']:
                similarities.append((memory, similarity))
                
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, sim in similarities[:top_k]]
        
    def retrieve_relevant_context(self, query: str, max_entries: int = 10) -> List[MemoryEntry]:
        """Retrieve relevant memories for current query"""
        relevant_memories = []
        
        # Always include working memory
        relevant_memories.extend(self.working_memory)
        
        # Search short-term memory
        short_term_relevant = self.find_similar_memories(query, self.short_term_memory, top_k=3)
        relevant_memories.extend(short_term_relevant)
        
        # Search long-term memory
        long_term_relevant = self.find_similar_memories(query, self.long_term_memory, top_k=5)
        relevant_memories.extend(long_term_relevant)
        
        # Remove duplicates and sort by relevance
        unique_memories = list({id(mem): mem for mem in relevant_memories}.values())
        unique_memories.sort(key=lambda x: x.calculate_relevance_score(), reverse=True)
        
        # Update access statistics
        for memory in unique_memories[:max_entries]:
            memory.update_access()
            
        return unique_memories[:max_entries]
        
    def add_semantic_fact(self, key: str, value: str, importance: float = 1.0):
        """Add factual information to semantic memory"""
        self.semantic_memory[key] = {
            'value': value,
            'importance': importance,
            'timestamp': datetime.now(),
            'access_count': 0
        }
        
    def get_semantic_fact(self, key: str) -> Optional[str]:
        """Retrieve factual information"""
        if key in self.semantic_memory:
            self.semantic_memory[key]['access_count'] += 1
            return self.semantic_memory[key]['value']
        return None
        
    def consolidate_memories(self):
        """Consolidate and organize memories (run periodically)"""
        # Merge similar memories in short-term
        consolidated_short_term = []
        processed_indices = set()
        
        for i, memory in enumerate(self.short_term_memory):
            if i in processed_indices:
                continue
                
            similar_memories = []
            for j, other_memory in enumerate(self.short_term_memory[i+1:], i+1):
                if j in processed_indices:
                    continue
                    
                # Check similarity
                memory_words = set(memory.content.lower().split())
                other_words = set(other_memory.content.lower().split())
                
                if len(memory_words) > 0 and len(other_words) > 0:
                    intersection = len(memory_words.intersection(other_words))
                    union = len(memory_words.union(other_words))
                    similarity = intersection / union
                    
                    if similarity > 0.8:  # High similarity threshold for consolidation
                        similar_memories.append(other_memory)
                        processed_indices.add(j)
                        
            if similar_memories:
                # Merge memories
                merged_content = memory.content
                merged_importance = memory.importance
                merged_access_count = memory.access_count
                
                for sim_mem in similar_memories:
                    merged_importance = max(merged_importance, sim_mem.importance)
                    merged_access_count += sim_mem.access_count
                    
                memory.importance = merged_importance
                memory.access_count = merged_access_count
                
            consolidated_short_term.append(memory)
            processed_indices.add(i)
            
        self.short_term_memory = consolidated_short_term
        
    def get_conversation_summary(self) -> str:
        """Generate summary of current conversation"""
        if not self.working_memory:
            return "የንግግር ታሪክ የለም።"  # "No conversation history."
            
        summary_parts = []
        for memory in self.working_memory[-5:]:  # Last 5 entries
            if memory.entry_type == 'user_input':
                summary_parts.append(f"ተጠቃሚ: {memory.content}")
            elif memory.entry_type == 'assistant_response':
                summary_parts.append(f"ረዳት: {memory.content}")
                
        return "\n".join(summary_parts)
        
    def clear_working_memory(self):
        """Clear working memory (end conversation)"""
        # Move all working memory to episodic memory as a conversation episode
        if self.working_memory:
            episode = {
                'memories': self.working_memory.copy(),
                'timestamp': datetime.now(),
                'summary': self.get_conversation_summary()
            }
            self.episodic_memory.append(episode)
            
        self.working_memory = []
        
    def save_memory_state(self, filepath: str):
        """Save memory state to file"""
        memory_state = {
            'working_memory': [mem.to_dict() for mem in self.working_memory],
            'short_term_memory': [mem.to_dict() for mem in self.short_term_memory],
            'long_term_memory': [mem.to_dict() for mem in self.long_term_memory],
            'semantic_memory': self.semantic_memory,
            'episodic_memory': [
                {
                    'memories': [mem.to_dict() for mem in episode['memories']],
                    'timestamp': episode['timestamp'].isoformat(),
                    'summary': episode['summary']
                }
                for episode in self.episodic_memory
            ],
            'config': self.config
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_state, f, ensure_ascii=False, indent=2)
            
    def load_memory_state(self, filepath: str):
        """Load memory state from file"""
        if not Path(filepath).exists():
            return
            
        with open(filepath, 'r', encoding='utf-8') as f:
            memory_state = json.load(f)
            
        # Restore memories
        self.working_memory = [MemoryEntry.from_dict(mem) for mem in memory_state.get('working_memory', [])]
        self.short_term_memory = [MemoryEntry.from_dict(mem) for mem in memory_state.get('short_term_memory', [])]
        self.long_term_memory = [MemoryEntry.from_dict(mem) for mem in memory_state.get('long_term_memory', [])]
        self.semantic_memory = memory_state.get('semantic_memory', {})
        
        # Restore episodic memory
        self.episodic_memory = []
        for episode_data in memory_state.get('episodic_memory', []):
            episode = {
                'memories': [MemoryEntry.from_dict(mem) for mem in episode_data['memories']],
                'timestamp': datetime.fromisoformat(episode_data['timestamp']),
                'summary': episode_data['summary']
            }
            self.episodic_memory.append(episode)
            
        # Update config
        self.config.update(memory_state.get('config', {}))
        
    def get_memory_statistics(self) -> Dict:
        """Get memory usage statistics"""
        return {
            'working_memory_size': len(self.working_memory),
            'short_term_memory_size': len(self.short_term_memory),
            'long_term_memory_size': len(self.long_term_memory),
            'semantic_facts_count': len(self.semantic_memory),
            'episodic_memories_count': len(self.episodic_memory),
            'total_memory_entries': (
                len(self.working_memory) + 
                len(self.short_term_memory) + 
                len(self.long_term_memory)
            )
        }

class AmharicMemoryManager:
    """High-level memory management for Amharic conversations"""
    
    def __init__(self, memory_system: EnhancedMemory):
        self.memory = memory_system
        
        # Amharic-specific memory patterns
        self.amharic_patterns = {
            'greetings': ['ሰላም', 'ሰላም ነህ', 'እንዴት ነህ', 'ደህና ነህ'],
            'questions': ['ምንድን', 'እንዴት', 'መቼ', 'የት', 'ለምን', 'ማን'],
            'polite_expressions': ['እባክህ', 'አመሰግናለሁ', 'ይቅርታ', 'በጣም ጥሩ'],
            'cultural_terms': ['ኢትዮጵያ', 'አማራ', 'ኦሮሞ', 'ትግራይ', 'ቡና', 'ኢንጀራ']
        }
        
        # Initialize with basic Amharic facts
        self.initialize_amharic_knowledge()
        
    def initialize_amharic_knowledge(self):
        """Initialize with basic Amharic cultural and linguistic knowledge"""
        basic_facts = {
            'ኢትዮጵያ_ዋና_ከተማ': 'አዲስ አበባ',
            'አማርኛ_ተናጋሪዎች': 'በ25 ሚሊዮን የሚጠጉ ሰዎች',
            'ኢትዮጵያ_ዋና_ምግብ': 'ኢንጀራ',
            'ኢትዮጵያ_ዋና_መጠጥ': 'ቡና',
            'አማርኛ_ፊደል': 'ግዕዝ ፊደል'
        }
        
        for key, value in basic_facts.items():
            self.memory.add_semantic_fact(key, value, importance=0.9)
            
    def process_amharic_input(self, user_input: str, assistant_response: str):
        """Process Amharic conversation input with cultural awareness"""
        # Determine importance based on Amharic patterns
        importance = self.calculate_cultural_importance(user_input)
        
        # Add to memory with appropriate context
        context = {
            'language': 'amharic',
            'cultural_relevance': self.assess_cultural_relevance(user_input),
            'conversation_type': self.classify_conversation_type(user_input)
        }
        
        self.memory.add_to_working_memory(user_input, 'user_input', context)
        self.memory.add_to_working_memory(assistant_response, 'assistant_response', context)
        
    def calculate_cultural_importance(self, text: str) -> float:
        """Calculate importance based on Amharic cultural content"""
        importance = 0.5  # Base importance
        
        text_lower = text.lower()
        
        # Boost importance for cultural terms
        for term in self.amharic_patterns['cultural_terms']:
            if term in text_lower:
                importance += 0.2
                
        # Boost for questions (learning opportunities)
        for question_word in self.amharic_patterns['questions']:
            if question_word in text_lower:
                importance += 0.1
                
        return min(importance, 1.0)
        
    def assess_cultural_relevance(self, text: str) -> float:
        """Assess cultural relevance of the text"""
        relevance = 0.0
        text_lower = text.lower()
        
        for category, terms in self.amharic_patterns.items():
            for term in terms:
                if term in text_lower:
                    relevance += 0.25
                    
        return min(relevance, 1.0)
        
    def classify_conversation_type(self, text: str) -> str:
        """Classify the type of conversation"""
        text_lower = text.lower()
        
        if any(greeting in text_lower for greeting in self.amharic_patterns['greetings']):
            return 'greeting'
        elif any(question in text_lower for question in self.amharic_patterns['questions']):
            return 'question'
        elif any(polite in text_lower for polite in self.amharic_patterns['polite_expressions']):
            return 'polite_conversation'
        else:
            return 'general_conversation'
            
    def get_culturally_relevant_context(self, query: str) -> List[str]:
        """Get culturally relevant context for response generation"""
        relevant_memories = self.memory.retrieve_relevant_context(query)
        
        # Format for cultural context
        context_strings = []
        for memory in relevant_memories:
            if memory.context and memory.context.get('cultural_relevance', 0) > 0.5:
                context_strings.append(memory.content)
                
        return context_strings