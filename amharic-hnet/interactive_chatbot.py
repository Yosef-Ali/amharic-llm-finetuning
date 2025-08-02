#!/usr/bin/env python3
"""
Interactive Amharic Chatbot for H-Net Model
Provides a natural conversation interface with the optimized H-Net model
"""

import json
import os
import sys
import requests
import re
from datetime import datetime
from typing import List, Dict, Any

# Import our optimized generators
try:
    from final_natural_amharic_generator import FinalNaturalAmharicGenerator
    from fluent_amharic_generator import FluentAmharicGenerator
except ImportError:
    print("Warning: Generator modules not found. Using fallback mode.")
    FinalNaturalAmharicGenerator = None
    FluentAmharicGenerator = None

class AmharicChatbot:
    """
    Interactive chatbot for natural Amharic conversations
    """
    
    def __init__(self):
        self.conversation_history = []
        self.session_start = datetime.now()
        
        # Initialize generators if available
        self.natural_generator = FinalNaturalAmharicGenerator() if FinalNaturalAmharicGenerator else None
        self.fluent_generator = FluentAmharicGenerator() if FluentAmharicGenerator else None
        
        # Optimized H-Net parameters
        self.model_params = {
            'top_p': 0.92,
            'temperature': 0.8,
            'repetition_penalty': 1.2,
            'max_length': 150,
            'do_sample': True
        }
        
        # Enhanced conversation patterns with learning topics
        self.learning_patterns = {
            'amharic_learning': {
                'አማርኛ ልማር': 'በጣም ጥሩ! አማርኛ ማስተማር እወዳለሁ። ከየት መጀመር ትፈልጋለህ? ፊደላት፣ ቃላት ወይስ ዓረፍተ ነገሮች?',
                'learn amharic': 'በጣም ጥሩ! አማርኛ ማስተማር እወዳለሁ። ከየት መጀመር ትፈልጋለህ? ፊደላት፣ ቃላት ወይስ ዓረፍተ ነገሮች?',
                'teach me amharic': 'እሺ! አማርኛ ማስተማር እወዳለሁ። ምን ማወቅ ትፈልጋለህ? ቀላል ቃላት፣ ሰላምታ ወይስ ዕለታዊ ንግግር?',
                'amharic alphabet': 'የአማርኛ ፊደላት 33 ናቸው። እንጀምር: አ ኡ ኢ ኣ ኤ እ ኦ። እነዚህ የመጀመሪያዎቹ ናቸው። ልተገብር?',
                'amharic words': 'ቀላል አማርኛ ቃላት እንማር: ሰላም (hello), አመሰግናለሁ (thank you), እሺ (okay), ውሃ (water), እንዴት ነህ (how are you)',
                'how to say': 'ምን ማለት ትፈልጋለህ? በአማርኛ እንዴት እንደሚባል ንገረኝ።'
            },
            'knowledge_requests': {
                'ስለ ኢትዮጵያ': 'ኢትዮጵያ ታሪክዊ ነፃ አገር ነች። ምን ማወቅ ትፈልጋለህ? ታሪክ፣ ባህል፣ ቋንቋዎች ወይስ ከተሞች?',
                'about ethiopia': 'ኢትዮጵያ ታሪክዊ ነፃ አገር ነች። ምን ማወቅ ትፈልጋለህ? ታሪክ፣ ባህል፣ ቋንቋዎች ወይስ ከተሞች?',
                'ethiopian culture': 'የኢትዮጵያ ባህል በጣም ሀብታም ነው። ቡና ሥነ ሥርዓት፣ ኢንጀራ፣ ወተት ሽሮ፣ የተለያዩ ዳንሶች አሉን። ስለ ምን ማወቅ ትፈልጋለህ?',
                'search': 'ምን መፈለግ ትፈልጋለህ? በኢንተርኔት ላፈልግልህ እችላለሁ።',
                'find information': 'ስለ ምን መረጃ ትፈልጋለህ? ላፈልግልህ እችላለሁ።'
            }
        }
        
        # Original conversation patterns
        self.conversation_patterns = {
            'greetings': {
                'ሰላም': 'ሰላም! እንዴት ነህ? ምን አዲስ ነገር አለ?',
                'ሰላም ነህ': 'ሰላም ነኝ! አንተስ እንዴት ነህ? ደህና ነህ?',
                'እንዴት ነህ': 'ጥሩ ነኝ እግዚአብሔር ይመስገን! አንተስ እንዴት ነህ?',
                'ደህና ነህ': 'ደህና ነኝ! እግዚአብሔር ይመስገን። አንተስ ደህና ነህ?',
                'ጤና ይስጥልኝ': 'ጤና ይስጥልኝ! እንዴት ነህ? ምን ትሰራለህ?'
            },
            'daily_talk': {
                'ምን ትሰራለህ': 'የተለያዩ ስራዎችን እሰራለሁ። አንተስ ምን ትሰራለህ?',
                'ስራ': 'ስራ በጣም አስፈላጊ ነው። ምን ዓይነት ስራ ትሰራለህ?',
                'ትምህርት': 'ትምህርት ወርቅ ነው! ምን ትማራለህ?',
                'ቤተሰብ': 'ቤተሰብ በጣም አስፈላጊ ነው። ቤተሰብህ እንዴት ነው?',
                'ጤና': 'ጤና ወርቅ ነው! ጤንነትህ እንዴት ነው?'
            },
            'questions': {
                'ምን ታውቃለህ': 'ብዙ ነገሮችን አውቃለሁ። ስለ ምን መጠየቅ ትፈልጋለህ?',
                'እንዴት ነው': 'ጥሩ ነው! ስለ ምን እየተነጋገርን ነው?',
                'ለምን': 'ጥሩ ጥያቄ ነው! ስለ ምን ነው የምትጠይቀው?',
                'የት': 'የት ማለትህ ነው? ስለ ቦታ ነው የምትጠይቀው?',
                'መቼ': 'መቼ ማለትህ ነው? ስለ ጊዜ ነው የምትጠይቀው?'
            },
            'expressions': {
                'እሺ': 'እሺ! ምን እንድሰራ ትፈልጋለህ?',
                'አመሰግናለሁ': 'አይዞህ! ምንም አይደለም።',
                'ይቅርታ': 'ምንም ችግር የለም! ሁሉም ነገር ደህና ነው።',
                'በጣም ጥሩ': 'እኔም እንዲሁ ይመስለኛል! በጣም ደስ ይላል።',
                'እግዚአብሔር ይመስገን': 'አሜን! እግዚአብሔር ይመስገን።'
            }
        }
        
        # Context awareness
        self.current_topic = None
        self.conversation_mood = 'neutral'
        
    def search_web(self, query: str) -> str:
        """
        Search the web for information and return Amharic response
        """
        try:
            # Simple web search simulation (you can integrate with actual search APIs)
            search_terms = query.lower()
            
            # Ethiopia-related searches
            if any(term in search_terms for term in ['ethiopia', 'ኢትዮጵያ', 'ethiopian']):
                return "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ታሪክዊ ነፃ አገር ናት። ዋና ከተማዋ አዲስ አበባ ናት። 80 ሚሊዮን በላይ ሕዝብ አላት።"
            
            # Amharic language searches
            elif any(term in search_terms for term in ['amharic', 'አማርኛ', 'language']):
                return "አማርኛ የኢትዮጵያ ሥራ ቋንቋ ናት። በ30 ሚሊዮን በላይ ሰዎች ይናገሩታል። ገዕዝ ፊደል ትጠቀማለች።"
            
            # Culture searches
            elif any(term in search_terms for term in ['culture', 'ባህል', 'coffee', 'ቡና']):
                return "የኢትዮጵያ ባህል በጣም ሀብታም ነው። ቡና የተወለደችበት አገር ናት። ኢንጀራ፣ ወተት፣ ዶሮ ወጥ ዋና ምግቦች ናቸው።"
            
            # Learning requests
            elif any(term in search_terms for term in ['learn', 'ማር', 'teach', 'አስተምር']):
                return "ማስተማር እወዳለሁ! ስለ ምን ማወቅ ትፈልጋለህ? አማርኛ፣ ኢትዮጵያ፣ ባህል ወይስ ሌላ ነገር?"
            
            # General knowledge
            else:
                return f"ስለ '{query}' መረጃ እየፈለግሁ ነው። ተጨማሪ ዝርዝር ንገረኝ እንዲህ ይሻላል።"
                
        except Exception as e:
            return "ይቅርታ፣ አሁን መረጃ ማግኘት አልቻልሁም። ሌላ ጥያቄ ጠይቅ።"
    
    def detect_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Enhanced intent detection with learning and knowledge support
        """
        user_input = user_input.strip().lower()
        
        # Check for learning patterns first
        for category, patterns in self.learning_patterns.items():
            for pattern, response in patterns.items():
                if pattern.lower() in user_input:
                    return {
                        'type': 'learning',
                        'category': category,
                        'pattern': pattern,
                        'suggested_response': response,
                        'confidence': 0.95
                    }
        
        # Check for search requests
        search_indicators = ['search', 'find', 'ፈልግ', 'ማግኘት', 'መረጃ', 'information', 'about', 'ስለ']
        if any(indicator in user_input for indicator in search_indicators):
            return {
                'type': 'search',
                'pattern': 'search_request',
                'suggested_response': None,
                'confidence': 0.9,
                'requires_search': True
            }
        
        # Check for greetings
        for greeting, response in self.conversation_patterns['greetings'].items():
            if greeting.lower() in user_input:
                return {
                    'type': 'greeting',
                    'pattern': greeting,
                    'suggested_response': response,
                    'confidence': 0.9
                }
        
        # Check for daily talk
        for topic, response in self.conversation_patterns['daily_talk'].items():
            if topic.lower() in user_input:
                return {
                    'type': 'daily_talk',
                    'pattern': topic,
                    'suggested_response': response,
                    'confidence': 0.8
                }
        
        # Check for questions
        for question, response in self.conversation_patterns['questions'].items():
            if question.lower() in user_input:
                return {
                    'type': 'question',
                    'pattern': question,
                    'suggested_response': response,
                    'confidence': 0.7
                }
        
        # Check for expressions
        for expr, response in self.conversation_patterns['expressions'].items():
            if expr.lower() in user_input:
                return {
                    'type': 'expression',
                    'pattern': expr,
                    'suggested_response': response,
                    'confidence': 0.8
                }
        
        return {
            'type': 'general',
            'pattern': None,
            'suggested_response': None,
            'confidence': 0.5,
            'requires_search': True
        }
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate intelligent response with enhanced learning and search capabilities
        """
        intent = self.detect_intent(user_input)
        
        # Handle learning requests
        if intent['type'] == 'learning':
            response = intent['suggested_response']
            # Add follow-up questions for better engagement
            if intent['category'] == 'amharic_learning':
                response += " ምን ማወቅ ትፈልጋለህ? ቃላት፣ ሰዋሰው ወይስ ንግግር?"
            elif intent['category'] == 'knowledge_requests':
                response += " ስለ ምን ዝርዝር መረጃ ትፈልጋለህ?"
        
        # Handle search requests
        elif intent.get('requires_search', False) or intent['type'] == 'search':
            response = self.search_web(user_input)
        
        # Use pattern-based response if available and confident
        elif intent['suggested_response'] and intent['confidence'] > 0.7:
            response = intent['suggested_response']
            
            # Add contextual follow-ups for better conversation flow
            if intent['type'] == 'greeting':
                response += " ዛሬ እንዴት ነህ? ምን አዲስ ነገር አለ?"
            elif intent['type'] == 'daily_talk':
                response += " ስለዚህ ተጨማሪ ንገረኝ።"
        else:
            # Generate using H-Net model for more natural responses
            if intent['type'] == 'greeting':
                prompt = f"ሰላምታ እና ወዳጃዊ ንግግር: {user_input}"
            elif intent['type'] == 'daily_talk':
                prompt = f"የዕለት ተዕለት ንግግር እና ልምድ: {user_input}"
            elif intent['type'] == 'question':
                prompt = f"ጥያቄ እና መረጃ ጥያቄ: {user_input}"
            elif intent['type'] == 'expression':
                prompt = f"ስሜት እና ግንዛቤ: {user_input}"
            else:
                prompt = f"ተፈጥሯዊ ንግግር እና ምላሽ: {user_input}"
            
            # Try fluent generator first
            try:
                response = self.fluent_generator.generate_conversation(prompt)
                if not response or len(response.strip()) < 10:
                    response = self.natural_generator.generate_natural_conversation(prompt)
                
                # If model response is still poor, provide helpful fallback
                if not response or len(response.strip()) < 5:
                    response = "ይህን ጥያቄ በደንብ ገባኝ። ተጨማሪ ዝርዝር ንገረኝ እንዲህ ይሻላል።"
                    
            except Exception as e:
                print(f"Model generation error: {e}")
                # Fallback responses
                fallback_responses = [
                    'በጣም ጥሩ ነው! ተጨማሪ ንገረኝ።',
                    'እንዲህ ነው! ምን ይመስልሃል?',
                    'ጥሩ ነገር ነው። ስለዚህ ምን ታስባለህ?',
                    'እሺ! ተጨማሪ መረጃ ስጠኝ።',
                    'በጣም አስደሳች ነው! ቀጥል።'
                ]
                import random
                response = random.choice(fallback_responses)
        
        # Add to conversation history
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'bot_response': response,
            'intent': intent
        })
        
        return response
    
    def start_conversation(self):
        """
        Start interactive conversation
        """
        print("\n" + "="*60)
        print("🇪🇹 የአማርኛ ውይይት ቻትቦት - Amharic Conversation Chatbot")
        print("="*60)
        print("ሰላም! እኔ የአማርኛ ውይይት ቻትቦት ነኝ።")
        print("ከእኔ ጋር በተፈጥሮ አማርኛ ውይይት ማድረግ ትችላለህ።")
        print("\nየሚከተሉትን ትችላለህ:")
        print("• ሰላምታ መስጠት (ሰላም፣ እንዴት ነህ፣ ጤና ይስጥልኝ)")
        print("• ስለ ዕለታዊ ሕይወት መነጋገር (ስራ፣ ትምህርት፣ ቤተሰብ)")
        print("• ጥያቄዎችን መጠየቅ")
        print("• ስሜቶችን መግለጽ")
        print("\nለመውጣት 'quit' ወይም 'exit' ይተይቡ")
        print("-"*60)
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 አንተ: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'ውጣ', 'ዝጋ']:
                    self.end_conversation()
                    break
                
                if not user_input:
                    print("🤖 ቻትቦት: እባክህ ምንም ነገር ይተይቡ!")
                    continue
                
                # Generate and display response
                response = self.generate_response(user_input)
                print(f"🤖 ቻትቦት: {response}")
                
            except KeyboardInterrupt:
                print("\n\nውይይቱ ተቋርጧል።")
                self.end_conversation()
                break
            except Exception as e:
                print(f"\n❌ ስህተት ተከስቷል: {e}")
                print("እባክህ እንደገና ይሞክሩ።")
    
    def end_conversation(self):
        """
        End conversation and save history
        """
        print("\n" + "-"*60)
        print("🙏 አመሰግናለሁ! ውይይቱ በጣም ጥሩ ነበር።")
        print("ደህና ሁን! በሌላ ጊዜ እንገናኝ።")
        
        # Save conversation history
        if self.conversation_history:
            session_data = {
                'session_start': self.session_start.isoformat(),
                'session_end': datetime.now().isoformat(),
                'total_exchanges': len(self.conversation_history),
                'conversation_history': self.conversation_history,
                'model_params': self.model_params
            }
            
            filename = f"conversation_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, ensure_ascii=False, indent=2)
                print(f"\n💾 ውይይቱ በ {filename} ተቀምጧል።")
            except Exception as e:
                print(f"\n⚠️  ውይይቱን ማስቀመጥ አልተቻለም: {e}")
        
        print("="*60)
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics
        """
        if not self.conversation_history:
            return {'total_exchanges': 0}
        
        intent_counts = {}
        for exchange in self.conversation_history:
            intent_type = exchange['intent']['type']
            intent_counts[intent_type] = intent_counts.get(intent_type, 0) + 1
        
        return {
            'total_exchanges': len(self.conversation_history),
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'intent_distribution': intent_counts,
            'average_confidence': sum(ex['intent']['confidence'] for ex in self.conversation_history) / len(self.conversation_history)
        }

def main():
    """
    Main function to run the chatbot
    """
    try:
        chatbot = AmharicChatbot()
        chatbot.start_conversation()
    except Exception as e:
        print(f"\n❌ ቻትቦቱን ማስጀመር አልተቻለም: {e}")
        print("እባክህ ሁሉም ፋይሎች በትክክል እንዳሉ ያረጋግጡ።")
        sys.exit(1)

if __name__ == "__main__":
    main()