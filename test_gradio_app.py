#!/usr/bin/env python3
"""
Simple Gradio Test App for Amharic LLM
Tests the interface without requiring model downloads
"""

import gradio as gr
import random
from datetime import datetime

def generate_amharic_text(prompt, max_length=100, temperature=0.7):
    """
    Mock text generation function for testing
    In production, this would use the actual trained model
    """
    
    # Sample Amharic responses for testing
    sample_responses = [
        "ሰላም ነው። እንዴት ነዎት? ዛሬ ቆንጆ ቀን ነው።",
        "ኢትዮጵያ በአፍሪካ ቀንድ የምትገኝ ሀገር ናት። ታሪካዊ እና ባህላዊ ሀብት የበዛባት ሀገር ናት።",
        "አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት። በዚህ ከተማ ውስጥ ብዙ ሰዎች ይኖራሉ።",
        "ትምህርት በሰው ልጅ ህይወት ውስጥ በጣም አስፈላጊ ነው። ትምህርት ሰውን ያበቃዋል።",
        "ቴክኖሎጂ በዘመናችን ህይወት ውስጥ ትልቅ ሚና ይጫወታል። ኮምፒውተር እና ስማርት ፎን ህይወታችንን ቀይረውታል።"
    ]
    
    # Simulate processing time
    import time
    time.sleep(1)
    
    # Select a random response or create a simple continuation
    if prompt.strip():
        response = f"{prompt} {random.choice(sample_responses)}"
    else:
        response = random.choice(sample_responses)
    
    # Truncate to max_length
    if len(response) > max_length:
        response = response[:max_length] + "..."
    
    return response

def create_interface():
    """
    Create the Gradio interface
    """
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .amharic-text {
        font-family: 'Noto Sans Ethiopic', 'Nyala', serif;
        font-size: 16px;
        line-height: 1.6;
    }
    """
    
    with gr.Blocks(css=css, title="🇪🇹 Amharic LLM Test") as interface:
        gr.Markdown(
            """
            # 🇪🇹 Amharic Language Model - Test Interface
            
            This is a test interface for the Amharic LLM. In production, this will be connected to a trained model.
            Currently showing mock responses for interface testing.
            
            **አማርኛ ቋንቋ ሞዴል - የሙከራ መገናኛ**
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="📝 Enter Amharic text prompt / አማርኛ ጽሁፍ ያስገቡ",
                    placeholder="ሰላም... (Type in Amharic)",
                    lines=3,
                    elem_classes=["amharic-text"]
                )
                
                with gr.Row():
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=150,
                        step=10,
                        label="📏 Maximum Length"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="🌡️ Temperature"
                    )
                
                generate_btn = gr.Button(
                    "🚀 Generate Text / ጽሁፍ ፍጠር",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=2):
                output_text = gr.Textbox(
                    label="📄 Generated Text / የተፈጠረ ጽሁፍ",
                    lines=8,
                    elem_classes=["amharic-text"],
                    interactive=False
                )
                
                gr.Markdown(
                    """
                    ### 📋 Sample Prompts / የናሙና ጽሁፎች:
                    - ሰላም ነው
                    - ኢትዮጵያ ስለ
                    - አዲስ አበባ
                    - ትምህርት ስለ
                    - ቴክኖሎጂ
                    """
                )
        
        # Event handlers
        generate_btn.click(
            fn=generate_amharic_text,
            inputs=[prompt_input, max_length, temperature],
            outputs=output_text
        )
        
        # Example inputs
        gr.Examples(
            examples=[
                ["ሰላም ነው", 100, 0.7],
                ["ኢትዮጵያ ስለ", 150, 0.8],
                ["አዲስ አበባ", 120, 0.6],
                ["ትምህርት ስለ", 180, 0.7],
                ["ቴክኖሎጂ", 140, 0.9]
            ],
            inputs=[prompt_input, max_length, temperature],
            outputs=output_text,
            fn=generate_amharic_text,
            cache_examples=False
        )
        
        gr.Markdown(
            """
            ---
            
            **🔧 Development Status:**
            - ✅ Interface: Working
            - ⏳ Model: In Training
            - ⏳ Data: Collecting
            - ⏳ Deployment: Pending
            
            **📊 Current Phase:** Local Development & Testing
            
            *This is a development preview. The actual model will be trained on Kaggle and deployed to Hugging Face Spaces.*
            """
        )
    
    return interface

def main():
    """
    Main function to launch the interface
    """
    print("🇪🇹 Starting Amharic LLM Test Interface...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    interface = create_interface()
    
    # Launch the interface
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )

if __name__ == "__main__":
    main()