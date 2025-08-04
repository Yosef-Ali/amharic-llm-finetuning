#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template_string, request, jsonify
import torch
import json
import os

# Simple mock model for testing
class MockAmharicModel:
    def __init__(self):
        self.responses = [
            "ሰላም! እንዴት ነዎት?",
            "ጥሩ ጥያቄ ነው። የበለጠ ንገሩኝ።",
            "አመሰግናለሁ። ሌላ ነገር ልረዳዎት?",
            "በጣም አስደሳች ነው!",
            "እሺ፣ ተረድቻለሁ።",
            "ይህ ጥሩ ሀሳብ ነው።",
            "እንዲህ ነው የሚሰራው።",
            "ተጨማሪ መረጃ ይፈልጋሉ?"
        ]
        self.counter = 0
    
    def generate_response(self, prompt):
        # Simple response generation for testing
        response = self.responses[self.counter % len(self.responses)]
        self.counter += 1
        return response

app = Flask(__name__)
model = MockAmharicModel()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🇪🇹 Smart Amharic Conversational AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            padding: 20px;
            box-sizing: border-box;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            width: 100%;
            max-width: 800px;
            min-height: 600px;
            max-height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        
        .messages {
            flex: 1;
            overflow-y: auto;
            overflow-x: hidden;
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #eee;
            border-radius: 15px;
            background: #f9f9f9;
            scroll-behavior: smooth;
            position: relative;
            min-height: 300px;
            max-height: calc(90vh - 200px);
        }
        
        .messages::-webkit-scrollbar {
            width: 6px;
        }
        
        .messages::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        
        .messages::-webkit-scrollbar-thumb {
            background: #c1c1c1;
            border-radius: 3px;
        }
        
        .messages::-webkit-scrollbar-thumb:hover {
            background: #a8a8a8;
        }
        
        .scroll-indicator {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(102, 126, 234, 0.8);
            color: white;
            border: none;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            cursor: pointer;
            display: none;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: all 0.3s;
            z-index: 10;
        }
        
        .scroll-indicator:hover {
            background: rgba(102, 126, 234, 1);
            transform: scale(1.1);
        }
        
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 18px;
            max-width: 85%;
            word-wrap: break-word;
            white-space: pre-wrap;
            line-height: 1.5;
            position: relative;
            animation: fadeIn 0.3s ease-in;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background: #667eea;
            color: white;
            margin-left: auto;
            text-align: right;
        }
        
        .bot-message {
            background: #e3f2fd;
            color: #333;
            margin-right: auto;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
            align-items: flex-end;
            position: relative;
        }
        
        .char-counter {
            position: absolute;
            bottom: -20px;
            right: 80px;
            font-size: 12px;
            color: #666;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .char-counter.visible {
            opacity: 1;
        }
        
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 5px;
            color: #666;
            font-style: italic;
            margin-bottom: 10px;
        }
        
        .typing-dots {
            display: flex;
            gap: 3px;
        }
        
        .typing-dots span {
            width: 6px;
            height: 6px;
            background: #667eea;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        
        .typing-dots span:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dots span:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        .message-input {
            flex: 1;
            padding: 12px 15px;
            border: 2px solid #ddd;
            border-radius: 15px;
            font-size: 16px;
            outline: none;
            transition: all 0.3s;
            resize: none;
            min-height: 50px;
            max-height: 150px;
            overflow-y: auto;
            font-family: inherit;
            line-height: 1.4;
        }
        
        .message-input:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .send-button {
            padding: 12px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            min-width: 60px;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 50px;
        }
        
        .send-button:hover:not(:disabled) {
            background: #5a6fd8;
            transform: translateY(-1px);
        }
        
        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        .examples {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }
        
        .example-btn {
            padding: 5px 10px;
            background: #f0f0f0;
            border: 1px solid #ddd;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
            transition: background 0.3s;
        }
        
        .example-btn:hover {
            background: #e0e0e0;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 10px;
                align-items: stretch;
            }
            
            .container {
                min-height: calc(100vh - 20px);
                max-height: calc(100vh - 20px);
                border-radius: 15px;
                margin: 0;
                max-width: 100%;
            }
            
            .messages {
                min-height: 250px;
                max-height: calc(100vh - 300px);
            }
            
            .message {
                max-width: 90%;
                font-size: 14px;
            }
            
            .header h1 {
                font-size: 20px;
            }
            
            .header p {
                font-size: 12px;
            }
            
            .message-input {
                font-size: 14px;
                max-height: 120px;
            }
            
            .send-button {
                padding: 10px 15px;
                font-size: 14px;
                min-width: 50px;
                height: 45px;
            }
            
            .examples {
                flex-direction: column;
                gap: 8px;
                max-height: 120px;
                overflow-y: auto;
            }
            
            .example-btn {
                width: 100%;
                text-align: left;
                padding: 8px 12px;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🇪🇹 Smart Amharic Conversational AI</h1>
            <p>Chat with an AI that understands Amharic language and culture!</p>
        </div>
        
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message bot-message">
                    ሰላም! እንዴት ነዎት? በአማርኛ ወይም በእንግሊዝኛ መጻፍ ይችላሉ።
                </div>
                <button class="scroll-indicator" id="scrollIndicator" onclick="scrollToBottom()" title="Scroll to bottom">↓</button>
            </div>
            
            <div class="typing-indicator" id="typingIndicator">
                <span>AI is typing</span>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
            
            <div class="input-container">
                <textarea class="message-input" id="messageInput" placeholder="መልእክትዎን እዚህ ይጻፉ..." onkeydown="handleKeyDown(event)" oninput="handleInput(event)" rows="1"></textarea>
                <button class="send-button" id="sendButton" onclick="sendMessage()">ላክ</button>
                <div class="char-counter" id="charCounter">0/1000</div>
            </div>
            
            <div class="examples">
                <button class="example-btn" onclick="setMessage('ሰላም እንዴት ነህ?')">ሰላም እንዴት ነህ?</button>
                <button class="example-btn" onclick="setMessage('የአማርኛ ቋንቋ ታሪክ ንገረኝ')">የአማርኛ ቋንቋ ታሪክ ንገረኝ</button>
                <button class="example-btn" onclick="setMessage('What is Ethiopia known for?')">What is Ethiopia known for?</button>
                <button class="example-btn" onclick="setMessage('እንዴት ነው የሚሰራው?')">እንዴት ነው የሚሰራው?</button>
            </div>
        </div>
    </div>
    
    <script>
        let isTyping = false;
        const maxChars = 1000;
        
        function addMessage(content, isUser) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
            messageDiv.textContent = content;
            
            // Add animation
            messageDiv.style.opacity = '0';
            messageDiv.style.transform = 'translateY(20px)';
            
            messagesDiv.appendChild(messageDiv);
            
            // Ensure messages container can handle overflow
            const containerHeight = messagesDiv.scrollHeight;
            const visibleHeight = messagesDiv.clientHeight;
            
            if (containerHeight > visibleHeight) {
                messagesDiv.style.overflowY = 'auto';
            }
            
            // Trigger animation
            setTimeout(() => {
                messageDiv.style.opacity = '1';
                messageDiv.style.transform = 'translateY(0)';
                scrollToBottom();
                updateScrollIndicator();
            }, 10);
        }
        
        function scrollToBottom() {
            const messagesDiv = document.getElementById('messages');
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function updateScrollIndicator() {
            const messagesDiv = document.getElementById('messages');
            const scrollIndicator = document.getElementById('scrollIndicator');
            const isAtBottom = messagesDiv.scrollTop + messagesDiv.clientHeight >= messagesDiv.scrollHeight - 10;
            
            if (isAtBottom || messagesDiv.children.length <= 2) {
                scrollIndicator.style.display = 'none';
            } else {
                scrollIndicator.style.display = 'flex';
            }
        }
        
        function autoResize(textarea) {
            textarea.style.height = 'auto';
            const newHeight = Math.min(textarea.scrollHeight, 150);
            textarea.style.height = newHeight + 'px';
        }
        
        function updateCharCounter(length) {
            const counter = document.getElementById('charCounter');
            counter.textContent = `${length}/${maxChars}`;
            counter.className = length > 0 ? 'char-counter visible' : 'char-counter';
            
            if (length > maxChars * 0.9) {
                counter.style.color = '#ff6b6b';
            } else if (length > maxChars * 0.7) {
                counter.style.color = '#ffa726';
            } else {
                counter.style.color = '#666';
            }
        }
        
        function showTypingIndicator() {
            const indicator = document.getElementById('typingIndicator');
            indicator.style.display = 'flex';
            isTyping = true;
        }
        
        function hideTypingIndicator() {
            const indicator = document.getElementById('typingIndicator');
            indicator.style.display = 'none';
            isTyping = false;
        }
        
        function setMessage(text) {
            const input = document.getElementById('messageInput');
            input.value = text;
            autoResize(input);
            updateCharCounter(text.length);
            input.focus();
            
            // Enable send button if text is valid
            const sendButton = document.getElementById('sendButton');
            const hasText = text.trim().length > 0;
            const withinLimit = text.length <= maxChars;
            sendButton.disabled = !hasText || !withinLimit;
            
            // Scroll to bottom of messages to show latest content
            scrollToBottom();
        }
        
        function handleInput(event) {
            const textarea = event.target;
            autoResize(textarea);
            updateCharCounter(textarea.value.length);
            
            const sendButton = document.getElementById('sendButton');
            const hasText = textarea.value.trim().length > 0;
            const withinLimit = textarea.value.length <= maxChars;
            sendButton.disabled = !hasText || !withinLimit;
            
            console.log('Input changed:', {
                text: textarea.value,
                hasText: hasText,
                withinLimit: withinLimit,
                buttonDisabled: sendButton.disabled
            });
        }
        
        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }
        
        async function sendMessage() {
            console.log('sendMessage called');
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            console.log('Message details:', {
                message: message,
                messageLength: message.length,
                maxChars: maxChars,
                isTyping: isTyping
            });
            
            if (!message || message.length > maxChars || isTyping) {
                console.log('Returning early - invalid message or typing');
                return;
            }
            
            const sendButton = document.getElementById('sendButton');
            sendButton.disabled = true;
            console.log('Send button disabled, proceeding with message send');
            
            addMessage(message, true);
            input.value = '';
            autoResize(input);
            updateCharCounter(0);
            
            showTypingIndicator();
            
            try {
                console.log('Sending request to /chat');
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                console.log('Response received:', response.status);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('Response data:', data);
                
                // Simulate typing delay for better UX
                setTimeout(() => {
                    hideTypingIndicator();
                    addMessage(data.response, false);
                    sendButton.disabled = false;
                    console.log('Message sent successfully');
                }, 500);
                
            } catch (error) {
                console.error('Error sending message:', error);
                hideTypingIndicator();
                addMessage('ይቅርታ፣ ችግር ገጠመኝ። እንደገና ይሞክሩ።', false);
                sendButton.disabled = false;
            }
        }
        
        // Initialize scroll indicator
        document.getElementById('messages').addEventListener('scroll', updateScrollIndicator);
        
        // Initialize send button state
        document.getElementById('sendButton').disabled = true;
        
        // Focus on input when page loads
        window.addEventListener('load', () => {
            document.getElementById('messageInput').focus();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message', '')
    
    if not message:
        return jsonify({'response': 'እባክዎ መልእክት ይጻፉ።'})
    
    try:
        response = model.generate_response(message)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': 'ይቅርታ፣ ችግር ገጠመኝ። እንደገና ይሞክሩ።'})

if __name__ == '__main__':
    print("🇪🇹 Starting Smart Amharic AI Flask App...")
    print("Open your browser and go to: http://127.0.0.1:5001")
    app.run(debug=True, host='127.0.0.1', port=5001)