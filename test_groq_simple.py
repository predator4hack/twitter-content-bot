#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from example_llm_analysis import create_sample_transcript
from src.analyzer import LLMAnalyzerFactory

async def test_groq():
    try:
        print("Creating transcript...")
        transcript = create_sample_transcript()
        print(f"Transcript created with {len(transcript.segments)} segments")
        
        print("Creating Groq analyzer...")
        analyzer = LLMAnalyzerFactory.create_analyzer('groq')
        print("Analyzer created")
        
        print("Running analysis...")
        result = await analyzer.analyze_transcript(transcript, max_clips=2)
        print('✅ Groq analysis successful!')
        print(f'Content type: {result.content_type.value}')
        print(f'Clips: {len(result.recommendations)}')
        print(f'Time: {result.analysis_time:.2f}s')
        
    except Exception as e:
        print(f'❌ Groq failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_groq())
