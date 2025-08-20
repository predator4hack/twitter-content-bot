"""
LLM-powered content analysis for YouTube video transcripts.

This module provides integration with multiple LLM providers (Gemini, Groq) 
to analyze video transcripts and recommend engaging segments for Twitter clips.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
from enum import Enum

import google.generativeai as genai
from groq import Groq

from ..core.config import config
from ..transcription.base import TranscriptionResult

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Supported content types for analysis."""
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    INTERVIEW = "interview"
    TUTORIAL = "tutorial"
    NEWS = "news"
    REVIEW = "review"
    VLOG = "vlog"
    GAMING = "gaming"
    UNKNOWN = "unknown"


class HookStrength(Enum):
    """Hook strength levels for content segments."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ClipRecommendation:
    """
    Represents a recommended clip segment with analysis.
    
    Attributes:
        start_time: Start time in MM:SS format
        end_time: End time in MM:SS format
        reasoning: Explanation for why this segment is engaging
        confidence: Confidence score (0-100)
        hook_strength: Strength of the hook (high/medium/low)
        keywords: List of relevant keywords
        sentiment: Sentiment analysis (positive/negative/neutral)
    """
    start_time: str
    end_time: str
    reasoning: str
    confidence: int
    hook_strength: HookStrength
    keywords: List[str] = None
    sentiment: str = "neutral"
    
    def __post_init__(self):
        """Post-initialization to set defaults."""
        if self.keywords is None:
            self.keywords = []
    
    @property
    def start_seconds(self) -> float:
        """Convert start_time to seconds."""
        return self._time_to_seconds(self.start_time)
    
    @property
    def end_seconds(self) -> float:
        """Convert end_time to seconds."""
        return self._time_to_seconds(self.end_time)
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.end_seconds - self.start_seconds
    
    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert MM:SS or HH:MM:SS to seconds."""
        parts = time_str.split(":")
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            raise ValueError(f"Invalid time format: {time_str}")


@dataclass
class AnalysisResult:
    """
    Complete analysis result for a video transcript.
    
    Attributes:
        content_type: Detected content type
        recommendations: List of recommended clips
        summary: Brief summary of the content
        total_duration: Total video duration
        analysis_time: Time taken for analysis (seconds)
        provider: LLM provider used
        metadata: Additional analysis metadata
    """
    content_type: ContentType
    recommendations: List[ClipRecommendation]
    summary: str
    total_duration: float
    analysis_time: float
    provider: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Post-initialization to set defaults."""
        if self.metadata is None:
            self.metadata = {}


class BaseLLMAnalyzer(ABC):
    """Abstract base class for LLM-powered content analyzers."""
    
    @abstractmethod
    async def analyze_transcript(
        self, 
        transcript: TranscriptionResult,
        max_clips: int = 3,
        target_duration: int = 60
    ) -> AnalysisResult:
        """
        Analyze a transcript and recommend clips.
        
        Args:
            transcript: The transcription result to analyze
            max_clips: Maximum number of clips to recommend
            target_duration: Target duration for each clip in seconds
            
        Returns:
            Analysis result with recommendations
        """
        pass
    
    def _format_transcript_for_analysis(self, transcript: TranscriptionResult) -> str:
        """Format transcript with timestamps for LLM analysis."""
        formatted_segments = []
        for segment in transcript.segments:
            start_time = self._seconds_to_time(segment.start_time)
            end_time = self._seconds_to_time(segment.end_time)
            formatted_segments.append(f"[{start_time}-{end_time}]: {segment.text}")
        
        return "\n".join(formatted_segments)
    
    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _create_analysis_prompt(
        self,
        transcript: TranscriptionResult,
        max_clips: int,
        target_duration: int
    ) -> str:
        """Create the analysis prompt for the LLM."""
        formatted_transcript = self._format_transcript_for_analysis(transcript)
        total_duration = self._seconds_to_time(transcript.duration)
        
        prompt = f"""
Analyze this video transcript and identify the {max_clips} most engaging segments for Twitter clips.

Video Information:
- Total Duration: {total_duration}
- Language: {transcript.language}
- Target Clip Duration: ~{target_duration} seconds each

Transcript with Timestamps:
{formatted_transcript}

Instructions:
1. Identify the content type (educational, entertainment, interview, tutorial, news, review, vlog, gaming)
2. Find {max_clips} segments that would make engaging Twitter clips
3. Each segment should be around {target_duration} seconds (±20 seconds acceptable)
4. Focus on segments with:
   - Strong hooks or attention-grabbing openings
   - Surprising or unexpected information
   - Emotional moments or reactions
   - Clear, standalone value
   - Quotable moments
   - Visual interest (if mentioned in audio)

Return ONLY a valid JSON response in this exact format:
{{
  "content_type": "educational|entertainment|interview|tutorial|news|review|vlog|gaming|unknown",
  "summary": "Brief 1-2 sentence summary of the video content",
  "recommendations": [
    {{
      "start_time": "MM:SS",
      "end_time": "MM:SS",
      "reasoning": "Clear explanation of why this segment is engaging for Twitter",
      "confidence": 85,
      "hook_strength": "high|medium|low",
      "keywords": ["keyword1", "keyword2"],
      "sentiment": "positive|negative|neutral"
    }}
  ]
}}

Important: Ensure all timestamps exist in the provided transcript and clips don't exceed the video duration."""

        return prompt


class GeminiAnalyzer(BaseLLMAnalyzer):
    """Google Gemini-powered content analyzer."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini analyzer.
        
        Args:
            api_key: Gemini API key (uses config if not provided)
        """
        self.api_key = api_key or config.GOOGLE_API_KEY
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini analyzer")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    async def analyze_transcript(
        self,
        transcript: TranscriptionResult,
        max_clips: int = 3,
        target_duration: int = 60
    ) -> AnalysisResult:
        """Analyze transcript using Gemini."""
        start_time = time.time()
        
        try:
            prompt = self._create_analysis_prompt(transcript, max_clips, target_duration)
            
            logger.info(f"Sending analysis request to Gemini (max_clips={max_clips})")
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2048,
                )
            )
            
            analysis_time = time.time() - start_time
            
            # Parse the JSON response
            try:
                # Clean the response text - remove markdown code blocks if present
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]  # Remove ```json
                if response_text.startswith("```"):
                    response_text = response_text[3:]  # Remove ```
                if response_text.endswith("```"):
                    response_text = response_text[:-3]  # Remove trailing ```
                response_text = response_text.strip()
                
                result_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {e}")
                logger.error(f"Raw response: {response.text}")
                raise ValueError(f"Invalid JSON response from Gemini: {e}")
            
            # Convert to structured result
            recommendations = []
            for rec_data in result_data.get("recommendations", []):
                recommendation = ClipRecommendation(
                    start_time=rec_data["start_time"],
                    end_time=rec_data["end_time"],
                    reasoning=rec_data["reasoning"],
                    confidence=rec_data["confidence"],
                    hook_strength=HookStrength(rec_data["hook_strength"]),
                    keywords=rec_data.get("keywords", []),
                    sentiment=rec_data.get("sentiment", "neutral")
                )
                recommendations.append(recommendation)
            
            result = AnalysisResult(
                content_type=ContentType(result_data["content_type"]),
                recommendations=recommendations,
                summary=result_data["summary"],
                total_duration=transcript.duration,
                analysis_time=analysis_time,
                provider="gemini",
                metadata={
                    "model": "gemini-1.5-flash",
                    "tokens_used": len(response.text),
                    "raw_response": response.text
                }
            )
            
            logger.info(f"Gemini analysis completed in {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            analysis_time = time.time() - start_time
            logger.error(f"Gemini analysis failed after {analysis_time:.2f}s: {e}")
            raise


class GroqAnalyzer(BaseLLMAnalyzer):
    """Groq-powered content analyzer."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Groq analyzer.
        
        Args:
            api_key: Groq API key (uses config if not provided)
        """
        self.api_key = api_key or config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("Groq API key is required for Groq analyzer")
        
        self.client = Groq(api_key=self.api_key)
        
    async def analyze_transcript(
        self,
        transcript: TranscriptionResult,
        max_clips: int = 3,
        target_duration: int = 60
    ) -> AnalysisResult:
        """Analyze transcript using Groq."""
        start_time = time.time()
        
        try:
            prompt = self._create_analysis_prompt(transcript, max_clips, target_duration)
            
            logger.info(f"Sending analysis request to Groq (max_clips={max_clips})")
            
            response = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert content analyzer specializing in identifying engaging video segments for social media. Always respond with valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2048,
            )
            
            analysis_time = time.time() - start_time
            
            # Parse the JSON response
            try:
                # Clean the response text - remove markdown code blocks if present
                response_text = response.choices[0].message.content or ""
                response_text = response_text.strip()
                
                # Remove common prefixes and markdown formatting
                if "Here is the" in response_text and "JSON" in response_text:
                    # Find the first line that starts with {
                    lines = response_text.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith('{'):
                            response_text = '\n'.join(lines[i:])
                            break
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                result_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Groq response as JSON: {e}")
                logger.error(f"Raw response: {response.choices[0].message.content}")
                raise ValueError(f"Invalid JSON response from Groq: {e}")
            
            # Convert to structured result
            recommendations = []
            for rec_data in result_data.get("recommendations", []):
                recommendation = ClipRecommendation(
                    start_time=rec_data["start_time"],
                    end_time=rec_data["end_time"],
                    reasoning=rec_data["reasoning"],
                    confidence=rec_data["confidence"],
                    hook_strength=HookStrength(rec_data["hook_strength"]),
                    keywords=rec_data.get("keywords", []),
                    sentiment=rec_data.get("sentiment", "neutral")
                )
                recommendations.append(recommendation)
            
            result = AnalysisResult(
                content_type=ContentType(result_data["content_type"]),
                recommendations=recommendations,
                summary=result_data["summary"],
                total_duration=transcript.duration,
                analysis_time=analysis_time,
                provider="groq",
                metadata={
                    "model": "llama3-8b-8192",
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "raw_response": response.choices[0].message.content
                }
            )
            
            logger.info(f"Groq analysis completed in {analysis_time:.2f}s")
            return result
            
        except Exception as e:
            analysis_time = time.time() - start_time
            logger.error(f"Groq analysis failed after {analysis_time:.2f}s: {e}")
            raise


class LLMAnalyzerFactory:
    """Factory for creating LLM analyzers based on configuration."""
    
    @staticmethod
    def create_analyzer(provider: Optional[str] = None) -> BaseLLMAnalyzer:
        """
        Create an LLM analyzer instance.
        
        Args:
            provider: LLM provider ("gemini" or "groq", uses config default if None)
            
        Returns:
            Configured analyzer instance
            
        Raises:
            ValueError: If provider is not supported or API key is missing
        """
        provider = provider or config.DEFAULT_LLM_PROVIDER
        
        if provider.lower() == "gemini":
            return GeminiAnalyzer()
        elif provider.lower() == "groq":
            return GroqAnalyzer()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available LLM providers based on configured API keys."""
        providers = []
        if config.GOOGLE_API_KEY:
            providers.append("gemini")
        if config.GROQ_API_KEY:
            providers.append("groq")
        return providers


# Convenience function for direct usage
async def analyze_content(
    transcript: TranscriptionResult,
    provider: Optional[str] = None,
    max_clips: int = 3,
    target_duration: int = 60
) -> AnalysisResult:
    """
    Analyze video transcript and get clip recommendations.
    
    Args:
        transcript: Transcription result to analyze
        provider: LLM provider to use (uses config default if None)
        max_clips: Maximum number of clips to recommend
        target_duration: Target duration for each clip in seconds
        
    Returns:
        Analysis result with recommendations
    """
    analyzer = LLMAnalyzerFactory.create_analyzer(provider)
    return await analyzer.analyze_transcript(transcript, max_clips, target_duration)
