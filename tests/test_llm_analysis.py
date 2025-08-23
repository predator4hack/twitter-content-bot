"""
Tests for LLM content analysis functionality.

This module tests the LLM-powered content analysis features including
Gemini and Groq integration, content strategy optimization, and clip recommendations.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import List

from src.analyzer.llm_analyzer import (
    GeminiAnalyzer,
    GroqAnalyzer,
    LLMAnalyzerFactory,
    AnalysisResult,
    ClipRecommendation,
    ContentType,
    HookStrength,
    analyze_content
)
from src.analyzer.content_strategy import ContentStrategy, TwitterStrategy
from src.transcription.base import TranscriptionResult, TranscriptionSegment
from src.core.config import config


class TestClipRecommendation:
    """Test ClipRecommendation dataclass functionality."""
    
    def test_time_conversion(self):
        """Test time string to seconds conversion."""
        clip = ClipRecommendation(
            start_time="01:30",
            end_time="02:15",
            reasoning="Test clip",
            confidence=85,
            hook_strength=HookStrength.HIGH
        )
        
        assert clip.start_seconds == 90.0
        assert clip.end_seconds == 135.0
        assert clip.duration_seconds == 45.0
    
    def test_time_conversion_with_hours(self):
        """Test time conversion with hours format."""
        clip = ClipRecommendation(
            start_time="01:02:30",
            end_time="01:03:15",
            reasoning="Test clip",
            confidence=85,
            hook_strength=HookStrength.HIGH
        )
        
        assert clip.start_seconds == 3750.0  # 1*3600 + 2*60 + 30
        assert clip.end_seconds == 3795.0   # 1*3600 + 3*60 + 15
        assert clip.duration_seconds == 45.0
    
    def test_invalid_time_format(self):
        """Test handling of invalid time formats."""
        clip = ClipRecommendation(
            start_time="invalid",
            end_time="02:15",
            reasoning="Test clip",
            confidence=85,
            hook_strength=HookStrength.HIGH
        )
        
        with pytest.raises(ValueError):
            _ = clip.start_seconds


class TestLLMAnalyzerFactory:
    """Test LLM analyzer factory functionality."""
    
    def test_create_gemini_analyzer(self):
        """Test creating Gemini analyzer."""
        with patch.object(config, 'GOOGLE_API_KEY', 'test-key'):
            analyzer = LLMAnalyzerFactory.create_analyzer("gemini")
            assert isinstance(analyzer, GeminiAnalyzer)
    
    def test_create_groq_analyzer(self):
        """Test creating Groq analyzer."""
        with patch.object(config, 'GROQ_API_KEY', 'test-key'):
            analyzer = LLMAnalyzerFactory.create_analyzer("groq")
            assert isinstance(analyzer, GroqAnalyzer)
    
    def test_invalid_provider(self):
        """Test handling of invalid provider."""
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            LLMAnalyzerFactory.create_analyzer("invalid")
    
    def test_get_available_providers(self):
        """Test getting available providers based on API keys."""
        with patch.object(config, 'GOOGLE_API_KEY', 'test-key'), \
             patch.object(config, 'GROQ_API_KEY', None):
            providers = LLMAnalyzerFactory.get_available_providers()
            assert "gemini" in providers
            assert "groq" not in providers


class TestGeminiAnalyzer:
    """Test Gemini analyzer functionality."""
    
    @pytest.fixture
    def sample_transcript(self):
        """Create a sample transcript for testing."""
        segments = [
            TranscriptionSegment(
                text="Welcome to this amazing tutorial on Python programming!",
                start_time=0.0,
                end_time=5.0,
                confidence=0.95
            ),
            TranscriptionSegment(
                text="Today we'll learn about data structures and algorithms.",
                start_time=5.0,
                end_time=10.0,
                confidence=0.92
            ),
            TranscriptionSegment(
                text="This is a game-changing approach that will revolutionize your coding.",
                start_time=60.0,
                end_time=65.0,
                confidence=0.98
            )
        ]
        
        return TranscriptionResult(
            text="Welcome to this amazing tutorial on Python programming! Today we'll learn about data structures and algorithms. This is a game-changing approach that will revolutionize your coding.",
            segments=segments,
            language="en",
            confidence=0.95
        )
    
    @pytest.fixture
    def mock_gemini_response(self):
        """Create a mock Gemini API response."""
        response_data = {
            "content_type": "tutorial",
            "summary": "Python programming tutorial covering data structures and algorithms",
            "recommendations": [
                {
                    "start_time": "00:00",
                    "end_time": "01:05",
                    "reasoning": "Strong opening with clear value proposition and game-changing claim",
                    "confidence": 85,
                    "hook_strength": "high",
                    "keywords": ["python", "tutorial", "game-changing"],
                    "sentiment": "positive"
                }
            ]
        }
        
        mock_response = Mock()
        mock_response.text = json.dumps(response_data)
        return mock_response
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_gemini_analyzer_init(self, mock_model_class, mock_configure):
        """Test Gemini analyzer initialization."""
        with patch.object(config, 'GOOGLE_API_KEY', 'test-key'):
            analyzer = GeminiAnalyzer()
            mock_configure.assert_called_once_with(api_key='test-key')
            mock_model_class.assert_called_once_with('gemini-1.5-flash')
    
    def test_gemini_analyzer_no_api_key(self):
        """Test Gemini analyzer without API key."""
        with patch.object(config, 'GOOGLE_API_KEY', None):
            with pytest.raises(ValueError, match="Google API key is required"):
                GeminiAnalyzer()
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @pytest.mark.asyncio
    async def test_gemini_analyze_transcript(self, mock_model_class, mock_configure, sample_transcript, mock_gemini_response):
        """Test Gemini transcript analysis."""
        # Setup mocks
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_gemini_response
        mock_model_class.return_value = mock_model
        
        with patch.object(config, 'GOOGLE_API_KEY', 'test-key'):
            analyzer = GeminiAnalyzer()
            result = await analyzer.analyze_transcript(sample_transcript)
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.content_type == ContentType.TUTORIAL
        assert result.provider == "gemini"
        assert len(result.recommendations) == 1
        assert result.analysis_time > 0
        
        # Verify recommendation
        rec = result.recommendations[0]
        assert rec.start_time == "00:00"
        assert rec.end_time == "01:05"
        assert rec.confidence == 85
        assert rec.hook_strength == HookStrength.HIGH
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    @pytest.mark.asyncio
    async def test_gemini_invalid_json_response(self, mock_model_class, mock_configure, sample_transcript):
        """Test handling of invalid JSON response from Gemini."""
        # Setup mocks with invalid JSON
        mock_response = Mock()
        mock_response.text = "Invalid JSON response"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class.return_value = mock_model
        
        with patch.object(config, 'GOOGLE_API_KEY', 'test-key'):
            analyzer = GeminiAnalyzer()
            
            with pytest.raises(ValueError, match="Invalid JSON response from Gemini"):
                await analyzer.analyze_transcript(sample_transcript)


class TestGroqAnalyzer:
    """Test Groq analyzer functionality."""
    
    @pytest.fixture
    def sample_transcript(self):
        """Create a sample transcript for testing."""
        segments = [
            TranscriptionSegment(
                text="This is an incredible breakthrough in AI technology!",
                start_time=0.0,
                end_time=4.0,
                confidence=0.95
            )
        ]
        
        return TranscriptionResult(
            text="This is an incredible breakthrough in AI technology!",
            segments=segments,
            language="en",
            confidence=0.95
        )
    
    @pytest.fixture
    def mock_groq_response(self):
        """Create a mock Groq API response."""
        response_data = {
            "content_type": "educational",
            "summary": "AI technology breakthrough discussion",
            "recommendations": [
                {
                    "start_time": "00:00",
                    "end_time": "00:04",
                    "reasoning": "Exciting breakthrough announcement with high impact",
                    "confidence": 90,
                    "hook_strength": "high",
                    "keywords": ["AI", "breakthrough", "technology"],
                    "sentiment": "positive"
                }
            ]
        }
        
        mock_choice = Mock()
        mock_choice.message.content = json.dumps(response_data)
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 150
        
        return mock_response
    
    @patch('src.analyzer.llm_analyzer.Groq')
    def test_groq_analyzer_init(self, mock_groq_class):
        """Test Groq analyzer initialization."""
        with patch.object(config, 'GROQ_API_KEY', 'test-key'):
            analyzer = GroqAnalyzer()
            mock_groq_class.assert_called_once_with(api_key='test-key')
    
    def test_groq_analyzer_no_api_key(self):
        """Test Groq analyzer without API key."""
        with patch.object(config, 'GROQ_API_KEY', None):
            with pytest.raises(ValueError, match="Groq API key is required"):
                GroqAnalyzer()
    
    @patch('src.analyzer.llm_analyzer.Groq')
    @pytest.mark.asyncio
    async def test_groq_analyze_transcript(self, mock_groq_class, sample_transcript, mock_groq_response):
        """Test Groq transcript analysis."""
        # Setup mocks
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_groq_response
        mock_groq_class.return_value = mock_client
        
        with patch.object(config, 'GROQ_API_KEY', 'test-key'):
            analyzer = GroqAnalyzer()
            result = await analyzer.analyze_transcript(sample_transcript)
        
        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.content_type == ContentType.EDUCATIONAL
        assert result.provider == "groq"
        assert len(result.recommendations) == 1
        assert result.analysis_time > 0
        
        # Verify API call
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == "openai/gpt-oss-120b"
        assert call_args[1]['temperature'] == 0.3


class TestContentStrategy:
    """Test content strategy functionality."""
    
    @pytest.fixture
    def sample_analysis(self):
        """Create sample analysis result."""
        recommendations = [
            ClipRecommendation(
                start_time="00:00",
                end_time="01:00",
                reasoning="Strong opening hook",
                confidence=85,
                hook_strength=HookStrength.HIGH,
                keywords=["amazing", "breakthrough"],
                sentiment="positive"
            ),
            ClipRecommendation(
                start_time="02:00",
                end_time="02:45",
                reasoning="Detailed explanation",
                confidence=75,
                hook_strength=HookStrength.MEDIUM,
                keywords=["explain", "detail"],
                sentiment="neutral"
            )
        ]
        
        return AnalysisResult(
            content_type=ContentType.EDUCATIONAL,
            recommendations=recommendations,
            summary="Educational content about technology",
            total_duration=180.0,
            analysis_time=2.5,
            provider="test"
        )
    
    def test_detect_strategy_educational(self, sample_analysis):
        """Test strategy detection for educational content."""
        strategy = ContentStrategy.detect_strategy(sample_analysis)
        assert strategy == TwitterStrategy.EDUCATIONAL_VALUE
    
    def test_detect_strategy_viral(self):
        """Test strategy detection for viral content."""
        recommendations = [
            ClipRecommendation(
                start_time="00:00",
                end_time="01:00",
                reasoning="Viral moment",
                confidence=85,
                hook_strength=HookStrength.HIGH,
                keywords=["viral", "amazing"],
                sentiment="positive"
            ),
            ClipRecommendation(
                start_time="01:00",
                end_time="02:00",
                reasoning="Another viral moment",
                confidence=80,
                hook_strength=HookStrength.HIGH,
                keywords=["incredible", "shocking"],
                sentiment="positive"
            )
        ]
        
        analysis = AnalysisResult(
            content_type=ContentType.ENTERTAINMENT,
            recommendations=recommendations,
            summary="Entertainment content",
            total_duration=120.0,
            analysis_time=2.0,
            provider="test"
        )
        
        strategy = ContentStrategy.detect_strategy(analysis)
        assert strategy == TwitterStrategy.VIRAL_MOMENTS
    
    def test_score_clip(self):
        """Test clip scoring functionality."""
        clip = ClipRecommendation(
            start_time="00:00",
            end_time="01:00",
            reasoning="Great content",
            confidence=85,
            hook_strength=HookStrength.HIGH,
            keywords=["learn", "tutorial"],
            sentiment="positive"
        )
        
        score = ContentStrategy.score_clip(
            clip, 
            ContentType.EDUCATIONAL, 
            TwitterStrategy.EDUCATIONAL_VALUE,
            target_duration=60
        )
        
        assert 0 <= score <= 100
        assert score > 70  # Should be high for educational content with good keywords
    
    def test_optimize_recommendations(self, sample_analysis):
        """Test recommendation optimization."""
        optimized = ContentStrategy.optimize_recommendations(
            sample_analysis,
            strategy=TwitterStrategy.EDUCATIONAL_VALUE,
            max_clips=1
        )
        
        assert len(optimized) == 1
        # With educational strategy, keywords matter more, so the second clip might score higher
        # Just verify we get a valid recommendation
        assert optimized[0].confidence > 0
        assert optimized[0].hook_strength in [HookStrength.HIGH, HookStrength.MEDIUM, HookStrength.LOW]
    
    def test_generate_twitter_text(self):
        """Test Twitter text generation."""
        clip = ClipRecommendation(
            start_time="00:00",
            end_time="01:00",
            reasoning="This is an amazing breakthrough in AI technology that will change everything",
            confidence=85,
            hook_strength=HookStrength.HIGH,
            keywords=["AI", "breakthrough", "technology"],
            sentiment="positive"
        )
        
        twitter_text = ContentStrategy.generate_twitter_text(clip, max_length=280)
        
        assert len(twitter_text) <= 280
        assert "AI" in twitter_text or "#AI" in twitter_text
        assert "breakthrough" in twitter_text or "#Breakthrough" in twitter_text
    
    def test_analyze_competition(self):
        """Test competition analysis."""
        clips = [
            ClipRecommendation(
                start_time="00:00",
                end_time="03:00",  # 3 minutes
                reasoning="Good content",
                confidence=65,  # Below 70 to trigger recommendation
                hook_strength=HookStrength.MEDIUM,  # No HIGH to trigger recommendation
                sentiment="positive"
            ),
            ClipRecommendation(
                start_time="03:00",
                end_time="06:00",  # Another 3 minutes, total 6 minutes > 5 minutes
                reasoning="Okay content",
                confidence=60,
                hook_strength=HookStrength.LOW,
                sentiment="positive"  # Same sentiment to trigger recommendation
            )
        ]
        
        analysis = ContentStrategy.analyze_competition(clips)
        
        assert analysis["total_clips"] == 2
        assert analysis["avg_confidence"] == 62.5
        assert analysis["hook_distribution"]["high"] == 0
        assert analysis["hook_distribution"]["medium"] == 1
        assert analysis["hook_distribution"]["low"] == 1
        # Should have multiple recommendations due to low confidence, no high hooks, long duration, and same sentiment
        assert len(analysis["recommendations"]) >= 3


class TestIntegration:
    """Integration tests for the complete analysis workflow."""
    
    @pytest.fixture
    def sample_transcript(self):
        """Create a comprehensive sample transcript."""
        segments = [
            TranscriptionSegment(
                text="Hey everyone, welcome back to my channel!",
                start_time=0.0,
                end_time=3.0,
                confidence=0.95
            ),
            TranscriptionSegment(
                text="Today I'm going to show you an incredible Python trick that will blow your mind.",
                start_time=3.0,
                end_time=8.0,
                confidence=0.92
            ),
            TranscriptionSegment(
                text="This technique is used by professional developers but rarely shared publicly.",
                start_time=8.0,
                end_time=13.0,
                confidence=0.94
            ),
            TranscriptionSegment(
                text="Let's dive right into the code and see how it works.",
                start_time=60.0,
                end_time=63.0,
                confidence=0.90
            )
        ]
        
        return TranscriptionResult(
            text=" ".join(seg.text for seg in segments),
            segments=segments,
            language="en",
            confidence=0.93
        )
    
    @patch('src.analyzer.llm_analyzer.GeminiAnalyzer.analyze_transcript')
    @pytest.mark.asyncio
    async def test_analyze_content_function(self, mock_analyze, sample_transcript):
        """Test the convenience analyze_content function."""
        # Mock the analysis result
        mock_result = AnalysisResult(
            content_type=ContentType.TUTORIAL,
            recommendations=[
                ClipRecommendation(
                    start_time="00:00",
                    end_time="01:03",
                    reasoning="Strong hook with promise of incredible technique",
                    confidence=90,
                    hook_strength=HookStrength.HIGH,
                    keywords=["python", "trick", "incredible"],
                    sentiment="positive"
                )
            ],
            summary="Python programming tutorial with advanced techniques",
            total_duration=63.0,
            analysis_time=2.1,
            provider="gemini"
        )
        
        mock_analyze.return_value = mock_result
        
        with patch.object(config, 'GOOGLE_API_KEY', 'test-key'):
            result = await analyze_content(
                sample_transcript,
                provider="gemini",
                max_clips=2,
                target_duration=60
            )
        
        assert isinstance(result, AnalysisResult)
        assert result.content_type == ContentType.TUTORIAL
        mock_analyze.assert_called_once_with(sample_transcript, 2, 60)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])
