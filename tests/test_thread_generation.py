"""
Unit tests for Twitter thread generation functionality.

Tests the TwitterThreadGenerator class and related thread generation features.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import only the data classes for testing
try:
    from src.analyzer.llm_analyzer import (
        ThreadTweet, TwitterThread, ContentType
    )
    from src.transcription.base import TranscriptionResult, TranscriptionSegment
except ImportError:
    # Fallback imports for testing
    pass


class TestThreadTweet:
    """Test ThreadTweet data class functionality."""
    
    def test_thread_tweet_creation(self):
        """Test basic ThreadTweet creation."""
        tweet = ThreadTweet(
            content="🧵 This is a test tweet about AI",
            tweet_number=1,
            character_count=0,  # Will be calculated in __post_init__
            hashtags=["#AI", "#test"],
            mentions=["@user"]
        )
        
        assert tweet.content == "🧵 This is a test tweet about AI"
        assert tweet.tweet_number == 1
        assert tweet.character_count == len(tweet.content)
        assert tweet.hashtags == ["#AI", "#test"]
        assert tweet.mentions == ["@user"]
    
    def test_thread_tweet_defaults(self):
        """Test ThreadTweet with default values."""
        tweet = ThreadTweet(
            content="Test tweet",
            tweet_number=2,
            character_count=0
        )
        
        assert tweet.hashtags == []
        assert tweet.mentions == []
        assert tweet.character_count == len("Test tweet")


class TestTwitterThread:
    """Test TwitterThread data class functionality."""
    
    def test_twitter_thread_creation(self):
        """Test basic TwitterThread creation."""
        tweets = [
            ThreadTweet("🧵 Hook tweet", 1, 0),
            ThreadTweet("2/ Second tweet", 2, 0),
            ThreadTweet("3/ Final tweet", 3, 0)
        ]
        
        thread = TwitterThread(
            tweets=tweets,
            hook_tweet=tweets[0],  # Will be overridden in __post_init__
            total_tweets=0,  # Will be calculated in __post_init__
            estimated_reading_time=60,
            reasoning="Test thread reasoning",
            content_type=ContentType.EDUCATIONAL,
            video_url="https://youtube.com/watch?v=123"
        )
        
        assert thread.total_tweets == 3
        assert thread.hook_tweet == tweets[0]
        assert thread.video_url == "https://youtube.com/watch?v=123"
        assert thread.content_type == ContentType.EDUCATIONAL


class TestTwitterThreadGenerator:
    """Test TwitterThreadGenerator class functionality."""
    
    @pytest.fixture
    def mock_transcript(self):
        """Create a mock transcription result."""
        segments = [
            TranscriptionSegment(
                text="Welcome to this educational video about machine learning.",
                start_time=0.0,
                end_time=5.0,
                confidence=0.95
            ),
            TranscriptionSegment(
                text="Today we'll learn about neural networks and how they work.",
                start_time=5.0,
                end_time=10.0,
                confidence=0.92
            ),
            TranscriptionSegment(
                text="Let's start with the basics of artificial intelligence.",
                start_time=10.0,
                end_time=15.0,
                confidence=0.88
            )
        ]
        
        return TranscriptionResult(
            text="Welcome to this educational video about machine learning. Today we'll learn about neural networks and how they work. Let's start with the basics of artificial intelligence.",
            segments=segments,
            language="en",
            confidence=0.92,
            metadata={"duration": 15.0}
        )
    
    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response for thread generation."""
        return {
            "content_type": "educational",
            "reasoning": "This educational video covers AI basics, perfect for a thread",
            "estimated_reading_time": 45,
            "tweets": [
                {
                    "tweet_number": 1,
                    "content": "🧵 This video breaks down machine learning in a way that actually makes sense. Here's what you need to know:",
                    "hashtags": ["#MachineLearning", "#AI"],
                    "mentions": []
                },
                {
                    "tweet_number": 2,
                    "content": "2/ Neural networks are the foundation of modern AI. Think of them as digital brains that learn patterns.",
                    "hashtags": ["#NeuralNetworks"],
                    "mentions": []
                },
                {
                    "tweet_number": 3,
                    "content": "3/ The key is understanding how these networks process information layer by layer.",
                    "hashtags": ["#Learning"],
                    "mentions": []
                },
                {
                    "tweet_number": 4,
                    "content": "4/ Watch the full explanation here: https://youtube.com/watch?v=123",
                    "hashtags": ["#watch"],
                    "mentions": []
                }
            ]
        }
    
    def test_thread_generator_initialization(self):
        """Test TwitterThreadGenerator initialization."""
        with patch('analyzer.thread_generator.LLMAnalyzerFactory.create_analyzer') as mock_factory:
            mock_analyzer = Mock()
            mock_factory.return_value = mock_analyzer
            
            generator = TwitterThreadGenerator("gemini")
            
            assert generator.provider == "gemini"
            assert generator.MAX_TWEETS == 6
            assert generator.MIN_TWEETS == 3
            assert generator.MAX_TWEET_LENGTH == 280
            mock_factory.assert_called_once_with("gemini")
    
    @pytest.mark.asyncio
    async def test_generate_thread_success(self, mock_transcript, mock_llm_response):
        """Test successful thread generation."""
        with patch('analyzer.thread_generator.LLMAnalyzerFactory.create_analyzer') as mock_factory:
            mock_analyzer = Mock()
            mock_factory.return_value = mock_analyzer
            
            generator = TwitterThreadGenerator("gemini")
            
            # Mock the LLM generation
            with patch.object(generator, '_generate_with_llm', return_value=mock_llm_response):
                thread = await generator.generate_thread(
                    transcript=mock_transcript,
                    video_title="AI Basics Tutorial",
                    video_url="https://youtube.com/watch?v=123",
                    target_length=4
                )
                
                assert isinstance(thread, TwitterThread)
                assert len(thread.tweets) == 4
                assert thread.content_type == ContentType.EDUCATIONAL
                assert thread.video_url == "https://youtube.com/watch?v=123"
                assert "🧵" in thread.hook_tweet.content
    
    @pytest.mark.asyncio
    async def test_generate_thread_fallback(self, mock_transcript):
        """Test thread generation with LLM failure and fallback."""
        with patch('analyzer.thread_generator.LLMAnalyzerFactory.create_analyzer') as mock_factory:
            mock_analyzer = Mock()
            mock_factory.return_value = mock_analyzer
            
            generator = TwitterThreadGenerator("gemini")
            
            # Mock the LLM generation to fail
            with patch.object(generator, '_generate_with_llm', side_effect=Exception("LLM failed")):
                with pytest.raises(Exception):
                    await generator.generate_thread(
                        transcript=mock_transcript,
                        video_title="Test Video"
                    )
    
    def test_create_thread_prompt(self, mock_transcript):
        """Test thread prompt creation."""
        with patch('analyzer.thread_generator.LLMAnalyzerFactory.create_analyzer') as mock_factory:
            mock_analyzer = Mock()
            mock_factory.return_value = mock_analyzer
            
            generator = TwitterThreadGenerator("gemini")
            
            prompt = generator._create_thread_prompt(
                transcript=mock_transcript,
                video_title="AI Tutorial",
                video_url="https://youtube.com/watch?v=123",
                target_length=4,
                tone="educational",
                target_audience="general"
            )
            
            assert "AI Tutorial" in prompt
            assert "4 tweets" in prompt
            assert "educational" in prompt
            assert "https://youtube.com/watch?v=123" in prompt
            assert "CRITICAL REQUIREMENTS" in prompt
    
    def test_clean_json_response(self):
        """Test JSON response cleaning functionality."""
        with patch('analyzer.thread_generator.LLMAnalyzerFactory.create_analyzer') as mock_factory:
            mock_analyzer = Mock()
            mock_factory.return_value = mock_analyzer
            
            generator = TwitterThreadGenerator("gemini")
            
            # Test with markdown code blocks
            markdown_response = "```json\n{\"test\": \"value\"}\n```"
            cleaned = generator._clean_json_response(markdown_response)
            assert cleaned == '{"test": "value"}'
            
            # Test with prefix text
            prefix_response = "Here is the JSON response:\n{\"test\": \"value\"}"
            cleaned = generator._clean_json_response(prefix_response)
            assert cleaned == '{"test": "value"}'
    
    def test_create_fallback_thread(self):
        """Test fallback thread creation."""
        with patch('analyzer.thread_generator.LLMAnalyzerFactory.create_analyzer') as mock_factory:
            mock_analyzer = Mock()
            mock_factory.return_value = mock_analyzer
            
            generator = TwitterThreadGenerator("gemini")
            
            fallback = generator._create_fallback_thread()
            
            assert fallback["content_type"] == "unknown"
            assert len(fallback["tweets"]) == 3
            assert fallback["tweets"][0]["content"].startswith("🧵")
            assert "[VIDEO_LINK]" in fallback["tweets"][-1]["content"]
    
    def test_validate_thread(self):
        """Test thread validation functionality."""
        with patch('analyzer.thread_generator.LLMAnalyzerFactory.create_analyzer') as mock_factory:
            mock_analyzer = Mock()
            mock_factory.return_value = mock_analyzer
            
            generator = TwitterThreadGenerator("gemini")
            
            # Test valid thread
            valid_tweets = [
                ThreadTweet("🧵 Great hook tweet that indicates threading", 1, 0),
                ThreadTweet("2/ Second tweet with good content", 2, 0),
                ThreadTweet("3/ Final tweet with video link", 3, 0)
            ]
            valid_thread = TwitterThread(
                tweets=valid_tweets,
                hook_tweet=valid_tweets[0],
                total_tweets=3,
                estimated_reading_time=45,
                reasoning="Test thread",
                content_type=ContentType.EDUCATIONAL,
                video_url="https://youtube.com/watch?v=123"
            )
            
            issues = generator.validate_thread(valid_thread)
            assert len(issues) == 1  # Missing video link in last tweet
            
            # Test invalid thread (too short)
            short_tweets = [ThreadTweet("🧵 Hook", 1, 0)]
            short_thread = TwitterThread(
                tweets=short_tweets,
                hook_tweet=short_tweets[0],
                total_tweets=1,
                estimated_reading_time=15,
                reasoning="Too short",
                content_type=ContentType.EDUCATIONAL
            )
            
            issues = generator.validate_thread(short_thread)
            assert any("too short" in issue for issue in issues)


class TestThreadStrategy:
    """Test ThreadStrategy class functionality."""
    
    def test_get_thread_strategy(self):
        """Test getting thread strategy for different content types."""
        edu_strategy = ThreadStrategy.get_thread_strategy(ContentType.EDUCATIONAL)
        assert edu_strategy["max_tweets"] == 6
        assert edu_strategy["tone"] == "explanatory"
        assert edu_strategy["hashtag_strategy"] == "educational"
        
        interview_strategy = ThreadStrategy.get_thread_strategy(ContentType.INTERVIEW)
        assert interview_strategy["max_tweets"] == 5
        assert interview_strategy["tone"] == "conversational"
    
    def test_generate_hook_suggestions(self):
        """Test hook suggestion generation."""
        suggestions = ThreadStrategy.generate_hook_suggestions(
            content_type=ContentType.EDUCATIONAL,
            video_title="Machine Learning Basics",
            key_insights=["Neural networks", "Deep learning", "AI applications"]
        )
        
        assert len(suggestions) == 3
        assert all("🧵" in suggestion for suggestion in suggestions)
        assert any("Machine Learning Basics" in suggestion for suggestion in suggestions)
    
    def test_optimize_thread_structure(self):
        """Test thread structure optimization."""
        # Create a thread that's too long
        tweets = [
            ThreadTweet(f"Tweet {i}", i, 0) for i in range(1, 9)  # 8 tweets
        ]
        long_thread = TwitterThread(
            tweets=tweets,
            hook_tweet=tweets[0],
            total_tweets=8,
            estimated_reading_time=120,
            reasoning="Long thread",
            content_type=ContentType.EDUCATIONAL
        )
        
        optimized = ThreadStrategy.optimize_thread_structure(long_thread)
        
        assert optimized.total_tweets <= 6  # Educational max
        assert optimized.hook_tweet.content == "Tweet 1"  # Hook preserved
        assert optimized.tweets[-1].content == "Tweet 8"  # Conclusion preserved
    
    def test_add_strategic_hashtags(self):
        """Test strategic hashtag addition."""
        tweets = [
            ThreadTweet("🧵 Hook tweet", 1, 0),
            ThreadTweet("2/ Middle tweet", 2, 0),
            ThreadTweet("3/ Final tweet", 3, 0)
        ]
        thread = TwitterThread(
            tweets=tweets,
            hook_tweet=tweets[0],
            total_tweets=3,
            estimated_reading_time=45,
            reasoning="Test thread",
            content_type=ContentType.EDUCATIONAL
        )
        
        enhanced = ThreadStrategy.add_strategic_hashtags(thread)
        
        # Hook tweet should have educational hashtags
        assert len(enhanced.tweets[0].hashtags) > 0
        assert any("#learning" in tag or "#education" in tag for tag in enhanced.tweets[0].hashtags)
        
        # Last tweet should have video hashtags
        assert any("#watch" in tag or "#video" in tag for tag in enhanced.tweets[-1].hashtags)
    
    def test_validate_thread_flow(self):
        """Test thread flow validation."""
        # Valid thread
        valid_tweets = [
            ThreadTweet("🧵 Good hook tweet that indicates threading clearly", 1, 0),
            ThreadTweet("2/ Second tweet with proper numbering", 2, 0),
            ThreadTweet("3/ Final tweet with video link: https://youtube.com/watch?v=123", 3, 0)
        ]
        valid_thread = TwitterThread(
            tweets=valid_tweets,
            hook_tweet=valid_tweets[0],
            total_tweets=3,
            estimated_reading_time=45,
            reasoning="Valid thread",
            content_type=ContentType.EDUCATIONAL,
            video_url="https://youtube.com/watch?v=123"
        )
        
        issues = ThreadStrategy.validate_thread_flow(valid_thread)
        assert len(issues) == 0
        
        # Invalid thread (no thread indicator)
        invalid_tweets = [
            ThreadTweet("Bad hook without threading indicator", 1, 0),
            ThreadTweet("Second tweet", 2, 0)
        ]
        invalid_thread = TwitterThread(
            tweets=invalid_tweets,
            hook_tweet=invalid_tweets[0],
            total_tweets=2,
            estimated_reading_time=30,
            reasoning="Invalid thread",
            content_type=ContentType.EDUCATIONAL
        )
        
        issues = ThreadStrategy.validate_thread_flow(invalid_thread)
        assert len(issues) > 0
        assert any("thread" in issue for issue in issues)


class TestConvenienceFunction:
    """Test the convenience generate_thread function."""
    
    @pytest.mark.asyncio
    async def test_generate_thread_function(self):
        """Test the convenience function for thread generation."""
        mock_transcript = TranscriptionResult(
            text="Test transcript",
            segments=[],
            language="en",
            confidence=0.9,
            metadata={"duration": 60.0}
        )
        
        with patch('analyzer.thread_generator.TwitterThreadGenerator') as mock_generator_class:
            mock_generator = Mock()
            mock_thread = TwitterThread(
                tweets=[ThreadTweet("🧵 Test", 1, 0)],
                hook_tweet=ThreadTweet("🧵 Test", 1, 0),
                total_tweets=1,
                estimated_reading_time=30,
                reasoning="Test",
                content_type=ContentType.EDUCATIONAL
            )
            mock_generator.generate_thread = AsyncMock(return_value=mock_thread)
            mock_generator_class.return_value = mock_generator
            
            result = await generate_thread(
                transcript=mock_transcript,
                video_title="Test Video",
                provider="gemini"
            )
            
            assert isinstance(result, TwitterThread)
            mock_generator_class.assert_called_once_with("gemini")
            mock_generator.generate_thread.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])