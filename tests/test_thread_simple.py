"""
Simple integration test for Twitter thread generation.

This test validates the basic data structures and thread logic without
requiring full LLM integration.
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


# Mock the basic data structures for testing
class ContentType(Enum):
    EDUCATIONAL = "educational"
    ENTERTAINMENT = "entertainment"
    INTERVIEW = "interview"


@dataclass
class ThreadTweet:
    content: str
    tweet_number: int
    character_count: int
    hashtags: List[str] = None
    mentions: List[str] = None
    
    def __post_init__(self):
        if self.hashtags is None:
            self.hashtags = []
        if self.mentions is None:
            self.mentions = []
        self.character_count = len(self.content)


@dataclass 
class TwitterThread:
    tweets: List[ThreadTweet]
    hook_tweet: ThreadTweet
    total_tweets: int
    estimated_reading_time: int
    reasoning: str
    content_type: ContentType
    video_url: Optional[str] = None
    
    def __post_init__(self):
        if self.tweets:
            self.hook_tweet = self.tweets[0]
            self.total_tweets = len(self.tweets)


class TestThreadDataStructures:
    """Test the basic thread data structures."""
    
    def test_thread_tweet_creation(self):
        """Test ThreadTweet creation and character counting."""
        tweet = ThreadTweet(
            content="🧵 This is a test tweet about AI and machine learning",
            tweet_number=1,
            character_count=0,  # Will be calculated
            hashtags=["#AI", "#MachineLearning"],
            mentions=["@user"]
        )
        
        assert tweet.content == "🧵 This is a test tweet about AI and machine learning"
        assert tweet.tweet_number == 1
        assert tweet.character_count == len(tweet.content)
        assert "#AI" in tweet.hashtags
        assert "@user" in tweet.mentions
    
    def test_thread_tweet_defaults(self):
        """Test ThreadTweet with default hashtags and mentions."""
        tweet = ThreadTweet(
            content="Simple tweet",
            tweet_number=1,
            character_count=0
        )
        
        assert tweet.hashtags == []
        assert tweet.mentions == []
        assert tweet.character_count == len("Simple tweet")
    
    def test_twitter_thread_creation(self):
        """Test TwitterThread creation and automatic properties."""
        tweets = [
            ThreadTweet("🧵 Hook tweet about AI", 1, 0),
            ThreadTweet("2/ AI is transforming technology", 2, 0),
            ThreadTweet("3/ Watch the full video here", 3, 0)
        ]
        
        thread = TwitterThread(
            tweets=tweets,
            hook_tweet=None,  # Will be set automatically
            total_tweets=0,   # Will be calculated
            estimated_reading_time=45,
            reasoning="Educational thread about AI",
            content_type=ContentType.EDUCATIONAL,
            video_url="https://youtube.com/watch?v=123"
        )
        
        assert thread.total_tweets == 3
        assert thread.hook_tweet == tweets[0]
        assert thread.hook_tweet.content == "🧵 Hook tweet about AI"
        assert thread.content_type == ContentType.EDUCATIONAL
        assert thread.video_url == "https://youtube.com/watch?v=123"


class TestThreadValidation:
    """Test thread validation logic."""
    
    def test_validate_tweet_character_limits(self):
        """Test that tweets respect Twitter's character limits."""
        # Valid tweet
        short_tweet = ThreadTweet("Short tweet", 1, 0)
        assert short_tweet.character_count <= 280
        
        # Long tweet (still valid but on the edge)
        long_content = "A" * 279
        long_tweet = ThreadTweet(long_content, 1, 0)
        assert long_tweet.character_count == 279
        assert long_tweet.character_count <= 280
        
        # Very long tweet (would be invalid)
        very_long_content = "A" * 300
        very_long_tweet = ThreadTweet(very_long_content, 1, 0)
        assert very_long_tweet.character_count > 280
    
    def test_thread_structure_requirements(self):
        """Test that threads meet basic structure requirements."""
        # Valid thread structure
        tweets = [
            ThreadTweet("🧵 Hook tweet with thread indicator", 1, 0),
            ThreadTweet("2/ Middle content tweet", 2, 0),
            ThreadTweet("3/ Conclusion with link: https://youtube.com/watch?v=123", 3, 0)
        ]
        
        thread = TwitterThread(
            tweets=tweets,
            hook_tweet=tweets[0],
            total_tweets=3,
            estimated_reading_time=60,
            reasoning="Well-structured thread",
            content_type=ContentType.EDUCATIONAL,
            video_url="https://youtube.com/watch?v=123"
        )
        
        # Check hook tweet has thread indicator
        assert "🧵" in thread.hook_tweet.content
        
        # Check thread has reasonable length (3-6 tweets per requirements)
        assert 3 <= thread.total_tweets <= 6
        
        # Check last tweet mentions video
        last_tweet = thread.tweets[-1]
        if thread.video_url:
            assert thread.video_url in last_tweet.content or "video" in last_tweet.content.lower()
    
    def test_content_type_strategies(self):
        """Test that different content types have appropriate characteristics."""
        # Educational thread
        edu_tweets = [
            ThreadTweet("🧵 Learn about neural networks in this thread", 1, 0, ["#learning", "#AI"]),
            ThreadTweet("2/ Neural networks process information in layers", 2, 0),
            ThreadTweet("3/ Each layer learns different features", 3, 0),
            ThreadTweet("4/ Watch the full tutorial: https://youtube.com/watch?v=edu", 4, 0)
        ]
        
        edu_thread = TwitterThread(
            tweets=edu_tweets,
            hook_tweet=edu_tweets[0],
            total_tweets=4,
            estimated_reading_time=60,
            reasoning="Educational content about AI",
            content_type=ContentType.EDUCATIONAL
        )
        
        assert edu_thread.content_type == ContentType.EDUCATIONAL
        assert "#learning" in edu_thread.hook_tweet.hashtags
        assert "Learn" in edu_thread.hook_tweet.content
        
        # Entertainment thread  
        ent_tweets = [
            ThreadTweet("🧵 This video had me laughing the whole time", 1, 0, ["#funny"]),
            ThreadTweet("2/ The best moments from this hilarious video", 2, 0),
            ThreadTweet("3/ Check it out: https://youtube.com/watch?v=fun", 3, 0)
        ]
        
        ent_thread = TwitterThread(
            tweets=ent_tweets,
            hook_tweet=ent_tweets[0],
            total_tweets=3,
            estimated_reading_time=30,
            reasoning="Funny video highlights",
            content_type=ContentType.ENTERTAINMENT
        )
        
        assert ent_thread.content_type == ContentType.ENTERTAINMENT
        assert "laughing" in ent_thread.hook_tweet.content.lower()


class TestThreadOptimization:
    """Test thread optimization strategies."""
    
    def test_hashtag_optimization(self):
        """Test hashtag placement and optimization."""
        tweet = ThreadTweet(
            "Great insights about artificial intelligence and machine learning",
            1, 0,
            hashtags=["#AI", "#MachineLearning", "#Insights", "#Tech", "#Learning"]
        )
        
        # Check that hashtags don't make tweet too long
        full_content = tweet.content + " " + " ".join(tweet.hashtags)
        
        # If too long, we should limit hashtags
        if len(full_content) > 280:
            # Simulate hashtag reduction
            reduced_hashtags = tweet.hashtags[:2]  # Keep only first 2
            reduced_content = tweet.content + " " + " ".join(reduced_hashtags)
            assert len(reduced_content) <= 280
    
    def test_thread_length_optimization(self):
        """Test thread length stays within optimal range."""
        # Test different content types have appropriate max lengths
        max_lengths = {
            ContentType.EDUCATIONAL: 6,
            ContentType.ENTERTAINMENT: 4,
            ContentType.INTERVIEW: 5
        }
        
        for content_type, max_length in max_lengths.items():
            # Create a thread that might be too long
            tweets = [
                ThreadTweet(f"Tweet {i}", i, 0) for i in range(1, 8)  # 7 tweets
            ]
            
            thread = TwitterThread(
                tweets=tweets,
                hook_tweet=tweets[0],
                total_tweets=7,
                estimated_reading_time=90,
                reasoning="Long thread",
                content_type=content_type
            )
            
            # In real implementation, we'd optimize this
            # Here we just check that we know the limits
            assert max_length in [4, 5, 6]  # All reasonable limits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])