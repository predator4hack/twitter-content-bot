"""
Twitter thread generation service.

This module provides functionality to generate engaging Twitter threads
from video transcripts using LLM analysis and content-type specific strategies.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
import asyncio

try:
    from .llm_analyzer import (
        BaseLLMAnalyzer, LLMAnalyzerFactory, ContentType, 
        ThreadTweet, TwitterThread
    )
    from ..transcription.base import TranscriptionResult
    from ..core.config import config
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from analyzer.llm_analyzer import (
        BaseLLMAnalyzer, LLMAnalyzerFactory, ContentType,
        ThreadTweet, TwitterThread
    )
    from transcription.base import TranscriptionResult
    from core.config import config

logger = logging.getLogger(__name__)


class TwitterThreadGenerator:
    """
    Generates engaging Twitter threads from video transcripts.
    
    Uses LLM analysis to create content-type aware threads that explain
    video content in simple, digestible chunks optimized for Twitter engagement.
    """
    
    def __init__(self, llm_provider: Optional[str] = None):
        """
        Initialize the thread generator.
        
        Args:
            llm_provider: LLM provider to use ("gemini" or "groq")
        """
        self.llm_analyzer = LLMAnalyzerFactory.create_analyzer(llm_provider)
        self.provider = llm_provider or config.DEFAULT_LLM_PROVIDER
        
        # Thread generation constraints based on requirements
        self.MAX_TWEETS = 6
        self.MIN_TWEETS = 3
        self.MAX_TWEET_LENGTH = 280
        self.TARGET_TWEET_LENGTH = 250  # Leave room for hashtags/mentions
        
        logger.info(f"Initialized TwitterThreadGenerator with provider: {self.provider}")
    
    async def generate_thread(
        self,
        transcript: TranscriptionResult,
        video_title: Optional[str] = None,
        video_url: Optional[str] = None,
        target_length: int = 5,
        tone: str = "educational",
        target_audience: str = "general"
    ) -> TwitterThread:
        """
        Generate a Twitter thread from a video transcript.
        
        Args:
            transcript: The video transcript to analyze
            video_title: Optional video title for context
            video_url: Optional video URL to include in final tweet
            target_length: Target number of tweets (3-6)
            tone: Thread tone ("educational", "casual", "professional")
            target_audience: Target audience ("general", "technical", "beginners")
            
        Returns:
            TwitterThread object with generated tweets
        """
        start_time = time.time()
        
        # Validate inputs
        target_length = max(self.MIN_TWEETS, min(self.MAX_TWEETS, target_length))
        
        try:
            logger.info(f"Generating Twitter thread (target_length={target_length}, tone={tone})")
            
            # Create thread generation prompt
            prompt = self._create_thread_prompt(
                transcript, video_title, video_url, target_length, tone, target_audience
            )
            
            # Generate thread using the LLM
            thread_data = await self._generate_with_llm(prompt)
            
            # Create TwitterThread object
            thread = self._create_twitter_thread(thread_data, transcript, video_url)
            
            generation_time = time.time() - start_time
            logger.info(f"Thread generation completed in {generation_time:.2f}s")
            
            return thread
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Thread generation failed after {generation_time:.2f}s: {e}")
            raise
    
    def _create_thread_prompt(
        self,
        transcript: TranscriptionResult,
        video_title: Optional[str],
        video_url: Optional[str],
        target_length: int,
        tone: str,
        target_audience: str
    ) -> str:
        """Create the LLM prompt for thread generation."""
        
        # Format transcript for analysis
        formatted_transcript = self._format_transcript_for_thread(transcript)
        
        # Content type detection for strategy
        content_strategy = self._get_content_strategy_hint(transcript.text, video_title)
        
        prompt = f"""
Generate an engaging Twitter thread from this video transcript. You have full creative control to determine the optimal thread structure.

Video Information:
- Title: {video_title or "Not provided"}
- Duration: {transcript.duration/60:.1f} minutes
- Language: {transcript.language}
- Target Audience: General/Beginners (explain complex topics simply)

CRITICAL REQUIREMENTS:
1. Decide the optimal thread length between 3-6 tweets based on content depth
2. Choose the best tone (educational, casual, professional) that fits the content
3. Each tweet must be under 280 characters
4. First tweet MUST be a compelling hook that makes people want to read more
5. Last tweet MUST include the video link: {video_url or '[VIDEO_LINK]'}
6. Explain content in simple terms (10th grade reading level or lower)
7. Use engaging, conversational language
8. Include relevant hashtags (1-2 per tweet max)

CREATIVE FREEDOM:
- Adapt the thread length to match content complexity (3-6 tweets)
- Choose the tone that best serves the content and audience
- Structure tweets for maximum engagement and comprehension
- Focus on making complex ideas accessible to general audiences

Content Strategy: {content_strategy}

Transcript:
{formatted_transcript}

CRITICAL: Return ONLY valid JSON. Do not include any text before or after the JSON. Use this EXACT format:

{{
  "content_type": "educational",
  "reasoning": "Brief explanation of thread structure and approach",
  "estimated_reading_time": 60,
  "tweets": [
    {{
      "tweet_number": 1,
      "content": "🧵 This video reveals the surprising truth about [topic] that most people get wrong. Here's what you need to know:",
      "hashtags": ["#topic", "#learning"],
      "mentions": []
    }},
    {{
      "tweet_number": 2,
      "content": "First key point explained simply...",
      "hashtags": ["#insight"],
      "mentions": []
    }}
  ]
}}

Important Guidelines:
- Keep tweets conversational and engaging
- Use thread numbering (🧵, 2/, 3/, etc.) 
- Include action words and emotional triggers
- Make each tweet valuable on its own
- End with clear call-to-action and video link
- Valid content_type: educational, entertainment, interview, tutorial, news, review, vlog, gaming, unknown
"""
        return prompt
    
    def _format_transcript_for_thread(self, transcript: TranscriptionResult) -> str:
        """Format transcript text for thread generation."""
        # For thread generation, we mainly need the text content
        # We can include timing for context but focus on content
        
        if len(transcript.text) > 8000:  # Truncate very long transcripts
            text = transcript.text[:8000] + "..."
        else:
            text = transcript.text
            
        return f"Full transcript: {text}"
    
    def _get_content_strategy_hint(self, transcript_text: str, video_title: Optional[str]) -> str:
        """Provide content strategy hints based on transcript and title analysis."""
        text_lower = transcript_text.lower()
        title_lower = (video_title or "").lower()
        
        # Simple keyword-based content type detection
        if any(word in text_lower for word in ["tutorial", "how to", "step", "guide", "learn"]):
            return "Educational/Tutorial: Focus on step-by-step explanations and key takeaways"
        elif any(word in text_lower for word in ["interview", "conversation", "discuss", "talk"]):
            return "Interview/Podcast: Highlight main insights and surprising revelations"
        elif any(word in text_lower for word in ["review", "opinion", "rating", "recommend"]):
            return "Review: Focus on pros/cons and key recommendations"
        elif any(word in text_lower for word in ["funny", "hilarious", "vlog", "daily", "life"]):
            return "Entertainment/Vlog: Highlight engaging moments and relatable insights"
        elif any(word in text_lower for word in ["news", "breaking", "update", "report"]):
            return "News: Focus on facts, implications, and expert analysis"
        else:
            return "General: Focus on the most interesting and valuable insights"
    
    async def _generate_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Generate thread data using the LLM."""
        try:
            # Use the existing LLM analyzer infrastructure
            if hasattr(self.llm_analyzer, 'model'):
                # For Gemini
                import google.generativeai as genai
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.llm_analyzer.model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.4,  # Slightly more creative for thread generation
                            max_output_tokens=2048,
                            response_mime_type="application/json"
                        )
                    )
                )
                response_text = response.text
                
            else:
                # For Groq
                response = self.llm_analyzer.client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert Twitter content creator specializing in creating engaging threads that simplify complex topics. Always respond with valid JSON."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.4,
                    max_tokens=2048,
                )
                response_text = response.choices[0].message.content or ""
            
            # Clean and parse JSON response
            response_text = self._clean_json_response(response_text)
            thread_data = json.loads(response_text)
            
            logger.info(f"Successfully generated thread with {len(thread_data.get('tweets', []))} tweets")
            return thread_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Raw response: {response_text}")
            
            # Create fallback thread
            return self._create_fallback_thread()
            
        except Exception as e:
            logger.error(f"LLM thread generation failed: {e}")
            return self._create_fallback_thread()
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean the LLM response to extract valid JSON."""
        response_text = response_text.strip()
        
        # Remove markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        # Remove common prefixes
        if "Here is the" in response_text and "JSON" in response_text:
            lines = response_text.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('{'):
                    response_text = '\n'.join(lines[i:])
                    break
        
        return response_text.strip()
    
    def _create_fallback_thread(self) -> Dict[str, Any]:
        """Create a fallback thread when LLM generation fails."""
        return {
            "content_type": "unknown",
            "reasoning": "Fallback thread generated due to LLM failure",
            "estimated_reading_time": 30,
            "tweets": [
                {
                    "tweet_number": 1,
                    "content": "🧵 Just watched an interesting video! Here are the key takeaways:",
                    "hashtags": ["#video", "#insights"],
                    "mentions": []
                },
                {
                    "tweet_number": 2,
                    "content": "The main points covered some fascinating topics that are worth exploring further.",
                    "hashtags": ["#learning"],
                    "mentions": []
                },
                {
                    "tweet_number": 3,
                    "content": "Check out the full video for more details: [VIDEO_LINK]",
                    "hashtags": ["#watch"],
                    "mentions": []
                }
            ]
        }
    
    def _create_twitter_thread(
        self, 
        thread_data: Dict[str, Any], 
        transcript: TranscriptionResult,
        video_url: Optional[str]
    ) -> TwitterThread:
        """Create a TwitterThread object from the LLM response data."""
        
        tweets = []
        for tweet_data in thread_data.get("tweets", []):
            # Replace video link placeholder if present
            content = tweet_data["content"]
            if "[VIDEO_LINK]" in content and video_url:
                content = content.replace("[VIDEO_LINK]", video_url)
            
            # Ensure tweet is within character limit
            if len(content) > self.MAX_TWEET_LENGTH:
                # Truncate and add ellipsis
                content = content[:self.MAX_TWEET_LENGTH-3] + "..."
                logger.warning(f"Tweet {tweet_data['tweet_number']} truncated to fit character limit")
            
            tweet = ThreadTweet(
                content=content,
                tweet_number=tweet_data["tweet_number"],
                character_count=len(content),
                hashtags=tweet_data.get("hashtags", []),
                mentions=tweet_data.get("mentions", [])
            )
            tweets.append(tweet)
        
        # Detect content type
        try:
            content_type = ContentType(thread_data.get("content_type", "unknown"))
        except ValueError:
            content_type = ContentType.UNKNOWN
        
        thread = TwitterThread(
            tweets=tweets,
            hook_tweet=tweets[0] if tweets else None,
            total_tweets=len(tweets),
            estimated_reading_time=thread_data.get("estimated_reading_time", 60),
            reasoning=thread_data.get("reasoning", "Thread generated from video transcript"),
            content_type=content_type,
            video_url=video_url
        )
        
        return thread
    
    def validate_thread(self, thread: TwitterThread) -> List[str]:
        """
        Validate a generated thread and return any issues found.
        
        Args:
            thread: The TwitterThread to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []
        
        # Check thread length constraints
        if thread.total_tweets < self.MIN_TWEETS:
            issues.append(f"Thread too short: {thread.total_tweets} tweets (minimum {self.MIN_TWEETS})")
        if thread.total_tweets > self.MAX_TWEETS:
            issues.append(f"Thread too long: {thread.total_tweets} tweets (maximum {self.MAX_TWEETS})")
        
        # Check individual tweets
        for tweet in thread.tweets:
            if len(tweet.content) > self.MAX_TWEET_LENGTH:
                issues.append(f"Tweet {tweet.tweet_number} too long: {len(tweet.content)} characters")
            if len(tweet.content.strip()) == 0:
                issues.append(f"Tweet {tweet.tweet_number} is empty")
        
        # Check hook tweet
        if not thread.hook_tweet:
            issues.append("Missing hook tweet")
        elif not any(indicator in thread.hook_tweet.content for indicator in ["🧵", "thread", "Thread"]):
            issues.append("Hook tweet should indicate it's a thread")
        
        # Check if last tweet has video link (if video_url provided)
        if thread.video_url and thread.tweets:
            last_tweet = thread.tweets[-1]
            if thread.video_url not in last_tweet.content and "video" not in last_tweet.content.lower():
                issues.append("Last tweet should reference the video")
        
        return issues


# Convenience function for direct usage
async def generate_thread(
    transcript: TranscriptionResult,
    video_title: Optional[str] = None,
    video_url: Optional[str] = None,
    provider: Optional[str] = None,
    target_length: int = 5,
    tone: str = "educational"
) -> TwitterThread:
    """
    Convenience function to generate a Twitter thread.
    
    Args:
        transcript: Video transcript to convert to thread
        video_title: Optional video title for context
        video_url: Optional video URL to include
        provider: LLM provider to use
        target_length: Target number of tweets
        tone: Thread tone
        
    Returns:
        Generated TwitterThread
    """
    generator = TwitterThreadGenerator(provider)
    return await generator.generate_thread(
        transcript, video_title, video_url, target_length, tone
    )