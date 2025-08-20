"""
Content strategy and optimization for Twitter clips.

This module provides strategy-based content analysis and optimization
for different types of video content to maximize engagement on Twitter.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Any

from .llm_analyzer import AnalysisResult, ClipRecommendation, ContentType, HookStrength

logger = logging.getLogger(__name__)


class TwitterStrategy(Enum):
    """Different Twitter content strategies."""
    VIRAL_MOMENTS = "viral_moments"          # Focus on highly shareable moments
    EDUCATIONAL_VALUE = "educational_value"  # Focus on learning content
    ENTERTAINMENT = "entertainment"          # Focus on fun/entertaining content
    THOUGHT_LEADERSHIP = "thought_leadership" # Focus on insights and expertise
    BEHIND_SCENES = "behind_scenes"          # Focus on personal/authentic moments
    CONTROVERSY = "controversy"              # Focus on debate-worthy content


@dataclass
class StrategyWeights:
    """Weights for different content characteristics based on strategy."""
    hook_strength: float = 1.0
    confidence: float = 1.0
    keyword_relevance: float = 1.0
    sentiment_bonus: float = 1.0
    duration_penalty: float = 1.0
    
    def normalize(self) -> 'StrategyWeights':
        """Normalize all weights to sum to 1.0."""
        total = (self.hook_strength + self.confidence + self.keyword_relevance + 
                self.sentiment_bonus + self.duration_penalty)
        if total == 0:
            return self
        
        return StrategyWeights(
            hook_strength=self.hook_strength / total,
            confidence=self.confidence / total,
            keyword_relevance=self.keyword_relevance / total,
            sentiment_bonus=self.sentiment_bonus / total,
            duration_penalty=self.duration_penalty / total
        )


class ContentStrategy:
    """
    Content strategy analyzer for optimizing clip selection based on Twitter best practices.
    """
    
    # Strategy-specific weights for scoring clips
    STRATEGY_WEIGHTS = {
        TwitterStrategy.VIRAL_MOMENTS: StrategyWeights(
            hook_strength=2.0,
            confidence=1.5,
            keyword_relevance=1.0,
            sentiment_bonus=1.5,
            duration_penalty=0.8
        ),
        TwitterStrategy.EDUCATIONAL_VALUE: StrategyWeights(
            hook_strength=1.0,
            confidence=2.0,
            keyword_relevance=2.0,
            sentiment_bonus=0.8,
            duration_penalty=1.2
        ),
        TwitterStrategy.ENTERTAINMENT: StrategyWeights(
            hook_strength=2.0,
            confidence=1.0,
            keyword_relevance=0.8,
            sentiment_bonus=2.0,
            duration_penalty=0.5
        ),
        TwitterStrategy.THOUGHT_LEADERSHIP: StrategyWeights(
            hook_strength=1.2,
            confidence=2.0,
            keyword_relevance=2.5,
            sentiment_bonus=1.0,
            duration_penalty=1.5
        ),
        TwitterStrategy.BEHIND_SCENES: StrategyWeights(
            hook_strength=1.5,
            confidence=1.0,
            keyword_relevance=1.0,
            sentiment_bonus=1.8,
            duration_penalty=1.0
        ),
        TwitterStrategy.CONTROVERSY: StrategyWeights(
            hook_strength=2.5,
            confidence=1.5,
            keyword_relevance=1.5,
            sentiment_bonus=0.5,
            duration_penalty=0.8
        )
    }
    
    # High-value keywords for different content types
    CONTENT_KEYWORDS = {
        ContentType.EDUCATIONAL: {
            "high_value": ["learn", "teach", "explain", "understand", "knowledge", "skill", "tip", "trick"],
            "medium_value": ["example", "practice", "method", "technique", "process", "step"],
            "viral_triggers": ["secret", "mistake", "wrong", "truth", "revealed", "exposed"]
        },
        ContentType.ENTERTAINMENT: {
            "high_value": ["funny", "hilarious", "amazing", "incredible", "unbelievable", "epic"],
            "medium_value": ["cool", "awesome", "great", "nice", "good"],
            "viral_triggers": ["fail", "win", "crazy", "insane", "shocking", "viral"]
        },
        ContentType.INTERVIEW: {
            "high_value": ["story", "experience", "journey", "reveal", "confession", "insight"],
            "medium_value": ["question", "answer", "discuss", "talk", "share"],
            "viral_triggers": ["controversial", "exclusive", "first time", "never told", "secret"]
        },
        ContentType.TUTORIAL: {
            "high_value": ["how to", "tutorial", "guide", "step by step", "easy", "simple"],
            "medium_value": ["create", "make", "build", "setup", "install"],
            "viral_triggers": ["hack", "shortcut", "faster", "better way", "pro tip"]
        }
    }
    
    @staticmethod
    def detect_strategy(analysis: AnalysisResult) -> TwitterStrategy:
        """
        Detect the most appropriate Twitter strategy based on content analysis.
        
        Args:
            analysis: LLM analysis result
            
        Returns:
            Recommended Twitter strategy
        """
        content_type = analysis.content_type
        recommendations = analysis.recommendations
        
        # Count characteristics
        high_hook_count = sum(1 for r in recommendations if r.hook_strength == HookStrength.HIGH)
        avg_confidence = sum(r.confidence for r in recommendations) / len(recommendations) if recommendations else 0
        
        # Extract all keywords
        all_keywords = []
        for rec in recommendations:
            all_keywords.extend(rec.keywords)
        
        # Strategy detection logic
        if content_type == ContentType.EDUCATIONAL and avg_confidence > 75:
            return TwitterStrategy.EDUCATIONAL_VALUE
        elif content_type == ContentType.ENTERTAINMENT or high_hook_count >= 2:
            return TwitterStrategy.VIRAL_MOMENTS
        elif content_type == ContentType.INTERVIEW and "exclusive" in " ".join(all_keywords).lower():
            return TwitterStrategy.THOUGHT_LEADERSHIP
        elif any(keyword in " ".join(all_keywords).lower() for keyword in ["behind", "personal", "story"]):
            return TwitterStrategy.BEHIND_SCENES
        elif any(keyword in " ".join(all_keywords).lower() for keyword in ["controversial", "debate", "disagree"]):
            return TwitterStrategy.CONTROVERSY
        else:
            return TwitterStrategy.VIRAL_MOMENTS  # Default strategy
    
    @classmethod
    def score_clip(
        cls,
        clip: ClipRecommendation,
        content_type: ContentType,
        strategy: TwitterStrategy,
        target_duration: int = 60
    ) -> float:
        """
        Score a clip based on strategy and content type.
        
        Args:
            clip: Clip recommendation to score
            content_type: Type of content
            strategy: Twitter strategy to apply
            target_duration: Target clip duration in seconds
            
        Returns:
            Score from 0-100
        """
        weights = cls.STRATEGY_WEIGHTS[strategy].normalize()
        
        # Base scores
        hook_score = cls._score_hook_strength(clip.hook_strength)
        confidence_score = clip.confidence
        keyword_score = cls._score_keywords(clip.keywords, content_type)
        sentiment_score = cls._score_sentiment(clip.sentiment, strategy)
        duration_score = cls._score_duration(clip.duration_seconds, target_duration)
        
        # Weighted final score
        final_score = (
            hook_score * weights.hook_strength +
            confidence_score * weights.confidence +
            keyword_score * weights.keyword_relevance +
            sentiment_score * weights.sentiment_bonus +
            duration_score * weights.duration_penalty
        )
        
        return min(100, max(0, final_score))
    
    @staticmethod
    def _score_hook_strength(hook_strength: HookStrength) -> float:
        """Score based on hook strength."""
        if hook_strength == HookStrength.HIGH:
            return 100
        elif hook_strength == HookStrength.MEDIUM:
            return 70
        else:
            return 40
    
    @classmethod
    def _score_keywords(cls, keywords: List[str], content_type: ContentType) -> float:
        """Score based on keyword relevance."""
        if not keywords:
            return 50  # Neutral score
        
        if content_type not in cls.CONTENT_KEYWORDS:
            return 50
        
        content_keywords = cls.CONTENT_KEYWORDS[content_type]
        keywords_lower = [k.lower() for k in keywords]
        
        score = 0
        total_possible = len(keywords) * 100
        
        for keyword in keywords_lower:
            keyword_text = " ".join(keywords_lower)
            
            # Check for high-value keywords
            for high_val in content_keywords["high_value"]:
                if high_val in keyword_text:
                    score += 100
                    break
            else:
                # Check for medium-value keywords
                for med_val in content_keywords["medium_value"]:
                    if med_val in keyword_text:
                        score += 70
                        break
                else:
                    # Check for viral triggers
                    for viral in content_keywords["viral_triggers"]:
                        if viral in keyword_text:
                            score += 120  # Bonus for viral triggers
                            break
                    else:
                        score += 30  # Base score for any keyword
        
        return min(100, score / len(keywords)) if keywords else 50
    
    @staticmethod
    def _score_sentiment(sentiment: str, strategy: TwitterStrategy) -> float:
        """Score based on sentiment and strategy alignment."""
        sentiment_scores = {
            "positive": 85,
            "neutral": 60,
            "negative": 40
        }
        
        base_score = sentiment_scores.get(sentiment.lower(), 60)
        
        # Strategy-specific adjustments
        if strategy == TwitterStrategy.CONTROVERSY and sentiment.lower() == "negative":
            return min(100, base_score + 30)  # Controversial content benefits from negative sentiment
        elif strategy == TwitterStrategy.VIRAL_MOMENTS and sentiment.lower() == "positive":
            return min(100, base_score + 20)  # Viral content benefits from positive sentiment
        
        return base_score
    
    @staticmethod
    def _score_duration(duration: float, target_duration: int) -> float:
        """Score based on how close the duration is to target."""
        if duration <= 0:
            return 0
        
        # Optimal range: target ± 20 seconds
        optimal_min = max(10, target_duration - 20)
        optimal_max = target_duration + 20
        
        if optimal_min <= duration <= optimal_max:
            return 100
        elif duration < optimal_min:
            # Too short - penalty increases as it gets shorter
            return max(30, 100 - (optimal_min - duration) * 2)
        else:
            # Too long - penalty increases as it gets longer
            return max(20, 100 - (duration - optimal_max) * 1.5)
    
    @classmethod
    def optimize_recommendations(
        cls,
        analysis: AnalysisResult,
        strategy: Optional[TwitterStrategy] = None,
        max_clips: int = 3,
        target_duration: int = 60
    ) -> List[ClipRecommendation]:
        """
        Optimize clip recommendations based on strategy.
        
        Args:
            analysis: Original LLM analysis result
            strategy: Twitter strategy to apply (auto-detected if None)
            max_clips: Maximum number of clips to return
            target_duration: Target duration for clips
            
        Returns:
            Optimized list of clip recommendations
        """
        if not analysis.recommendations:
            return []
        
        # Auto-detect strategy if not provided
        if strategy is None:
            strategy = cls.detect_strategy(analysis)
        
        logger.info(f"Optimizing recommendations with strategy: {strategy.value}")
        
        # Score all recommendations
        scored_clips = []
        for clip in analysis.recommendations:
            score = cls.score_clip(clip, analysis.content_type, strategy, target_duration)
            scored_clips.append((score, clip))
        
        # Sort by score (highest first) and take top max_clips
        scored_clips.sort(key=lambda x: x[0], reverse=True)
        optimized_clips = [clip for score, clip in scored_clips[:max_clips]]
        
        logger.info(f"Selected {len(optimized_clips)} clips from {len(analysis.recommendations)} candidates")
        
        return optimized_clips
    
    @staticmethod
    def generate_twitter_text(clip: ClipRecommendation, max_length: int = 280) -> str:
        """
        Generate optimized Twitter text for a clip.
        
        Args:
            clip: Clip recommendation
            max_length: Maximum character length for tweet
            
        Returns:
            Optimized tweet text
        """
        # Start with the reasoning as base content
        base_text = clip.reasoning
        
        # Add relevant hashtags based on keywords
        hashtags = []
        for keyword in clip.keywords[:3]:  # Limit to 3 keywords
            # Convert to hashtag format
            hashtag = "#" + keyword.replace(" ", "").replace("-", "").capitalize()
            if len(hashtag) <= 20:  # Reasonable hashtag length
                hashtags.append(hashtag)
        
        # Add sentiment-based elements
        if clip.sentiment == "positive":
            prefixes = ["🔥 ", "✨ ", "💯 "]
        elif clip.sentiment == "negative":
            prefixes = ["⚠️ ", "🚨 ", "📢 "]
        else:
            prefixes = ["🎯 ", "📺 ", "🎬 "]
        
        # Construct tweet
        prefix = prefixes[0] if prefixes else ""
        hashtag_text = " ".join(hashtags) if hashtags else ""
        
        # Calculate available space
        overhead = len(prefix) + len(hashtag_text) + 3  # 3 for spaces
        available_length = max_length - overhead
        
        # Truncate base text if necessary
        if len(base_text) > available_length:
            base_text = base_text[:available_length - 3] + "..."
        
        # Combine elements
        tweet_parts = [prefix + base_text]
        if hashtag_text:
            tweet_parts.append(hashtag_text)
        
        return " ".join(tweet_parts)
    
    @classmethod
    def analyze_competition(cls, clips: List[ClipRecommendation]) -> Dict[str, Any]:
        """
        Analyze the competitive landscape of selected clips.
        
        Args:
            clips: List of selected clips
            
        Returns:
            Competition analysis report
        """
        if not clips:
            return {"total_clips": 0, "recommendations": []}
        
        total_duration = sum(clip.duration_seconds for clip in clips)
        avg_confidence = sum(clip.confidence for clip in clips) / len(clips)
        hook_distribution = {
            "high": sum(1 for c in clips if c.hook_strength == HookStrength.HIGH),
            "medium": sum(1 for c in clips if c.hook_strength == HookStrength.MEDIUM),
            "low": sum(1 for c in clips if c.hook_strength == HookStrength.LOW)
        }
        
        # Generate recommendations
        recommendations = []
        
        if avg_confidence < 70:
            recommendations.append("Consider finding clips with higher confidence scores")
        
        if hook_distribution["high"] == 0:
            recommendations.append("Include at least one high-hook-strength clip for better engagement")
        
        if total_duration > 300:  # 5 minutes total
            recommendations.append("Consider shorter clips for better Twitter engagement")
        
        if len(set(clip.sentiment for clip in clips)) == 1:
            recommendations.append("Mix different sentiment types for broader appeal")
        
        return {
            "total_clips": len(clips),
            "total_duration": total_duration,
            "avg_confidence": avg_confidence,
            "hook_distribution": hook_distribution,
            "sentiment_variety": len(set(clip.sentiment for clip in clips)),
            "recommendations": recommendations
        }
