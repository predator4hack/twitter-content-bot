# Twitter Thread Generation Feature - Requirements & Acceptance Criteria

## Overview

Extend the existing YouTube-to-Twitter clip extraction application with an intelligent Twitter thread generation feature. This feature will analyze video transcripts and generate engaging, educational Twitter threads that explain video content in simple, digestible chunks.

## Current System Architecture

The application currently processes YouTube videos through this pipeline:

1. **Video Download**: `yt-dlp` downloads video and extracts metadata/thumbnails
2. **Transcription**: Whisper (`faster-whisper`) converts speech to text with timestamps
3. **LLM Analysis**: Gemini/Groq analyzes transcripts to identify engaging clips
4. **Clip Extraction**: FFmpeg extracts video segments based on LLM recommendations
5. **Twitter Optimization**: Clips are optimized for Twitter requirements (2:20 max, 512MB)

## Important Considerations

-   The thread should not contain more than 6 tweets
-   The first tweet should be a hook tweet, something that would make the audience open the tweet
-   The last tweet should have the link to the video

## Feature Requirements

### FR-1: Twitter Thread Generation

**Description**: Generate structured Twitter threads from video transcripts that explain content in an accessible, engaging manner.

**Input**:

-   Video transcript (from existing Whisper transcription)
-   Video metadata (title, duration, content type)
-   User preferences (thread length, tone, target audience)

**Output**:

-   Structured Twitter thread (array of tweets)
-   Thread metadata (total tweets, estimated reading time)
-   Reasoning for thread structure

### FR-2: Content Simplification

**Description**: Transform complex video content into simple, understandable explanations suitable for Twitter's format and audience.

**Requirements**:

-   Break down complex concepts into digestible chunks
-   Use simple language (10th-grade reading level or lower)
-   Maintain technical accuracy while improving accessibility
-   Include relevant hashtags and mentions where appropriate

### FR-3: Thread Structure Optimization

**Description**: Create well-structured threads that maximize engagement and readability.

**Requirements**:

-   Hook tweet that captures attention and summarizes value
-   Logical flow between tweets (chronological, thematic, or progressive)
-   Clear transitions between concepts
-   Engaging conclusion with call-to-action
-   Optimal thread length (3-15 tweets based on content)

### FR-4: Content-Type Aware Generation

**Description**: Generate threads tailored to different video content types.

**Content Type Strategies**:

-   **Educational/Tutorial**: Step-by-step explanations, key takeaways
-   **Interview/Podcast**: Main insights, controversial points, surprising revelations
-   **Review**: Pros/cons, recommendations, key features
-   **Entertainment/Vlog**: Highlights, funny moments, relatable insights
-   **News**: Facts, implications, context, expert opinions

### FR-5: UI Integration

**Description**: Integrate thread generation into the existing Streamlit interface.

**Requirements**:

-   Thread should be automatically rendered after processing the video
-   Display generated threads with each tweet in a separate card. Each card should have a small copy icon, which can be used to copy the content of that tweet
-   Provide option to regenerate a particular tweet, or regenerate the entire tweet
-   Provide a feedback box on what changes should be made when regenerating the tweet/thread

## Technical Implementation

### TI-1: LLM Integration

**Extend existing LLM analyzer** (`src/analyzer/llm_analyzer.py`) to support thread generation:

```python
@dataclass
class ThreadTweet:
    content: str
    tweet_number: int
    character_count: int
    hashtags: List[str]
    mentions: List[str]

@dataclass
class TwitterThread:
    tweets: List[ThreadTweet]
    hook_tweet: ThreadTweet
    total_tweets: int
    estimated_reading_time: int
    reasoning: str
    content_type: ContentType
```

### TI-2: Thread Generation Service

**Create new module** `src/analyzer/thread_generator.py`:

```python
class TwitterThreadGenerator:
    async def generate_thread(
        self,
        transcript: TranscriptionResult,
        target_length: int = 10,
        tone: str = "educational",
        target_audience: str = "general"
    ) -> TwitterThread
```

### TI-3: Content Strategy Extension

**Extend** `src/analyzer/content_strategy.py` to include thread-specific strategies for each content type.

### TI-4: UI Components

**Extend** `src/ui/streamlit_app.py` and `src/ui/components.py` to include:

-   Thread generation settings panel
-   Thread preview component with individual tweet display
-   Edit functionality for tweaking individual tweets
-   Export options (copy to clipboard, download as text file)

## Acceptance Criteria

### AC-1: Thread Quality

-   [ ] Generated threads maintain factual accuracy from source video
-   [ ] Individual tweets stay within 280-character limit (including hashtags/mentions)
-   [ ] Thread maintains logical flow and coherence
-   [ ] Language complexity appropriate for general Twitter audience
-   [ ] Hook tweet effectively summarizes thread value proposition

### AC-2: Content Coverage

-   [ ] Thread covers main points from video transcript
-   [ ] Important concepts are simplified without losing meaning
-   [ ] Thread length appropriate for content depth (3-15 tweets)
-   [ ] No critical information is omitted or misrepresented

### AC-3: User Experience

-   [ ] Thread generation completes within 30 seconds for typical videos
-   [ ] Users can preview entire thread before export
-   [ ] Individual tweets can be edited in-place
-   [ ] Multiple export formats available (plain text, formatted)
-   [ ] Clear error handling and user feedback

### AC-4: Technical Integration

-   [ ] Thread generation integrates seamlessly with existing pipeline
-   [ ] Uses same transcript data as clip extraction feature
-   [ ] Supports both Gemini and Groq LLM providers
-   [ ] Maintains existing code patterns and architecture
-   [ ] Includes comprehensive error handling and logging

### AC-5: Performance

-   [ ] Thread generation does not significantly impact clip extraction performance
-   [ ] LLM API costs remain reasonable (target: <$0.05 per thread)
-   [ ] Memory usage stays within acceptable limits for typical videos
-   [ ] Concurrent thread and clip generation possible

## Success Metrics

### Quantitative Metrics

-   **Generation Speed**: <30 seconds per thread for videos up to 1 hour
-   **Character Efficiency**: 95%+ of tweets use 200-280 characters optimally
-   **User Satisfaction**: Thread quality rated 4+ out of 5 by users
-   **Technical Reliability**: 99%+ success rate for thread generation

### Qualitative Metrics

-   Thread readability and engagement potential
-   Accuracy of content simplification
-   Logical flow and structure quality
-   Integration smoothness with existing workflow

## Implementation Phases

### Phase 1: Core Thread Generation (Week 1)

-   Extend LLM analyzer with thread generation prompts
-   Create basic TwitterThreadGenerator class
-   Implement content-type aware thread strategies
-   Add unit tests for thread generation logic

### Phase 2: UI Integration (Week 2)

-   Add thread generation UI components to Streamlit app
-   Implement thread preview and editing functionality
-   Create export options and formatting
-   Add user settings for thread customization

### Phase 3: Optimization & Polish (Week 3)

-   Performance optimization for concurrent processing
-   Enhanced error handling and user feedback
-   A/B testing different thread generation strategies
-   Documentation and user guides

### Phase 4: Testing & Validation (Week 4)

-   Comprehensive testing with various video types
-   User acceptance testing and feedback incorporation
-   Performance benchmarking and optimization
-   Final bug fixes and deployment preparation

## Technical Considerations

### LLM Prompt Engineering

-   Design prompts that encourage clear, engaging thread structure
-   Include examples of high-quality Twitter threads in prompts
-   Implement fallback strategies for edge cases

### Content Safety

-   Ensure generated threads don't inadvertently misrepresent source content
-   Implement content filtering for sensitive topics
-   Maintain attribution to original video creators

### Scalability

-   Design for potential future features (thread scheduling, multi-platform support)
-   Consider API rate limits and cost optimization
-   Plan for increased user load and concurrent processing

## Dependencies

### Required Libraries (Existing)

-   `google-generativeai` or `groq` - LLM providers
-   `streamlit` - UI framework
-   `faster-whisper` - Transcription (existing)

### New Dependencies (if needed)

-   `tweepy` - Future Twitter API integration (optional)
-   Additional text processing libraries as needed

## Risks & Mitigation

### Risk 1: LLM Output Quality

**Mitigation**: Implement robust prompt engineering, fallback strategies, and user editing capabilities

### Risk 2: Character Limit Management

**Mitigation**: Implement smart text truncation, abbreviation strategies, and user preview/edit functionality

### Risk 3: Content Accuracy

**Mitigation**: Include source verification, user review process, and clear attribution

### Risk 4: Performance Impact

**Mitigation**: Optimize LLM calls, implement caching, and provide asynchronous processing options

## Future Enhancements

-   Multi-language thread generation support
-   Integration with Twitter scheduling tools
-   Thread template system for different content types
-   Analytics integration for thread performance tracking
-   Community-driven thread quality improvement

---

This feature will significantly enhance the application's value proposition by providing users with both short-form video clips and engaging Twitter threads from a single YouTube URL, creating a comprehensive content repurposing solution.
