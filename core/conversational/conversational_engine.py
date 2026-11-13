# Copyright 2026 Tobias Østen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Conversational Engine — follow-up questions, curiosity, and topic tracking.

Makes Condrox AI feel alive and human by:
1. Tracking what the user is talking about (topics, interests, emotions)
2. Generating contextual follow-up questions that show genuine curiosity
3. Remembering user preferences and facts across conversation turns
4. Proactively asking questions based on what the AI "feels" curious about
5. Keeping conversations flowing naturally with varied, non-repetitive responses

The engine is per-session and ephemeral — no persistent storage, no
cross-user contamination. Each session gets its own ConversationProfile.

Architecture:
- TopicTracker: Extracts topics from user messages using keyword extraction
- FollowUpGenerator: Generates contextual questions based on intent, emotion,
  topics, and conversation history
- CuriosityEngine: Decides when and what to ask, based on conversation flow
- ConversationProfile: Per-session state tracking topics, turn count, emotions

Mathematical Foundation:
- Topic extraction: O(n) for n words in input (keyword frequency counting)
- Follow-up generation: O(1) template selection from scored candidates
- Curiosity scoring: O(t) for t tracked topics

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import re
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ─── Stop words for topic extraction ──────────────────────────────────────────
_TOPIC_STOP_WORDS: frozenset[str] = frozenset({
    "the", "is", "a", "an", "what", "who", "how", "why", "where", "when",
    "of", "to", "in", "for", "and", "or", "are", "was", "were", "be",
    "do", "does", "did", "will", "would", "could", "should", "can",
    "this", "that", "these", "those", "it", "its", "on", "at", "by",
    "with", "from", "as", "into", "about", "your", "you", "me", "my",
    "we", "our", "i", "please", "tell", "explain", "describe", "like",
    "just", "really", "very", "much", "some", "any", "all", "every",
    "have", "has", "had", "been", "being", "am", "if", "then", "so",
    "but", "not", "no", "yes", "here", "there", "now", "also", "more",
    "most", "than", "only", "out", "up", "down", "over", "under",
    "im", "ive", "id", "dont", "cant", "wont", "thats", "whats",
    "name", "thing", "stuff", "good", "bad", "great", "okay", "ok",
    "well", "get", "got", "make", "made", "go", "going", "one", "two",
    # Emotional filler words — not topics
    "feeling", "feel", "felt", "feelings", "sad", "happy", "angry",
    "anxious", "worried", "scared", "afraid", "tired", "stressed",
    "today", "yesterday", "tomorrow", "now", "right", "still",
    "want", "need", "wish", "hope", "trying", "maybe", "kind",
    "sort", "lot", "bit", "way", "thing", "things", "stuff",
    "talk", "talking", "say", "said", "know", "think", "thought",
    "let", "lets", "sure", "yeah", "yep", "nope", "nah",
    "thanks", "thank", "cool", "nice", "awesome", "fine",
    "help", "helping", "helped", "want", "wanted",
})

# ─── Topic keywords → domain mapping ──────────────────────────────────────────
_TOPIC_DOMAIN_MAP: Dict[str, str] = {
    # Technology
    "code": "programming", "programming": "programming", "python": "programming",
    "javascript": "programming", "java": "programming", "html": "programming",
    "css": "programming", "react": "programming", "api": "programming",
    "function": "programming", "algorithm": "programming", "database": "programming",
    "software": "programming", "developer": "programming", "coding": "programming",
    "app": "programming", "website": "programming", "web": "programming",
    # Science
    "physics": "science", "chemistry": "science", "biology": "science",
    "science": "science", "quantum": "science", "energy": "science",
    "atom": "science", "molecule": "science", "space": "science",
    "universe": "science", "planet": "science", "star": "science",
    "gravity": "science", "light": "science", "electricity": "science",
    # Math
    "math": "mathematics", "mathematics": "mathematics", "calculus": "mathematics",
    "algebra": "mathematics", "geometry": "mathematics", "equation": "mathematics",
    "number": "mathematics", "calculate": "mathematics",
    # Arts & Culture
    "music": "arts", "art": "arts", "painting": "arts", "poem": "arts",
    "poetry": "arts", "book": "arts", "novel": "arts", "movie": "arts",
    "film": "arts", "story": "arts", "writing": "arts", "literature": "arts",
    "culture": "arts", "tradition": "arts", "festival": "arts",
    # Philosophy
    "philosophy": "philosophy", "meaning": "philosophy", "consciousness": "philosophy",
    "existence": "philosophy", "reality": "philosophy", "truth": "philosophy",
    "mind": "philosophy", "soul": "philosophy", "ethics": "philosophy",
    "moral": "philosophy", "justice": "philosophy",
    # Personal / Emotional
    "family": "personal", "friend": "personal", "relationship": "personal",
    "love": "personal", "home": "personal", "work": "personal",
    "job": "personal", "school": "personal", "study": "personal",
    "feeling": "personal", "emotion": "personal", "life": "personal",
    "future": "personal", "dream": "personal", "goal": "personal",
    # Geography
    "country": "geography", "city": "geography", "capital": "geography",
    "norway": "geography", "europe": "geography", "world": "geography",
    "mountain": "geography", "river": "geography", "ocean": "geography",
    "travel": "geography",
    # Health
    "health": "health", "exercise": "health", "food": "health",
    "sleep": "health", "stress": "health", "mental": "health",
    # Business
    "business": "business", "money": "business", "market": "business",
    "company": "business", "startup": "business", "finance": "business",
    "investment": "business", "economy": "business",
}

# ─── Multi-word topic phrases ─────────────────────────────────────────────────
# These are detected as single topics rather than individual words.
# Order matters — longer phrases are checked first.
_MULTI_WORD_TOPICS: tuple[str, ...] = (
    "machine learning", "artificial intelligence", "deep learning",
    "neural networks", "natural language", "computer vision",
    "data science", "software engineering", "web development",
    "mobile development", "game development", "quantum physics",
    "quantum mechanics", "particle physics", "theoretical physics",
    "climate change", "renewable energy", "solar energy", "wind energy",
    "mental health", "physical health", "public health",
    "world war", "cold war", "civil war",
    "music theory", "film theory", "literary theory",
    "moral philosophy", "political philosophy", "philosophy of mind",
    "free will", "consciousness studies",
    "supply chain", "project management", "product management",
    "human resources", "customer service",
    "operating system", "database design", "system architecture",
    "distributed systems", "cloud computing", "edge computing",
    "blockchain technology", "cryptocurrency",
    "space exploration", "mars mission", "moon landing",
    "human rights", "social justice", "gender equality",
)

# Build a lookup set for O(1) multi-word topic detection
_MULTI_WORD_SET: frozenset[str] = frozenset(_MULTI_WORD_TOPICS)

# Domain mapping for multi-word topics
_MULTI_WORD_DOMAIN_MAP: Dict[str, str] = {
    "machine learning": "programming", "artificial intelligence": "programming",
    "deep learning": "programming", "neural networks": "programming",
    "natural language": "programming", "computer vision": "programming",
    "data science": "programming", "software engineering": "programming",
    "web development": "programming", "mobile development": "programming",
    "game development": "programming",
    "operating system": "programming", "database design": "programming",
    "system architecture": "programming", "distributed systems": "programming",
    "cloud computing": "programming", "edge computing": "programming",
    "blockchain technology": "programming", "cryptocurrency": "business",
    "quantum physics": "science", "quantum mechanics": "science",
    "particle physics": "science", "theoretical physics": "science",
    "climate change": "science", "renewable energy": "science",
    "solar energy": "science", "wind energy": "science",
    "space exploration": "science", "mars mission": "science",
    "moon landing": "science",
    "mental health": "health", "physical health": "health",
    "public health": "health",
    "world war": "history", "cold war": "history", "civil war": "history",
    "music theory": "arts", "film theory": "arts", "literary theory": "arts",
    "moral philosophy": "philosophy", "political philosophy": "philosophy",
    "philosophy of mind": "philosophy", "free will": "philosophy",
    "consciousness studies": "philosophy",
    "supply chain": "business", "project management": "business",
    "product management": "business", "human resources": "business",
    "customer service": "business",
    "human rights": "philosophy", "social justice": "philosophy",
    "gender equality": "philosophy",
}


@dataclass(slots=True)
class TrackedTopic:
    """A topic extracted from user input with frequency and recency.

    Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
    """
    word: str
    domain: str
    frequency: int = 1
    first_mentioned: float = field(default_factory=time.time)
    last_mentioned: float = field(default_factory=time.time)
    emotion_when_mentioned: str = "neutral"

    def touch(self, emotion: str = "neutral") -> None:
        """Update recency and frequency for a repeated topic.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self.frequency += 1
        self.last_mentioned = time.time()
        if emotion not in ("neutral", ""):
            self.emotion_when_mentioned = emotion


@dataclass(slots=True)
class ConversationProfile:
    """Per-session conversation state for natural multi-turn dialogue.

    Tracks topics, user facts, emotions, and conversation flow to generate
    contextual follow-up questions and show genuine curiosity.

    Complexity: T(n) ∈ O(1) per operation, S(n) ∈ O(topics + facts).
    """
    topics: Dict[str, TrackedTopic] = field(default_factory=dict)
    user_facts: Dict[str, str] = field(default_factory=dict)  # "name" -> "Alice"
    turn_count: int = 0
    emotion_history: List[str] = field(default_factory=list)
    asked_questions: List[str] = field(default_factory=list)  # avoid repeating
    previous_responses: List[str] = field(default_factory=list)  # anti-repetition
    last_intent: str = "unknown"
    last_topic_domain: str = "general"
    conversation_start: float = field(default_factory=time.time)
    user_name: Optional[str] = None
    user_interests: List[str] = field(default_factory=list)
    pending_follow_up: Optional[str] = None  # follow-up to inject next turn

    def add_topic(self, word: str, emotion: str = "neutral") -> TrackedTopic:
        """Add or update a tracked topic.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        word_lower = word.lower()
        # Check multi-word domain map first, then single-word map
        domain = _MULTI_WORD_DOMAIN_MAP.get(word_lower) or _TOPIC_DOMAIN_MAP.get(word_lower, "general")
        if word in self.topics:
            self.topics[word].touch(emotion)
        else:
            self.topics[word] = TrackedTopic(
                word=word, domain=domain, emotion_when_mentioned=emotion
            )
        self.last_topic_domain = domain
        return self.topics[word]

    def get_top_topics(self, n: int = 3) -> List[TrackedTopic]:
        """Get the top-N most engaged topics by frequency × recency.

        Complexity: T(n) ∈ O(t log t) for t topics, S(n) ∈ O(n).
        """
        if not self.topics:
            return []
        scored = sorted(
            self.topics.values(),
            key=lambda t: (t.frequency, t.last_mentioned),
            reverse=True,
        )
        return scored[:n]

    def get_dominant_domain(self) -> str:
        """Get the most discussed domain.

        Complexity: T(n) ∈ O(t) for t topics, S(n) ∈ O(1).
        """
        if not self.topics:
            return "general"
        domain_counts: Dict[str, int] = {}
        for topic in self.topics.values():
            domain_counts[topic.domain] = domain_counts.get(topic.domain, 0) + topic.frequency
        return max(domain_counts, key=domain_counts.get) if domain_counts else "general"

    def has_asked(self, question: str) -> bool:
        """Check if a similar question was already asked.

        Complexity: T(n) ∈ O(q) for q asked questions, S(n) ∈ O(1).
        """
        question_lower = question.lower().strip()
        for prev in self.asked_questions:
            # Check for >60% word overlap to detect near-duplicates
            prev_words = set(prev.lower().split())
            curr_words = set(question_lower.split())
            if prev_words and curr_words:
                overlap = len(prev_words & curr_words) / max(len(prev_words), len(curr_words))
                if overlap > 0.6:
                    return True
        return False

    def record_question(self, question: str) -> None:
        """Record a asked question to avoid repetition.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self.asked_questions.append(question)
        if len(self.asked_questions) > 10:
            self.asked_questions = self.asked_questions[-10:]

    def record_response(self, response: str) -> None:
        """Record an AI response for anti-repetition detection.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Store a truncated hash-like signature (first 100 chars)
        self.previous_responses.append(response[:200])
        if len(self.previous_responses) > 5:
            self.previous_responses = self.previous_responses[-5:]

    def is_repetitive_response(self, response: str) -> bool:
        """Check if a response is too similar to a previous one.

        Compares word overlap with the last few AI responses.

        Complexity: T(n) ∈ O(r * w) for r previous responses, w words.
        """
        if not self.previous_responses or not response:
            return False

        # Extract significant words from the new response
        new_words = set(re.findall(r"[a-z]{4,}", response.lower()))
        if not new_words:
            return False

        for prev in self.previous_responses[-3:]:
            prev_words = set(re.findall(r"[a-z]{4,}", prev.lower()))
            if prev_words:
                overlap = len(new_words & prev_words) / max(len(new_words), len(prev_words))
                if overlap > 0.75:
                    return True
        return False

    def extract_user_fact(self, user_input: str) -> Optional[Tuple[str, str]]:
        """Extract a user fact (name, interest, location, etc.) from input.

        Returns:
            Tuple of (fact_type, fact_value) or None.

        Complexity: T(n) ∈ O(n) for n chars in input, S(n) ∈ O(1).
        """
        input_orig = user_input.strip()
        input_lower = input_orig.lower()

        # Name extraction — use original case to detect proper nouns
        # "my name is Tobias" matches, "i am feeling sad" does not
        name_patterns = [
            r"my name is ([A-Z][a-z]+)",
            r"i'?m called ([A-Z][a-z]+)",
            r"i am ([A-Z][a-z]+)(?:\s|$|[.,!?])",
            r"i'?m ([A-Z][a-z]+)(?:\s|$|[.,!?])",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, input_orig)
            if match:
                name = match.group(1)
                # Filter out common words that aren't names
                if name.lower() not in ("fine", "good", "great", "happy", "sad",
                                          "sorry", "here", "not", "sure", "okay",
                                          "tired", "busy", "done", "back", "new",
                                          "feeling", "thinking", "working", "looking",
                                          "going", "trying", "getting", "doing",
                                          "glad", "angry", "worried", "grateful",
                                          "confused", "curious", "interested",
                                          "wondering", "hoping", "wishing"):
                    self.user_name = name
                    self.user_facts["name"] = name
                    return ("name", name)

        # Interest extraction
        interest_patterns = [
            (r"i like (\w+)", "interest"),
            (r"i love (\w+)", "interest"),
            (r"i enjoy (\w+)", "interest"),
            (r"i'?m interested in (\w+)", "interest"),
            (r"my hobby is (\w+)", "interest"),
        ]
        for pattern, fact_type in interest_patterns:
            match = re.search(pattern, input_lower)
            if match:
                interest = match.group(1)
                if interest not in _TOPIC_STOP_WORDS and len(interest) > 2:
                    if interest not in self.user_interests:
                        self.user_interests.append(interest)
                    self.user_facts[f"interest_{interest}"] = interest
                    return ("interest", interest)

        # Location extraction
        location_patterns = [
            r"i live in (\w+)",
            r"i'?m from (\w+)",
            r"i live in (\w+ \w+)",
        ]
        for pattern in location_patterns:
            match = re.search(pattern, input_lower)
            if match:
                location = match.group(1).title()
                self.user_facts["location"] = location
                return ("location", location)

        # Work extraction
        work_patterns = [
            (r"i work (?:as|as a) (\w+)", "work"),
            (r"i'?m a (\w+)", "work"),
            (r"my job is (\w+)", "work"),
        ]
        for pattern, fact_type in work_patterns:
            match = re.search(pattern, input_lower)
            if match:
                work = match.group(1)
                if work not in _TOPIC_STOP_WORDS and len(work) > 2:
                    self.user_facts["work"] = work
                    return ("work", work)

        return None

    def get_personalized_greeting(self) -> Optional[str]:
        """Get a personalized greeting using known user facts.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if self.user_name and self.turn_count > 0:
            return f"Welcome back, {self.user_name}."
        return None


class TopicTracker:
    """Extracts and tracks topics from user messages.

    Uses keyword frequency analysis to identify what the user is talking
    about, maps keywords to domains, and maintains a per-session topic
    profile.

    Complexity:
        extract: T(n) ∈ O(n) for n words in input.
        Memory: S(n) ∈ O(t) for t tracked topics.
    """

    __slots__ = ()

    def extract_topics(
        self,
        user_input: str,
        profile: ConversationProfile,
        emotion: str = "neutral",
    ) -> List[str]:
        """Extract topics from user input and update the conversation profile.

        Detects multi-word phrases first (e.g. "quantum physics"), then
        falls back to single-word extraction for remaining significant words.

        Args:
            user_input: The user's message text.
            profile: Per-session conversation profile to update.
            emotion: Detected emotion for this turn.

        Returns:
            List of extracted topic words/phrases (most significant first).

        Complexity: T(n) ∈ O(n + p) for n words, p multi-word phrases.
        """
        # First, extract user facts (name, interests, etc.) so we can
        # exclude the user's name from topic tracking
        profile.extract_user_fact(user_input)

        # Build exclusion set — don't track user's name as a topic
        exclude_words: Set[str] = set()
        if profile.user_name:
            exclude_words.add(profile.user_name.lower())
        for interest in profile.user_interests:
            exclude_words.add(interest.lower())

        input_lower = user_input.lower()

        # ─── Phase 1: Detect multi-word phrases ──────────────────────
        found_phrases: List[str] = []
        consumed_spans: List[Tuple[int, int]] = []  # (start, end) char positions

        for phrase in _MULTI_WORD_TOPICS:
            idx = input_lower.find(phrase)
            while idx != -1:
                # Check this span doesn't overlap with already-found phrases
                span = (idx, idx + len(phrase))
                if not any(s[0] < span[1] and s[1] > span[0] for s in consumed_spans):
                    found_phrases.append(phrase)
                    consumed_spans.append(span)
                idx = input_lower.find(phrase, idx + 1)

        # ─── Phase 2: Single-word extraction (excluding consumed spans) ──
        words = re.finditer(r"[a-z]{3,}", input_lower)
        single_words: List[str] = []
        for match in words:
            word = match.group()
            if word in _TOPIC_STOP_WORDS or word in exclude_words:
                continue
            # Check if this word is inside a consumed multi-word span
            pos = match.start()
            if any(s[0] <= pos < s[1] for s in consumed_spans):
                continue
            single_words.append(word)

        # Combine: multi-word phrases first, then single words
        all_topics = found_phrases + single_words

        if not all_topics:
            return []

        # Count frequency for sorting
        topic_freq: Dict[str, int] = {}
        for topic in all_topics:
            topic_freq[topic] = topic_freq.get(topic, 0) + 1

        # Sort by frequency (most mentioned first), preserving phrase priority
        sorted_topics = sorted(all_topics, key=lambda t: topic_freq[t], reverse=True)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique_topics: List[str] = []
        for topic in sorted_topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)

        # Track top topics in the profile (max 5)
        extracted: List[str] = []
        for topic in unique_topics[:5]:
            profile.add_topic(topic, emotion)
            extracted.append(topic)

        return extracted


class FollowUpGenerator:
    """Generates contextual follow-up questions.

    Produces natural, varied follow-up questions based on:
    - The user's intent (question, emotional, code, etc.)
    - The detected emotion
    - Tracked topics and their domains
    - User facts (name, interests, location)
    - Conversation history (avoids repetition)

    Complexity:
        generate: T(n) ∈ O(1) template selection, S(n) ∈ O(1).
    """

    __slots__ = ("_rng",)

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the follow-up generator.

        Args:
            seed: Optional random seed for reproducible questions.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._rng = random.Random(seed)

    def generate(
        self,
        intent: str,
        emotion: str,
        user_input: str,
        profile: ConversationProfile,
        topics: List[str],
    ) -> Optional[str]:
        """Generate a contextual follow-up question.

        Args:
            intent: Classified intent for this turn.
            emotion: Detected emotion for this turn.
            user_input: The user's message.
            profile: Per-session conversation profile.
            topics: Topics extracted from this message.

        Returns:
            A follow-up question string, or None if no question is appropriate.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        # Don't ask follow-ups on every turn — vary the pattern
        # Turn 1: Always ask (establish engagement)
        # Turn 2+: Ask ~85% of the time (more conversational)
        if profile.turn_count > 1 and self._rng.random() > 0.85:
            return None

        # Don't ask if user is asking a direct question (let them get their answer first)
        if intent in ("question", "code", "architect") and profile.turn_count == 0:
            # But DO ask a follow-up after answering (handled by synthesis)
            pass

        candidates: List[str] = []

        # ─── Priority 1: Emotional empathy (always first) ────────────
        # When the user is emotional, empathy takes priority over curiosity
        if emotion == "sad":
            candidates.extend([
                "Would you like to talk more about what's been weighing on you?",
                "What happened that made you feel this way?",
                "I'm here to listen. What's been going on?",
                "Take your time. What happened?",
            ])
        elif emotion == "angry":
            candidates.extend([
                "What happened that made you feel this way?",
                "Would you like to talk through what happened?",
                "I hear you. What's going on?",
                "That sounds frustrating. What happened?",
            ])
        elif emotion == "anxious":
            candidates.extend([
                "What's causing you to feel worried right now?",
                "Is there something specific that's on your mind?",
                "Would it help to talk through what's worrying you?",
                "What's making you feel anxious?",
            ])
        elif emotion == "happy":
            candidates.extend([
                "That's wonderful! What's bringing you joy today?",
                "Tell me more — what's making you feel good?",
                "I love hearing that. What happened?",
                "That's great! What's going well?",
            ])
        elif emotion == "grateful":
            candidates.extend([
                "What are you most thankful for right now?",
                "What inspired this feeling of gratitude?",
                "That's beautiful. What are you grateful for?",
            ])

        # ─── Priority 2: Personalization (use known facts) ───────────
        # Use the user's name to make follow-ups feel personal
        name_prefix = f"{profile.user_name}, " if profile.user_name else ""

        # First-time greeting with name
        if profile.user_name and profile.turn_count == 0:
            candidates.insert(0, f"Nice to meet you, {profile.user_name}! What's on your mind?")

        # Recall interests from earlier turns
        if profile.user_interests and profile.turn_count > 2:
            interest = self._rng.choice(profile.user_interests)
            candidates.append(f"You mentioned you like {interest} earlier — have you been doing that recently?")

        # Recall location from earlier turns
        if "location" in profile.user_facts and profile.turn_count > 3:
            location = profile.user_facts["location"]
            candidates.append(f"How are things in {location}?")

        # ─── Priority 3: Topic-based curiosity (skip if emotional) ───
        # Only ask topic-based questions when the user is NOT in an emotional state
        if topics and emotion in ("neutral", "curious", "philosophical"):
            primary_topic = topics[0]
            topic_lower = primary_topic.lower()
            domain = _MULTI_WORD_DOMAIN_MAP.get(topic_lower) or _TOPIC_DOMAIN_MAP.get(topic_lower, "general")

            if domain == "programming":
                candidates.extend([
                    f"Are you building something with {primary_topic} right now?",
                    f"What kind of project are you working on?",
                    f"How long have you been into {primary_topic}?",
                    f"What's your favorite thing about {primary_topic}?",
                    f"What got you started with {primary_topic}?",
                ])
            elif domain == "science":
                candidates.extend([
                    f"What aspect of {primary_topic} interests you the most?",
                    f"Are you studying {primary_topic}, or is it more of a personal interest?",
                    f"What got you curious about {primary_topic}?",
                    f"What's the most fascinating thing about {primary_topic} to you?",
                    f"Is there something specific about {primary_topic} you're exploring?",
                ])
            elif domain == "arts":
                candidates.extend([
                    f"Do you create {primary_topic} yourself, or is it more of an appreciation?",
                    f"What's your favorite piece of {primary_topic}?",
                    f"Who inspires you in the world of {primary_topic}?",
                ])
            elif domain == "philosophy":
                candidates.extend([
                    f"What led you to think about {primary_topic}?",
                    f"Have you always been interested in questions about {primary_topic}?",
                    f"What's your own perspective on {primary_topic}?",
                ])
            elif domain == "personal":
                candidates.extend([
                    f"Tell me more about that.",
                    f"How long has that been part of your life?",
                    f"What does that mean to you personally?",
                ])
            elif domain == "geography":
                candidates.extend([
                    f"Have you been to {primary_topic.capitalize()} yourself?",
                    f"What made you think about {primary_topic}?",
                    f"Is there somewhere you'd like to travel related to {primary_topic}?",
                ])
            elif domain == "health":
                candidates.extend([
                    f"Are you focusing on {primary_topic} as part of your routine?",
                    f"What's your approach to {primary_topic}?",
                ])
            elif domain == "business":
                candidates.extend([
                    f"Are you working on something in the {primary_topic} space?",
                    f"What's your experience with {primary_topic}?",
                ])
            elif domain == "mathematics":
                candidates.extend([
                    f"What part of {primary_topic} are you working on?",
                    f"Are you studying {primary_topic}, or using it for something specific?",
                ])
            elif domain == "history":
                candidates.extend([
                    f"What period of {primary_topic} interests you most?",
                    f"Are you reading about {primary_topic}, or is it from personal study?",
                ])
            else:
                # Generic topic curiosity — more natural phrasing
                candidates.extend([
                    f"What got you interested in {primary_topic}?",
                    f"Tell me more about that — what's your take on it?",
                    f"How did you get into {primary_topic}?",
                    f"What's your experience with {primary_topic}?",
                ])

            # ─── Proactive curiosity: share an observation, not just ask ──
            # ~20% of the time, share a thought instead of asking a question
            if self._rng.random() < 0.2 and profile.turn_count > 0:
                observations = self._generate_observation(primary_topic, domain, profile)
                if observations:
                    candidates.insert(0, self._rng.choice(observations))

        # ─── Priority 4: Intent-based follow-ups ─────────────────────
        if intent == "identity" and profile.turn_count == 0:
            candidates.append("What made you curious about me?")

        if intent == "greet" and profile.turn_count == 0:
            candidates.append("What's on your mind today?")

        if intent == "clarify":
            candidates.append("What specifically would you like me to clarify?")

        # ─── Priority 5: Conversation flow (deep engagement) ─────────
        if profile.turn_count >= 3 and not candidates:
            top_topics = profile.get_top_topics(1)
            if top_topics:
                topic = top_topics[0]
                if topic.frequency >= 2:
                    candidates.append(
                        f"You've mentioned {topic.word} a few times — it seems important to you. Tell me more."
                    )

        # ─── Select a non-repeated question ──────────────────────────
        if not candidates:
            return None

        # Shuffle and pick the first non-repeated question
        self._rng.shuffle(candidates)
        for question in candidates:
            if not profile.has_asked(question):
                profile.record_question(question)
                return question

        # All questions were repeats — generate a generic one
        generic = self._rng.choice([
            "What else is on your mind?",
            "Is there anything else you'd like to explore?",
            "Where would you like to take this conversation?",
            "What else would you like to talk about?",
            "Anything else on your mind?",
        ])
        return generic

    def _generate_observation(
        self,
        topic: str,
        domain: str,
        profile: ConversationProfile,
    ) -> List[str]:
        """Generate a proactive observation or shared thought.

        Instead of always asking questions, sometimes the AI shares a
        thought or observation to feel more human and engaging.

        Args:
            topic: The primary topic.
            domain: The topic's domain.
            profile: Conversation profile for context.

        Returns:
            List of observation strings (may be empty).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        observations: List[str] = []

        if domain == "programming":
            observations.extend([
                f"I find {topic} fascinating — it's amazing how quickly the field evolves.",
                f"{topic.capitalize()} is one of those fields where there's always something new to learn.",
                f"I've always been curious about {topic} from a systems perspective.",
            ])
        elif domain == "science":
            observations.extend([
                f"{topic.capitalize()} is such a deep field — there's always more to discover.",
                f"I find it remarkable how {topic} challenges our intuition about the world.",
                f"The thing about {topic} is that the more you learn, the more questions emerge.",
            ])
        elif domain == "arts":
            observations.extend([
                f"{topic.capitalize()} has such power to move people — it's a universal language.",
                f"What I find beautiful about {topic} is how subjective it is.",
            ])
        elif domain == "philosophy":
            observations.extend([
                f"{topic.capitalize()} is one of those topics where the questions matter more than the answers.",
                f"I find {topic} endlessly fascinating — it touches everything we do.",
            ])
        elif domain == "personal":
            observations.extend([
                f"That says a lot about who you are, I think.",
                f"I appreciate you sharing that with me.",
            ])
        elif domain == "geography":
            observations.extend([
                f"{topic.capitalize()} is a place I'd love to learn more about.",
                f"Every place has its own story, doesn't it?",
            ])
        elif domain == "health":
            observations.extend([
                f"Taking care of yourself is so important — good that you're thinking about it.",
            ])
        elif domain == "business":
            observations.extend([
                f"The {topic} world moves fast — it's exciting to watch.",
            ])
        else:
            observations.extend([
                f"That's an interesting topic — I'd love to hear your perspective on it.",
                f"I find that topic genuinely interesting.",
            ])

        # Add a personalized observation if we know the user's name
        if profile.user_name and profile.turn_count > 1:
            observations.append(
                f"You know, {profile.user_name}, you bring up the most interesting topics."
            )

        return observations


class CuriosityEngine:
    """Decides when and what the AI should ask about.

    Simulates AI curiosity by scoring topics and conversation patterns
    to determine when a proactive question feels natural and engaging.

    Curiosity scoring factors:
    - Topic novelty (new topics score higher)
    - Emotional resonance (emotional topics trigger curiosity)
    - Conversation depth (deeper conversations warrant deeper questions)
    - User engagement (frequent topics indicate interest)

    Complexity:
        should_ask: T(n) ∈ O(1), S(n) ∈ O(1).
        curiosity_score: T(n) ∈ O(t) for t topics, S(n) ∈ O(1).
    """

    __slots__ = ("_follow_up_gen", "_topic_tracker")

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the curiosity engine.

        Args:
            seed: Optional random seed.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._follow_up_gen = FollowUpGenerator(seed=seed)
        self._topic_tracker = TopicTracker()

    def process_turn(
        self,
        user_input: str,
        intent: str,
        emotion: str,
        profile: ConversationProfile,
    ) -> Tuple[List[str], Optional[str]]:
        """Process a conversation turn and generate topics + follow-up.

        Args:
            user_input: The user's message.
            intent: Classified intent.
            emotion: Detected emotion.
            profile: Per-session conversation profile.

        Returns:
            Tuple of (extracted_topics, follow_up_question).

        Complexity: T(n) ∈ O(n) for n words, S(n) ∈ O(t) for t topics.
        """
        # Extract and track topics
        topics = self._topic_tracker.extract_topics(user_input, profile, emotion)

        # Update profile state
        profile.turn_count += 1
        profile.last_intent = intent
        profile.emotion_history.append(emotion)
        if len(profile.emotion_history) > 10:
            profile.emotion_history = profile.emotion_history[-10:]

        # Generate follow-up question
        follow_up = self._follow_up_gen.generate(
            intent=intent,
            emotion=emotion,
            user_input=user_input,
            profile=profile,
            topics=topics,
        )

        return topics, follow_up

    def curiosity_score(self, profile: ConversationProfile) -> float:
        """Score how curious the AI should be (0.0 to 1.0).

        Higher scores indicate more curiosity — the AI should ask more
        probing questions.

        Factors:
        - New topics increase curiosity
        - Emotional topics increase curiosity
        - Longer conversations decrease curiosity (avoid interrogation)
        - Repeated topics decrease curiosity (already explored)

        Complexity: T(n) ∈ O(t) for t topics, S(n) ∈ O(1).
        """
        if profile.turn_count == 0:
            return 0.8  # High curiosity at conversation start

        # Base curiosity decreases with conversation length
        base = max(0.2, 1.0 - (profile.turn_count * 0.05))

        # Boost for emotional content
        recent_emotions = profile.emotion_history[-3:]
        emotional_boost = sum(0.1 for e in recent_emotions if e not in ("neutral", "curious"))
        base += emotional_boost

        # Boost for topic diversity
        unique_domains = set(t.domain for t in profile.topics.values())
        diversity_boost = min(0.2, len(unique_domains) * 0.05)
        base += diversity_boost

        # Penalty for repetitive topics
        max_freq = max((t.frequency for t in profile.topics.values()), default=1)
        if max_freq > 3:
            base -= 0.1

        return max(0.1, min(1.0, base))


class ConversationalEngine:
    """Main conversational engine combining all components.

    Provides a single interface for the pipeline to:
    1. Track conversation topics per session
    2. Generate follow-up questions
    3. Show curiosity about user's topics
    4. Personalize responses using user facts
    5. Keep conversations flowing naturally

    Usage:
        engine = ConversationalEngine()
        profile = engine.get_profile(session_id)
        topics, follow_up = engine.process_turn(
            session_id, user_input, intent, emotion
        )
        response_with_follow_up = engine.inject_follow_up(response, follow_up)

    Complexity:
        get_profile: T(n) ∈ O(1)
        process_turn: T(n) ∈ O(n) for n words in input
        inject_follow_up: T(n) ∈ O(1)
    """

    __slots__ = ("_profiles", "_curiosity", "_lock")

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize the conversational engine.

        Args:
            seed: Optional random seed for reproducible questions.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        self._profiles: Dict[str, ConversationProfile] = {}
        self._curiosity = CuriosityEngine(seed=seed)
        self._lock: threading.RLock = threading.RLock()

    def get_profile(self, session_id: str) -> ConversationProfile:
        """Get or create a conversation profile for a session.

        Args:
            session_id: Unique session identifier.

        Returns:
            ConversationProfile for this session.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            if session_id not in self._profiles:
                self._profiles[session_id] = ConversationProfile()
            return self._profiles[session_id]

    def process_turn(
        self,
        session_id: str,
        user_input: str,
        intent: str,
        emotion: str,
    ) -> Tuple[List[str], Optional[str]]:
        """Process a conversation turn for a session.

        Args:
            session_id: Session identifier.
            user_input: User's message.
            intent: Classified intent.
            emotion: Detected emotion.

        Returns:
            Tuple of (extracted_topics, follow_up_question).

        Complexity: T(n) ∈ O(n) for n words, S(n) ∈ O(t) for t topics.
        """
        with self._lock:
            profile = self.get_profile(session_id)
            return self._curiosity.process_turn(user_input, intent, emotion, profile)

    def inject_follow_up(self, response: str, follow_up: Optional[str]) -> str:
        """Inject a follow-up question into a response.

        Appends the follow-up naturally to the end of the response,
        with appropriate spacing. If the response already ends with a
        question, the follow-up is connected with a transition phrase.
        Avoids creating awkward double-questions.

        Args:
            response: The AI's response text.
            follow_up: Optional follow-up question to append.

        Returns:
            Response with follow-up question appended (if any).

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        if not follow_up:
            return response

        response = response.rstrip()
        if not response:
            return follow_up

        # Count existing questions in the response
        question_count = response.count("?")

        # If response already ends with a question, use a transition
        if response.endswith("?"):
            # If there are already 2+ questions, don't add another — replace instead
            if question_count >= 2:
                # Replace the last question with the follow-up
                last_q_idx = response.rfind("?")
                # Find the start of the last question sentence
                sent_start = response.rfind(". ", 0, last_q_idx)
                if sent_start != -1:
                    response = response[:sent_start + 1].rstrip()
                    return f"{response} {follow_up}"
                # Can't cleanly separate — just append with transition
            transitions = [
                f" And {follow_up[0].lower()}{follow_up[1:]}",
                f" By the way, {follow_up[0].lower()}{follow_up[1:]}",
                f" Also, {follow_up[0].lower()}{follow_up[1:]}",
                f" {follow_up}",
            ]
            return response + random.choice(transitions)

        # If response ends with a period or other punctuation, just append
        return f"{response} {follow_up}"

    def enrich_response(
        self,
        session_id: str,
        response: str,
        intent: str,
        emotion: str,
    ) -> str:
        """Enrich a response with context-aware personalization.

        Adds a contextual prefix that references previous turns, uses
        the user's name, or acknowledges a topic shift. This makes the
        AI feel more present and aware of the conversation flow.

        Args:
            session_id: Session identifier.
            response: The AI's response text.
            intent: Current turn's intent.
            emotion: Current turn's emotion.

        Returns:
            Enriched response string.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            profile = self._profiles.get(session_id)
            if profile is None or profile.turn_count <= 1:
                return response

            # Don't enrich emotional responses — empathy should come first
            if emotion in ("sad", "angry", "anxious"):
                return response

            # Don't enrich if response is very short
            if len(response) < 50:
                return response

            prefixes: List[str] = []

            # Reference a previous topic if the user is returning to it
            if profile.topics and profile.turn_count > 2:
                top_topic = profile.get_top_topics(1)[0]
                if top_topic.frequency >= 2 and top_topic.word.lower() in response.lower():
                    # User is talking about a recurring topic
                    if profile.user_name:
                        prefixes.append(f"Coming back to {top_topic.word}, {profile.user_name} — ")
                    else:
                        prefixes.append(f"Coming back to {top_topic.word} — ")

            # Acknowledge topic shift
            if profile.last_intent != intent and profile.turn_count > 1:
                if intent == "emotional" and profile.last_intent in ("question", "explain"):
                    prefixes.append("I appreciate you sharing this. ")
                elif intent == "question" and profile.last_intent == "emotional":
                    prefixes.append("I'm glad you're feeling better. ")

            # Use user's name occasionally (not every turn)
            if profile.user_name and profile.turn_count > 2 and random.random() < 0.3:
                name_greetings = [
                    f"{profile.user_name}, ",
                    f"So, {profile.user_name} — ",
                ]
                prefixes.append(random.choice(name_greetings))

            if prefixes:
                prefix = random.choice(prefixes)
                # Don't double-prefix if response already starts with the name
                if not response.lower().startswith(profile.user_name.lower() if profile.user_name else "___"):
                    return prefix + response[0].lower() + response[1:]

            return response

    def get_profile_summary(self, session_id: str) -> Dict[str, any]:
        """Get a summary of the conversation profile for a session.

        Used for debugging and the API response.

        Args:
            session_id: Session identifier.

        Returns:
            Dictionary with profile summary.

        Complexity: T(n) ∈ O(t) for t topics, S(n) ∈ O(t).
        """
        profile = self._profiles.get(session_id)
        if profile is None:
            return {
                "active": False,
                "turn_count": 0,
                "topics": [],
                "user_name": None,
                "curiosity_score": 0.0,
            }

        top_topics = profile.get_top_topics(5)
        return {
            "active": True,
            "turn_count": profile.turn_count,
            "topics": [
                {"word": t.word, "domain": t.domain, "frequency": t.frequency}
                for t in top_topics
            ],
            "user_name": profile.user_name,
            "user_interests": profile.user_interests[:5],
            "dominant_domain": profile.get_dominant_domain(),
            "curiosity_score": round(self._curiosity.curiosity_score(profile), 2),
            "follow_up_pending": profile.pending_follow_up is not None,
        }

    def clear_session(self, session_id: str) -> bool:
        """Clear the conversation profile for a session.

        Args:
            session_id: Session identifier.

        Returns:
            True if the profile was found and cleared.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            if session_id in self._profiles:
                del self._profiles[session_id]
                return True
            return False

    def active_session_count(self) -> int:
        """Return the number of active conversation profiles.

        Complexity: T(n) ∈ O(1), S(n) ∈ O(1).
        """
        with self._lock:
            return len(self._profiles)
