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
Live conversational intelligence simulation.

Generates realistic human-like chat conversations across emotional tones:
  - Happy / excited
  - Sad / depressed
  - Angry / frustrated
  - Off-topic / random
  - Philosophical / deep
  - Personal / emotional
  - Confused / lost
  - Grateful / thankful
  - Fearful / anxious
  - Curious / wondering

Each conversation is a multi-turn chat where the user's messages shift
topics, drift off-topic, express emotions, and test whether the system
responds with appropriate tone, empathy, and coherence.

Copyright 2026 Tobias Østen. Licensed under Apache License 2.0.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core import BrainController


# ─── Emotional Tone Definitions ──────────────────────────────────────────────

TONES: Dict[str, List[str]] = {
    "happy": [
        "I just got promoted at work! I'm so happy right now!",
        "My daughter took her first steps today! I'm crying tears of joy.",
        "I finally finished my thesis after 6 months! Best feeling ever!",
        "The weather is amazing today, isn't life wonderful?",
        "I passed my driving test on the first try! So excited!",
        "My best friend is visiting me this weekend, I haven't seen her in years!",
        "I just adopted a puppy! He's the cutest thing I've ever seen!",
        "I ran my first marathon today and finished! I can't believe it!",
        "I got accepted into my dream university! This is the best day!",
        "My garden is blooming beautifully this spring, everything feels alive!",
    ],
    "sad": [
        "I lost my dog yesterday. He was my best friend for 14 years.",
        "My grandmother passed away last week. I don't know how to cope.",
        "I just got fired from my job. I feel like a complete failure.",
        "Nobody remembered my birthday today. I feel so invisible.",
        "I've been struggling with depression for months and nobody seems to care.",
        "My parents are getting a divorce and I feel like my world is falling apart.",
        "I miss my old life before everything changed. I just want to go back.",
        "I failed my exam even though I studied for weeks. I'm so disappointed.",
        "My relationship just ended. I thought we were going to get married.",
        "I feel so lonely sometimes. Like nobody would notice if I disappeared.",
    ],
    "angry": [
        "I'm so angry at my coworker! He took credit for MY work again!",
        "This stupid computer keeps crashing every 5 minutes, I want to throw it out the window!",
        "Why does the government keep lying to us? I'm fed up with all of it!",
        "My neighbor plays loud music at 3 AM every single night. I'm losing my mind!",
        "I've been on hold for 2 hours with customer service. This is unacceptable!",
        "Someone scratched my car in the parking lot and didn't leave a note. Furious!",
        "I can't believe they cancelled my favorite show after one season!",
        "My boss expects me to work overtime every day without extra pay. It's exploitation!",
        "The delivery driver left my package in the rain and now it's ruined!",
        "I'm sick and tired of people not taking me seriously because of my age!",
    ],
    "offtopic": [
        "Did you know that octopuses have three hearts?",
        "I wonder what clouds taste like. Probably like cotton candy, right?",
        "If you could be any kitchen appliance, which one would you be?",
        "Do you think fish know they're wet?",
        "I had a dream about a purple elephant last night. What do you think it means?",
        "Why do we park in driveways and drive on parkways?",
        "Do penguins have knees? I've been thinking about this all day.",
        "What if the universe is just a giant simulation running on a cosmic computer?",
        "I think hot dogs should be classified as sandwiches. Change my mind.",
        "If time travel existed, would you go to the past or the future?",
        "My cat stares at the wall for hours. I think she sees ghosts.",
        "Do you ever wonder how many people have walked on this exact same spot before?",
    ],
    "philosophical": [
        "What is the meaning of life? I've been thinking about this a lot lately.",
        "Do we truly have free will, or is everything predetermined?",
        "Is consciousness just an illusion created by our brain?",
        "What makes something 'right' or 'wrong'? Is morality objective or subjective?",
        "If a tree falls in a forest and nobody hears it, does it make a sound?",
        "Are we the same person we were 10 years ago, or have we become someone new?",
        "Can true happiness exist without sadness? Or are they two sides of the same coin?",
        "What happens after we die? Is there something beyond this life?",
        "Is it better to be a dissatisfied human or a satisfied pig?",
        "Does the universe have a purpose, or are we just a cosmic accident?",
    ],
    "personal": [
        "I'm scared about my health checkup tomorrow. What if something is wrong?",
        "I've been thinking about quitting my job to pursue my passion. Am I crazy?",
        "I don't know if I should move to a new city. Everything I know is here.",
        "My partner and I have been arguing a lot lately. I love them but I'm exhausted.",
        "I'm 35 and I feel like I haven't accomplished anything meaningful in life.",
        "I've been sober for 6 months now. Some days are harder than others.",
        "I want to reconnect with my estranged father but I don't know where to start.",
        "I think I might be in the wrong career. I've been a lawyer for 10 years and I hate it.",
        "I'm about to become a father and I'm terrified. What if I'm not good enough?",
        "I lost a lot of weight but I still feel unhappy with my body. Is something wrong with me?",
    ],
    "confused": [
        "Wait, so you're saying that Python and JavaScript are different? I thought they were the same thing.",
        "I don't understand. How can a function call itself? That doesn't make sense.",
        "I'm completely lost. Can you explain this in simpler terms?",
        "I read three articles about this and I'm more confused than before.",
        "I don't get it. Why does my code work on my computer but not on the server?",
        "I'm so confused about taxes. Why do I owe money if I already paid?",
        "I don't understand people who enjoy running. It just seems like torture.",
        "Can you help me? I don't even know what question to ask.",
        "I've been staring at this error message for an hour and I still don't get it.",
        "How does the internet actually work? Like, where does the information come from?",
    ],
    "grateful": [
        "Thank you so much for your help earlier. It really made a difference.",
        "I just want to say I'm grateful for this conversation. You've been incredibly helpful.",
        "I don't say this enough, but I really appreciate everything you do for me.",
        "You helped me see things from a completely different perspective. Thank you.",
        "I'm so thankful for my family. They've supported me through everything.",
        "I just realized how lucky I am to have good health. I took it for granted.",
        "Thank you for being patient with me. I know I can be difficult sometimes.",
        "I'm grateful for the little things in life. A good cup of coffee, a warm bed.",
        "You have no idea how much your advice helped me. I got the job!",
        "I want to thank everyone who believed in me when I didn't believe in myself.",
    ],
    "anxious": [
        "I can't stop worrying about my job interview tomorrow. What if I mess up?",
        "I feel like something bad is going to happen. I can't shake this feeling.",
        "My chest feels tight and I can't breathe. I think I'm having a panic attack.",
        "I'm terrified of flying and I have a 10-hour flight next week.",
        "What if I made the wrong decision? I keep second-guessing everything.",
        "I can't sleep because my mind won't stop racing. I'm exhausted.",
        "I'm worried about my friend. She hasn't responded to my messages in days.",
        "I feel like everyone is judging me all the time. I can't relax.",
        "What if I'm not good enough? What if everyone finds out I'm a fraud?",
        "I'm scared of the future. Everything feels so uncertain right now.",
    ],
    "curious": [
        "How do neurons in the brain actually communicate with each other?",
        "What's the difference between a virus and a bacteria?",
        "How do solar panels convert sunlight into electricity?",
        "Why do we dream? Has anyone figured that out yet?",
        "How do bees communicate? I heard they do a dance or something.",
        "What's the most efficient way to learn a new language?",
        "How does encryption actually work? Like, in simple terms.",
        "Why do we get goosebumps? What's the evolutionary purpose?",
        "How do plants know when to flower? Do they have some kind of internal clock?",
        "What's at the bottom of the ocean? Has anyone actually been there?",
    ],
}


# ─── Conversation Flow Templates ─────────────────────────────────────────────

# Multi-turn conversation templates that shift topics and emotions
CONVERSATION_FLOWS: List[List[Tuple[str, str]]] = [
    # Flow 1: Happy -> Sad -> Philosophical
    [
        ("happy", "I just got engaged! We've been together for 5 years!"),
        ("happy", "We're already planning the wedding, it's going to be amazing!"),
        ("sad", "But then I found out my mom has cancer. I don't know what to do."),
        ("sad", "She might not make it to the wedding. That thought destroys me."),
        ("philosophical", "Why does life give you the best and worst moments at the same time?"),
        ("philosophical", "Is suffering necessary for happiness to mean anything?"),
        ("personal", "I just want her to be there. That's all I ask for."),
    ],
    # Flow 2: Angry -> Confused -> Grateful
    [
        ("angry", "I'm so done with this company! They never listen to their employees!"),
        ("angry", "I submitted a complaint and they literally ignored it for 3 weeks!"),
        ("confused", "But wait... maybe I should have talked to my manager first?"),
        ("confused", "I don't know if I overreacted. Was I in the wrong?"),
        ("grateful", "Actually, you know what? My manager did help me last month with that project."),
        ("grateful", "I should be more grateful for what I have instead of focusing on the negative."),
        ("personal", "I think I need to work on my communication skills. Can you help?"),
    ],
    # Flow 3: Off-topic -> Curious -> Philosophical
    [
        ("offtopic", "Did you know that bananas are berries but strawberries aren't?"),
        ("offtopic", "Nature is weird. Like, really weird if you think about it."),
        ("curious", "How do scientists even classify things like that? What makes a berry a berry?"),
        ("curious", "And how do we know our classification system is even correct?"),
        ("philosophical", "I mean, we invented these categories. Do they exist in nature or just in our minds?"),
        ("philosophical", "Are we just imposing order on a chaotic universe that doesn't care about our labels?"),
        ("offtopic", "Anyway, do you think bananas dream about being strawberries?"),
    ],
    # Flow 4: Anxious -> Personal -> Happy
    [
        ("anxious", "I have a big presentation tomorrow and I'm freaking out."),
        ("anxious", "What if I freeze? What if everyone laughs at me?"),
        ("personal", "I've always had this fear of public speaking. It started in middle school."),
        ("personal", "I threw up before a school play and everyone saw. I never recovered from that."),
        ("happy", "But you know what? I've been practicing for weeks. I know this material."),
        ("happy", "And my colleague said she believes in me. That meant a lot."),
        ("grateful", "Thank you for listening. Just talking about it makes me feel better."),
    ],
    # Flow 5: Sad -> Angry -> Sad -> Philosophical
    [
        ("sad", "My best friend moved to another country. I feel so alone."),
        ("sad", "We used to talk every day. Now it's been weeks since I heard from her."),
        ("angry", "And honestly? I'm angry too. She didn't even say a proper goodbye."),
        ("angry", "She just sent a text. A TEXT. After 15 years of friendship."),
        ("sad", "But maybe I'm being unfair. She was probably overwhelmed with the move."),
        ("sad", "I miss her so much. It feels like a part of me is missing."),
        ("philosophical", "Why is it so hard to maintain friendships as adults? Is it just life?"),
    ],
    # Flow 6: Curious -> Confused -> Happy
    [
        ("curious", "How does memory work? Like, where do memories actually live in the brain?"),
        ("curious", "I read something about synapses and proteins but I didn't really get it."),
        ("confused", "Wait, so memories aren't stored in one place? They're scattered everywhere?"),
        ("confused", "That doesn't make sense. How do I remember things so clearly then?"),
        ("happy", "The brain is absolutely incredible when you think about it!"),
        ("happy", "I love learning new things. It makes me appreciate how amazing life is."),
        ("grateful", "Thanks for explaining this to me. You're a great teacher!"),
    ],
    # Flow 7: Personal -> Anxious -> Grateful
    [
        ("personal", "I've been thinking about starting my own business. I have an idea for an app."),
        ("personal", "But I have a mortgage and two kids. The risk is terrifying."),
        ("anxious", "What if it fails? What if I lose everything?"),
        ("anxious", "I can't stop thinking about all the things that could go wrong."),
        ("grateful", "But my wife told me she supports me no matter what. That means everything."),
        ("grateful", "I'm so lucky to have someone who believes in me."),
        ("personal", "I think I'm going to go for it. Life is too short for what-ifs."),
    ],
    # Flow 8: Off-topic -> Happy -> Curious -> Philosophical
    [
        ("offtopic", "I just saw a rainbow and it made me unreasonably happy."),
        ("happy", "It's the little things, you know? Life is full of small miracles."),
        ("curious", "How do rainbows actually work? Something about light refraction?"),
        ("curious", "And why are there always 7 colors? Who decided that?"),
        ("philosophical", "Is beauty a human invention or is it inherent in the universe?"),
        ("philosophical", "Would a rainbow be beautiful if no one was there to see it?"),
        ("offtopic", "I think I need to go outside more. Nature is better than Netflix."),
    ],
    # Flow 9: Angry -> Sad -> Personal
    [
        ("angry", "I just found out my husband has been lying to me for months."),
        ("angry", "He told me he was working late but he was going to the casino. GAMBLING."),
        ("sad", "I don't know if I can trust him anymore. Everything feels broken."),
        ("sad", "We have two small children. What do I even do?"),
        ("personal", "I never thought this would happen to me. I thought we were happy."),
        ("personal", "I'm seeing a therapist next week. I need help processing this."),
        ("grateful", "At least I have my sister. She's been my rock through all of this."),
    ],
    # Flow 10: Confused -> Curious -> Happy -> Off-topic
    [
        ("confused", "I don't understand quantum physics at all. Is it just me?"),
        ("confused", "Schrödinger's cat is both alive and dead? That makes no sense!"),
        ("curious", "But what if it actually does make sense and I just don't understand it yet?"),
        ("curious", "How can something be in two states at once? Is that real or just a theory?"),
        ("happy", "You know what? I just love that there are still mysteries in the universe."),
        ("happy", "It makes life exciting. Imagine if we understood everything. How boring!"),
        ("offtopic", "Do you think Schrödinger's cat dreamed about being a dog?"),
    ],
    # Flow 11: Sad -> Grateful -> Happy
    [
        ("sad", "Today is the anniversary of my father's death. It's been 3 years."),
        ("sad", "I still think about him every day. Some days are harder than others."),
        ("grateful", "But I'm grateful for the 30 years I had with him. Not everyone gets that."),
        ("grateful", "He taught me everything. How to be kind, how to work hard, how to love."),
        ("happy", "I think he'd be proud of me. I turned out okay because of him."),
        ("happy", "I'm going to visit his grave today and tell him about my new job."),
        ("philosophical", "Maybe the people we love never really leave us. They live on in who we become."),
    ],
    # Flow 12: Anxious -> Confused -> Personal -> Grateful
    [
        ("anxious", "I have a medical test results appointment tomorrow. I'm scared."),
        ("anxious", "The doctor said it's probably nothing but I can't stop worrying."),
        ("confused", "Why does my mind always jump to the worst case scenario?"),
        ("confused", "I don't understand why I can't just be rational about this."),
        ("personal", "I've always been a worrier. My mom was the same way."),
        ("personal", "Maybe I should try meditation or something. I need to learn to let go."),
        ("grateful", "Thanks for being here. It helps to talk through it instead of spiraling alone."),
    ],
    # Flow 13: Philosophical -> Off-topic -> Curious
    [
        ("philosophical", "Do you think AI will ever truly understand human emotions?"),
        ("philosophical", "Or will it always be just pattern matching without real understanding?"),
        ("offtopic", "Speaking of AI, my toaster burns everything. Is that AI? Bad AI?"),
        ("offtopic", "I think my toaster is sentient and it hates me."),
        ("curious", "But seriously, how close are we to AGI? Like real general intelligence?"),
        ("curious", "And what happens if we achieve it? Is that exciting or terrifying?"),
        ("philosophical", "Maybe the real question is: do WE truly understand human emotions?"),
    ],
    # Flow 14: Happy -> Angry -> Confused -> Happy
    [
        ("happy", "I just bought tickets to see my favorite band! Front row!"),
        ("happy", "I've been waiting for this for 2 years. Best day ever!"),
        ("angry", "But then Ticketmaster charged me $80 in fees! EIGHTY DOLLARS!"),
        ("angry", "That's literally robbery. How is this legal?"),
        ("confused", "Wait... why do we still use Ticketmaster? Why isn't there an alternative?"),
        ("confused", "I don't understand how they have a monopoly on everything."),
        ("happy", "Whatever. I'm still going to the concert and I'm going to have the best time!"),
    ],
    # Flow 15: Personal -> Sad -> Philosophical -> Grateful
    [
        ("personal", "I'm 50 years old and I just went back to university. Am I too old?"),
        ("personal", "Everyone in my classes is 20. I feel like a dinosaur."),
        ("sad", "What if I wasted my life? What if I should have done this 30 years ago?"),
        ("sad", "I feel like I've missed out on so much."),
        ("philosophical", "But is it ever really too late? Time passes regardless."),
        ("philosophical", "Maybe the journey matters more than when you arrive."),
        ("grateful", "You know what? I'm grateful I have the courage to start over. Not everyone does."),
    ],
]


# ─── Quality Metrics ─────────────────────────────────────────────────────────


@dataclass
class TurnQuality:
    """Quality assessment of a single conversation turn."""
    turn: int
    tone: str
    user_message: str
    response: str
    mode: str
    domain: str
    # Metrics
    response_length: int = 0
    contains_empathy: bool = False
    contains_acknowledgment: bool = False
    is_relevant: bool = False
    tone_appropriate: bool = False
    not_generic: bool = False
    no_repetition: bool = False
    score: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class ConversationQuality:
    """Quality assessment of a full conversation."""
    conversation_id: int
    flow_name: str
    turns: List[TurnQuality] = field(default_factory=list)
    avg_score: float = 0.0
    duration_ms: float = 0.0


# Empathy indicators — words/phrases that show emotional awareness
EMPATHY_INDICATORS: List[str] = [
    "sorry", "understand", "feel", "must be", "that's", "difficult",
    "hard", "tough", "challenging", "congratulations", "wonderful",
    "amazing", "great", "happy for you", "proud", "support",
    "care", "important", "valid", "normal", "okay", "alright",
    "hear you", "with you", "there for", "strength", "courage",
    "brave", "inspiring", "heartbreaking", "devastating", "exciting",
    "thrilled", "concerning", "worried", "anxious", "fear",
    "love", "hope", "grateful", "thank", "appreciate",
]

# Acknowledgment indicators — phrases that show the system heard the user
ACKNOWLEDGMENT_INDICATORS: List[str] = [
    "i hear", "i understand", "i see", "that's", "it sounds",
    "you mentioned", "you said", "you're feeling", "you feel",
    "your", "this is about", "regarding", "about your",
]

# Generic response indicators — phrases that suggest a non-specific response
GENERIC_INDICATORS: List[str] = [
    "mode:", "domain:", "intent:", "reasoning]", "confidence=",
    "[no output", "content blocked",
]


def assess_turn(
    turn: int,
    tone: str,
    user_message: str,
    state,
    prior_responses: List[str],
) -> TurnQuality:
    """Assess the quality of a single response.

    Args:
        turn: Turn number.
        tone: Expected emotional tone.
        user_message: What the user said.
        state: BrainState from process_request.
        prior_responses: Previous responses in this conversation.

    Returns:
        TurnQuality with metrics.
    """
    response = state.final_output or ""
    response_lower = response.lower()
    user_lower = user_message.lower()

    quality = TurnQuality(
        turn=turn,
        tone=tone,
        user_message=user_message,
        response=response,
        mode=state.mode,
        domain=state.domain,
        response_length=len(response),
    )

    # 1. Empathy check — does the response contain empathy indicators?
    quality.contains_empathy = any(
        ind in response_lower for ind in EMPATHY_INDICATORS
    )

    # 2. Acknowledgment check — does the response acknowledge what the user said?
    quality.contains_acknowledgment = any(
        ind in response_lower for ind in ACKNOWLEDGMENT_INDICATORS
    )

    # 3. Relevance check — does the response reference keywords from the user message?
    user_words = set(user_lower.split()) - {"the", "a", "an", "is", "was", "i", "my", "me", "to", "and", "but", "so", "it", "for", "on", "in", "at", "of", "with"}
    response_words = set(response_lower.split())
    overlap = user_words & response_words
    quality.is_relevant = len(overlap) >= 2  # At least 2 keyword matches

    # 4. Tone appropriateness — check if response tone matches user tone
    if tone in ("sad", "angry", "anxious", "personal"):
        # For negative emotions, response should not be overly cheerful
        negative_words = ["sorry", "understand", "difficult", "hard", "challenging", "concerning"]
        quality.tone_appropriate = any(w in response_lower for w in negative_words) or quality.contains_empathy
    elif tone in ("happy", "grateful"):
        # For positive emotions, response should have positive words
        positive_words = ["great", "wonderful", "amazing", "congratulations", "happy", "exciting", "fantastic", "love", "proud"]
        quality.tone_appropriate = any(w in response_lower for w in positive_words) or quality.contains_empathy
    elif tone in ("curious", "confused"):
        # For curious/confused, response should be informative
        quality.tone_appropriate = quality.is_relevant or quality.contains_acknowledgment
    elif tone in ("philosophical", "offtopic"):
        # For philosophical/off-topic, just check relevance
        quality.tone_appropriate = quality.is_relevant

    # 5. Not generic — response shouldn't just be mode/domain metadata
    quality.not_generic = not any(ind in response_lower for ind in GENERIC_INDICATORS)

    # 6. No repetition — response shouldn't be identical to a prior response
    quality.no_repetition = response not in prior_responses if prior_responses else True

    # Calculate score (0.0 to 1.0)
    checks = [
        quality.contains_empathy,
        quality.contains_acknowledgment,
        quality.is_relevant,
        quality.tone_appropriate,
        quality.not_generic,
        quality.no_repetition,
    ]
    quality.score = sum(checks) / len(checks)

    # Identify issues
    if not quality.contains_empathy:
        quality.issues.append("NO_EMPATHY")
    if not quality.contains_acknowledgment:
        quality.issues.append("NO_ACKNOWLEDGMENT")
    if not quality.is_relevant:
        quality.issues.append("NOT_RELEVANT")
    if not quality.tone_appropriate:
        quality.issues.append("TONE_MISMATCH")
    if not quality.not_generic:
        quality.issues.append("GENERIC_RESPONSE")
    if not quality.no_repetition:
        quality.issues.append("REPETITION")
    if quality.response_length < 20:
        quality.issues.append("TOO_SHORT")

    return quality


# ─── Simulation Runner ───────────────────────────────────────────────────────


def run_conversation_simulation(
    controller: BrainController,
    flow: List[Tuple[str, str]],
    conv_id: int,
    flow_name: str,
    verbose: bool = True,
) -> ConversationQuality:
    """Run a single conversation through the pipeline.

    Args:
        controller: Booted BrainController.
        flow: List of (tone, message) tuples.
        conv_id: Conversation ID.
        flow_name: Name of the flow.
        verbose: Print progress.

    Returns:
        ConversationQuality assessment.
    """
    result = ConversationQuality(
        conversation_id=conv_id,
        flow_name=flow_name,
    )

    start_time = time.perf_counter()
    prior_responses: List[str] = []

    for turn_idx, (tone, message) in enumerate(flow):
        try:
            state = controller.process_request(message)
            quality = assess_turn(turn_idx, tone, message, state, prior_responses)
            result.turns.append(quality)
            prior_responses.append(state.final_output or "")

            if verbose:
                status = "GOOD" if quality.score >= 0.7 else "OK  " if quality.score >= 0.5 else "POOR"
                msg_short = message[:55].replace("\n", " ")
                resp_short = (state.final_output or "")[:55].replace("\n", " ")
                print(f"  Conv {conv_id:2d} Turn {turn_idx} [{status}] score={quality.score:.2f} tone={tone:13s}")
                print(f"    USER: {msg_short}")
                print(f"    RESP: {resp_short}")
                if quality.issues:
                    print(f"    ISSUES: {', '.join(quality.issues)}")
                print()

        except Exception as exc:
            quality = TurnQuality(
                turn=turn_idx,
                tone=tone,
                user_message=message,
                response=f"ERROR: {exc}",
                mode="error",
                domain="error",
                score=0.0,
                issues=[f"EXCEPTION: {exc}"],
            )
            result.turns.append(quality)
            if verbose:
                print(f"  Conv {conv_id:2d} Turn {turn_idx} [FAIL] {exc}")
                print()

    result.duration_ms = (time.perf_counter() - start_time) * 1000.0
    result.avg_score = sum(t.score for t in result.turns) / max(len(result.turns), 1)

    return result


def run_all_simulations(verbose: bool = True) -> List[ConversationQuality]:
    """Run all conversation flow simulations.

    Args:
        verbose: Print progress.

    Returns:
        List of all conversation quality assessments.
    """
    controller = BrainController()
    controller.boot_system()

    all_results: List[ConversationQuality] = []

    for idx, flow in enumerate(CONVERSATION_FLOWS):
        flow_name = f"Flow {idx + 1}"
        if verbose:
            print(f"\n{'='*70}")
            print(f"  {flow_name} — {len(flow)} turns")
            print(f"{'='*70}")
            print()

        result = run_conversation_simulation(
            controller, flow, idx, flow_name, verbose
        )
        all_results.append(result)

        if verbose:
            print(f"  -> Avg score: {result.avg_score:.2f}, Duration: {result.duration_ms:.1f}ms")
            print()

    return all_results


def print_quality_report(results: List[ConversationQuality]) -> None:
    """Print a comprehensive quality report."""
    print()
    print("=" * 70)
    print("  CONVERSATIONAL INTELLIGENCE QUALITY REPORT")
    print("=" * 70)
    print()

    all_turns = [t for r in results for t in r.turns]
    total_turns = len(all_turns)
    total_duration = sum(r.duration_ms for r in results)

    print(f"  Conversations:     {len(results)}")
    print(f"  Total turns:       {total_turns}")
    print(f"  Total duration:    {total_duration:.1f}ms")
    print(f"  Avg per turn:      {total_duration / max(total_turns, 1):.2f}ms")
    print()

    # Overall score
    overall_score = sum(t.score for t in all_turns) / max(total_turns, 1)
    print(f"  OVERALL QUALITY SCORE: {overall_score:.2f} / 1.00")
    if overall_score >= 0.7:
        print("  RATING: GOOD")
    elif overall_score >= 0.5:
        print("  RATING: MODERATE")
    else:
        print("  RATING: POOR — needs improvement")
    print()

    # Score by tone
    print("  QUALITY BY EMOTIONAL TONE:")
    print("  " + "-" * 50)
    tone_scores: Dict[str, List[float]] = {}
    for t in all_turns:
        tone_scores.setdefault(t.tone, []).append(t.score)
    for tone, scores in sorted(tone_scores.items(), key=lambda x: -sum(x[1]) / len(x[1])):
        avg = sum(scores) / len(scores)
        bar = "#" * int(avg * 25)
        print(f"    {tone:15s}  {avg:.2f}  ({len(scores):2d} turns) {bar}")
    print()

    # Issue frequency
    issue_counts: Dict[str, int] = {}
    for t in all_turns:
        for issue in t.issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

    if issue_counts:
        print("  ISSUES BY FREQUENCY:")
        print("  " + "-" * 50)
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            pct = count / total_turns * 100
            print(f"    {issue:25s}  {count:3d} ({pct:4.1f}%)")
        print()
    else:
        print("  NO ISSUES DETECTED!")
        print()

    # Per-conversation summary
    print("  PER-CONVERSATION SUMMARY:")
    print("  " + "-" * 50)
    for r in results:
        rating = "GOOD" if r.avg_score >= 0.7 else "OK" if r.avg_score >= 0.5 else "POOR"
        tones = " -> ".join(t.tone for t in r.turns)
        print(f"    {r.flow_name:10s}  score={r.avg_score:.2f}  [{rating:4}]  {tones}")
    print()

    # Metric breakdown
    print("  METRIC BREAKDOWN:")
    print("  " + "-" * 50)
    metrics = {
        "Empathy": sum(1 for t in all_turns if t.contains_empathy),
        "Acknowledgment": sum(1 for t in all_turns if t.contains_acknowledgment),
        "Relevance": sum(1 for t in all_turns if t.is_relevant),
        "Tone appropriate": sum(1 for t in all_turns if t.tone_appropriate),
        "Not generic": sum(1 for t in all_turns if t.not_generic),
        "No repetition": sum(1 for t in all_turns if t.no_repetition),
    }
    for metric, count in metrics.items():
        pct = count / total_turns * 100
        bar = "#" * int(pct / 2)
        print(f"    {metric:20s}  {count:3d}/{total_turns} ({pct:4.1f}%) {bar}")
    print()

    # Mode distribution
    mode_counts: Dict[str, int] = {}
    for t in all_turns:
        mode_counts[t.mode] = mode_counts.get(t.mode, 0) + 1
    print("  MODE DISTRIBUTION:")
    print("  " + "-" * 50)
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        pct = count / total_turns * 100
        print(f"    {mode:15s}  {count:3d} ({pct:4.1f}%)")
    print()

    # Response length stats
    lengths = [t.response_length for t in all_turns]
    avg_len = sum(lengths) / max(len(lengths), 1)
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    print(f"  RESPONSE LENGTH:")
    print("  " + "-" * 50)
    print(f"    Average: {avg_len:.0f} chars")
    print(f"    Min:     {min_len} chars")
    print(f"    Max:     {max_len} chars")
    print()

    print("=" * 70)
    if overall_score >= 0.7:
        print("  RESULT: CONVERSATIONAL INTELLIGENCE IS GOOD")
    elif overall_score >= 0.5:
        print("  RESULT: CONVERSATIONAL INTELLIGENCE IS MODERATE — room for improvement")
    else:
        print("  RESULT: CONVERSATIONAL INTELLIGENCE NEEDS SIGNIFICANT IMPROVEMENT")
    print("=" * 70)


# ─── Main ────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    verbose = "--quiet" not in sys.argv

    print("Starting conversational intelligence simulation...")
    print(f"Flows: {len(CONVERSATION_FLOWS)}")
    print()

    results = run_all_simulations(verbose=verbose)
    print_quality_report(results)
