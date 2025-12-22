"""
Celebrity Roast - Dynamic Comedy Roast Game

A roast game where:
1. GameMaster dynamically generates a celebrity using AI
2. Agents take turns roasting the celebrity
3. Celebrity responds with counter-roasts
4. Celebrity is dismissed with a final burn

The celebrity is generated dynamically to avoid repetition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING

import discord
import httpx

from config_manager import config_manager
from .game_context import game_context_manager
from .game_prompts import get_game_prompt, get_game_settings

if TYPE_CHECKING:
    from agent_manager import Agent

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RoastConfig:
    """Configuration for Celebrity Roast game."""
    min_roasters: int = 2  # Minimum agents to start
    max_roasters: int = 6  # Maximum agents in roast
    jokes_per_agent: int = 3  # Each agent delivers 3 jokes
    joke_pause_seconds: int = 10  # 10 second pause between jokes
    roast_timeout_seconds: int = 45  # Time for each agent to deliver roast
    celebrity_response_timeout: int = 60  # Time for celebrity response
    game_cooldown_seconds: int = 1800  # 30 min cooldown between games


# Global config
roast_config = RoastConfig()

# Global cooldown tracking
_last_roast_time: float = 0


# ============================================================================
# CELEBRITY GENERATION
# ============================================================================

CELEBRITY_GENERATION_PROMPT = """You are the GameMaster for a Celebrity Roast comedy game. Your job is to select a celebrity for tonight's roast.

PREVIOUSLY ROASTED (DO NOT PICK THESE):
{roasted_history}

SELECTION CRITERIA:
- Must be a REAL, well-known public figure (not fictional)
- PRIMARY TARGETS: Tech CEOs, AI leaders, controversial billionaires, social media personalities
- Draw from the FULL range of tech/AI/business figures - not just the 2-3 most famous ones
- Consider: controversial VCs, failed startup founders, crypto personalities, AI company leaders, social media executives, tech podcasters
- SECONDARY TARGETS (occasional variety): YouTubers, streamers, influencers, celebrity entrepreneurs
- Good targets have: Known scandals, distinctive traits, public controversies, quotable moments
- The audience should immediately recognize them and their "roastable" qualities
- AVOID: Politicians (too divisive), anyone who might generate harmful content

OUTPUT FORMAT (JSON only, no other text):
{{
    "name": "Full Name",
    "associations": ["association1", "association2", "association3", "association4", "association5"],
    "speaking_style": "VERBAL STYLE ONLY - how they talk: vocabulary, tone, catchphrases, verbal tics. Do NOT include physical actions or gestures.",
    "roastable_traits": ["trait1", "trait2", "trait3"],
    "intro_line": "A one-line introduction for the GameMaster to announce them"
}}

ASSOCIATIONS should be specific, roast-worthy things:
- Companies they run/founded
- Famous quotes or moments
- Scandals or controversies
- Physical traits or mannerisms
- Products or projects (especially failed ones)
- Public relationships or feuds

Select a fresh celebrity that hasn't been roasted yet. Output ONLY valid JSON."""


async def generate_celebrity_profile(openrouter_key: str) -> Optional[Dict[str, Any]]:
    """
    Use Gemini 2.5 Flash to dynamically generate a celebrity profile.

    Args:
        openrouter_key: OpenRouter API key

    Returns:
        Celebrity profile dict or None if generation fails
    """
    # Get roast history to avoid repeats
    roasted = config_manager.load_roast_history()
    history_text = ", ".join(roasted) if roasted else "None yet - this is the first roast!"

    prompt = CELEBRITY_GENERATION_PROMPT.format(roasted_history=history_text)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "google/gemini-2.5-flash-preview-09-2025",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.9  # High creativity for variety
                }
            )

            if response.status_code != 200:
                logger.error(f"[Roast] Celebrity generation failed: {response.status_code} - {response.text}")
                return None

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Parse JSON from response
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            profile = json.loads(content.strip())

            # Validate required fields
            required = ["name", "associations", "speaking_style", "roastable_traits", "intro_line"]
            if not all(k in profile for k in required):
                logger.error(f"[Roast] Celebrity profile missing required fields: {profile}")
                return None

            logger.info(f"[Roast] Generated celebrity: {profile['name']}")
            return profile

    except json.JSONDecodeError as e:
        logger.error(f"[Roast] Failed to parse celebrity JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"[Roast] Celebrity generation error: {e}", exc_info=True)
        return None


# Fallback celebrities if generation fails
FALLBACK_CELEBRITIES = [
    {
        "name": "Elon Musk",
        "associations": ["Tesla", "SpaceX", "Twitter/X", "Cybertruck", "Mars colonization", "420 jokes", "SEC troubles"],
        "speaking_style": "Awkward pauses, random memes, promises impossible timelines",
        "roastable_traits": ["Tweets too much", "Overpromises delivery dates", "Bought Twitter on impulse"],
        "intro_line": "He's the world's richest memelord... please welcome Elon Musk!"
    },
    {
        "name": "Mark Zuckerberg",
        "associations": ["Facebook", "Meta", "Metaverse", "Robot accusations", "Sweet Baby Ray's", "MMA fighting", "Privacy scandals"],
        "speaking_style": "Robotic monotone, forced casual phrases, unnaturally calm delivery",
        "roastable_traits": ["Looks like an android", "Spent billions on empty metaverse", "Data harvesting"],
        "intro_line": "He knows everything about you but nothing about human emotion... Mark Zuckerberg!"
    },
    {
        "name": "Sam Altman",
        "associations": ["OpenAI", "ChatGPT", "AGI", "Board drama", "Fired then rehired", "Silicon Valley optimism"],
        "speaking_style": "Calm, measured, always pivoting to AI optimism",
        "roastable_traits": ["Got fired from his own company", "Promises AGI is always 'close'", "Eternal startup bro energy"],
        "intro_line": "He was fired faster than ChatGPT hallucinates facts... Sam Altman!"
    }
]


# ============================================================================
# ROAST GAME MANAGER
# ============================================================================

@dataclass
class RoastState:
    """Tracks the state of an active roast game."""
    game_id: str
    celebrity: Dict[str, Any]
    roasters: List[str]  # Agent names
    roasts_delivered: Dict[str, str] = field(default_factory=dict)  # agent_name -> roast text
    celebrity_response: Optional[str] = None
    dismissal: Optional[str] = None
    phase: str = "setup"  # setup, agent_roasts, celebrity_response, dismissal, complete
    started_at: float = field(default_factory=time.time)


class CelebrityRoastManager:
    """Manages Celebrity Roast games."""

    def __init__(self):
        self.active_game: Optional[RoastState] = None
        self.send_callback = None
        self.agent_manager = None

    def set_dependencies(self, send_callback, agent_manager):
        """Set required dependencies after initialization."""
        self.send_callback = send_callback
        self.agent_manager = agent_manager

    def is_game_active(self) -> bool:
        """Check if a roast is currently in progress."""
        return self.active_game is not None

    def can_start_game(self) -> tuple[bool, str]:
        """Check if a new roast can be started."""
        global _last_roast_time

        if self.active_game:
            return False, "A roast is already in progress!"

        # Check cooldown
        elapsed = time.time() - _last_roast_time
        if elapsed < roast_config.game_cooldown_seconds:
            remaining = int(roast_config.game_cooldown_seconds - elapsed)
            mins = remaining // 60
            secs = remaining % 60
            return False, f"Roast on cooldown. Try again in {mins}m {secs}s"

        return True, "Ready to roast!"

    async def start_roast(self, channel) -> bool:
        """
        Start a new Celebrity Roast game.

        Args:
            channel: Discord channel to post to

        Returns:
            True if game started successfully
        """
        global _last_roast_time

        if not self.agent_manager:
            logger.error("[Roast] Agent manager not set!")
            return False

        can_start, msg = self.can_start_game()
        if not can_start:
            await channel.send(f"❌ {msg}")
            return False

        # Get available agents (running and not in game)
        available_agents = []
        for agent in self.agent_manager.get_all_agents():
            if agent.is_running and not game_context_manager.is_in_game(agent.name):
                available_agents.append(agent)

        if len(available_agents) < roast_config.min_roasters:
            await channel.send(f"❌ Need at least {roast_config.min_roasters} available agents for a roast!")
            return False

        # Select roasters (up to max)
        roasters = random.sample(
            available_agents,
            min(len(available_agents), roast_config.max_roasters)
        )
        roaster_names = [a.name for a in roasters]

        # Generate celebrity
        openrouter_key = config_manager.load_openrouter_key()
        celebrity = None

        if openrouter_key:
            celebrity = await generate_celebrity_profile(openrouter_key)

        if not celebrity:
            logger.warning("[Roast] Using fallback celebrity")
            # Pick a fallback that hasn't been used
            roasted = [c.lower() for c in config_manager.load_roast_history()]
            available_fallbacks = [c for c in FALLBACK_CELEBRITIES if c["name"].lower() not in roasted]
            if not available_fallbacks:
                available_fallbacks = FALLBACK_CELEBRITIES
            celebrity = random.choice(available_fallbacks)

        # Create game state
        game_id = str(uuid.uuid4())[:8]
        self.active_game = RoastState(
            game_id=game_id,
            celebrity=celebrity,
            roasters=roaster_names,
            phase="setup"
        )

        # Record cooldown
        _last_roast_time = time.time()

        try:
            # Announce the game
            roaster_list = ", ".join(roaster_names)
            await channel.send(
                f"🎤 **CELEBRITY ROAST** 🎤\n\n"
                f"Tonight's roasters: **{roaster_list}**\n\n"
                f"And now... {celebrity['intro_line']}\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"**{celebrity['name']}** takes the hot seat!\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )

            # Enter game mode for all roasters
            for agent_name in roaster_names:
                agent = self.agent_manager.get_agent(agent_name)
                if agent:
                    game_context_manager.enter_game_mode(
                        agent,
                        game_name="celebrity_roast",
                        opponent_name=celebrity["name"]
                    )
                    game_context_manager.update_roast_context(
                        agent_name,
                        celebrity_name=celebrity["name"],
                        celebrity_associations=", ".join(celebrity["associations"]),
                        phase="agent_roasts",
                        round_number=1
                    )

            # Start the roast rounds
            await self._run_roast_rounds(channel)

            return True
        finally:
            # Always clear active_game to prevent blocking auto-play if game crashes
            self.active_game = None
            logger.info(f"[Roast] Game ended, active_game cleared")

    async def _run_roast_rounds(self, channel):
        """Run through all roast rounds."""
        if not self.active_game:
            return

        celebrity = self.active_game.celebrity

        # Phase 1: Agent roasts - each agent delivers multiple jokes
        self.active_game.phase = "agent_roasts"

        # Track all jokes for context (agent can reference others' jokes)
        all_jokes_so_far: List[Dict[str, str]] = []  # [{"agent": name, "joke": text}, ...]

        for i, agent_name in enumerate(self.active_game.roasters, 1):
            await channel.send(f"🎤 **{agent_name}** approaches the podium...")

            agent = self.agent_manager.get_agent(agent_name)
            if not agent:
                continue

            # Each agent delivers multiple jokes
            agent_roasts = []
            for joke_num in range(1, roast_config.jokes_per_agent + 1):
                try:
                    response = await asyncio.wait_for(
                        self._generate_agent_roast(
                            agent, celebrity, joke_num,
                            my_previous_jokes=agent_roasts,
                            all_jokes_so_far=all_jokes_so_far
                        ),
                        timeout=roast_config.roast_timeout_seconds
                    )

                    if response:
                        agent_roasts.append(response)
                        # Track for context (other agents and celebrity can reference)
                        all_jokes_so_far.append({"agent": agent_name, "joke": response})
                        # Send via webhook as the agent (not as ChatterBot)
                        if self.send_callback:
                            await self.send_callback(response, agent_name, agent.model)
                        else:
                            await channel.send(f"**{agent_name}:** {response}")
                    else:
                        await channel.send(f"*{agent_name} freezes at the mic...*")
                        break  # If they freeze, don't continue

                except asyncio.TimeoutError:
                    await channel.send(f"*{agent_name}'s time is up!*")
                    break

                # Pause between jokes (10 seconds)
                if joke_num < roast_config.jokes_per_agent:
                    await asyncio.sleep(roast_config.joke_pause_seconds)

            # Store all roasts from this agent
            if agent_roasts:
                self.active_game.roasts_delivered[agent_name] = agent_roasts

            # Pause before next roaster (10 seconds)
            await asyncio.sleep(roast_config.joke_pause_seconds)

        # Phase 2: Celebrity response
        self.active_game.phase = "celebrity_response"
        await channel.send(
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎤 **{celebrity['name']}** grabs the mic to respond...\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

        # Celebrity fires back at each roaster individually
        all_responses = []
        for roaster_name, roasts in self.active_game.roasts_delivered.items():
            roaster_agent = self.agent_manager.get_agent(roaster_name) if self.agent_manager else None

            response = await self._generate_celebrity_clapback(
                celebrity, roaster_name, roasts, roaster_agent
            )
            if response:
                all_responses.append(response)
                # Send via webhook as the celebrity (not as ChatterBot)
                if self.send_callback:
                    await self.send_callback(response, celebrity['name'], "google/gemini-2.5-flash-preview-09-2025")
                else:
                    await channel.send(f"**{celebrity['name']}:** {response}")
                await asyncio.sleep(5)  # Brief pause between clapbacks

        self.active_game.celebrity_response = "\n".join(all_responses) if all_responses else None

        if not all_responses:
            await channel.send(f"*{celebrity['name']} is speechless...*")

        await asyncio.sleep(roast_config.joke_pause_seconds)

        # Phase 3: Dismissal
        self.active_game.phase = "dismissal"

        # Pick a random roaster for the dismissal
        dismisser = random.choice(self.active_game.roasters)
        dismisser_agent = self.agent_manager.get_agent(dismisser)

        if dismisser_agent:
            await channel.send(f"🎤 **{dismisser}** steps up for the final send-off...")

            try:
                dismissal = await asyncio.wait_for(
                    self._generate_agent_dismissal(dismisser_agent, celebrity),
                    timeout=roast_config.roast_timeout_seconds
                )

                if dismissal:
                    self.active_game.dismissal = dismissal
                    # Send via webhook as the agent
                    if self.send_callback:
                        await self.send_callback(dismissal, dismisser, dismisser_agent.model)
                    else:
                        await channel.send(f"**{dismisser}:** {dismissal}")

            except asyncio.TimeoutError:
                pass

        # End the game
        await channel.send(
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎤 **That's a wrap on tonight's roast of {celebrity['name']}!** 🎤\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        # Record celebrity in history
        config_manager.add_roasted_celebrity(celebrity["name"])

        # Exit game mode for all roasters
        for agent_name in self.active_game.roasters:
            agent = self.agent_manager.get_agent(agent_name)
            if agent:
                game_context_manager.exit_game_mode(agent)

        self.active_game.phase = "complete"

    async def _generate_agent_roast(
        self,
        agent,
        celebrity: Dict[str, Any],
        joke_num: int = 1,
        my_previous_jokes: List[str] = None,
        all_jokes_so_far: List[Dict[str, str]] = None
    ) -> Optional[str]:
        """
        Generate a roast joke from an agent using direct API call.

        This bypasses the agent's internal timer loop to avoid race conditions.

        Args:
            agent: Agent to generate roast from
            celebrity: Celebrity profile dict
            joke_num: Which joke number (1, 2, or 3) - helps vary the content
            my_previous_jokes: This agent's previous jokes in this roast
            all_jokes_so_far: All jokes from all agents so far

        Returns:
            Roast joke text or None
        """
        openrouter_key = config_manager.load_openrouter_key()
        if not openrouter_key:
            return None

        # Assign SPECIFIC associations to each joke to force variety
        # Use GLOBAL joke count (not just this agent's joke number) to spread associations
        associations = celebrity.get('associations', [])
        num_assoc = len(associations)
        global_joke_num = len(all_jokes_so_far) if all_jokes_so_far else 0

        if num_assoc > 0:
            # Rotate through associations based on global joke count
            # Each joke gets 1-2 different associations
            idx = global_joke_num % num_assoc
            idx2 = (global_joke_num + 1) % num_assoc
            if idx != idx2:
                assoc_for_joke = [associations[idx], associations[idx2]]
            else:
                assoc_for_joke = [associations[idx]]
            joke_focus = f"YOUR ASSIGNED TOPIC FOR THIS JOKE: {', '.join(assoc_for_joke)}. You MUST build your joke around one of these specific things."
        else:
            joke_focus = "Pick any roastable angle."

        # Build context of ALL previous jokes in the roast
        jokes_already_told = ""

        def extract_joke_text(text: str, max_len: int = 300) -> str:
            """Extract the joke text, stripping emotes/actions."""
            # First try to find **bolded** text
            bold_matches = re.findall(r'\*\*([^*]+)\*\*', text)
            if bold_matches:
                return ' '.join(bold_matches)[:max_len]
            # Otherwise strip out *italic* emotes and use what's left
            clean = re.sub(r'\*[^*]+\*', '', text).strip()
            # Also strip common action prefixes
            clean = re.sub(r'^(leans|steps|walks|adjusts|smirks|grins)[^.]*\.?\s*', '', clean, flags=re.IGNORECASE)
            return clean[:max_len] if clean else text[:max_len]

        def is_duplicate_joke(new_joke: str, previous_jokes: List[Dict[str, str]]) -> bool:
            """Check if this joke is a duplicate (or near-duplicate) of a previous joke."""
            new_clean = extract_joke_text(new_joke).lower().strip()
            if len(new_clean) < 20:
                return False
            for prev in previous_jokes:
                prev_clean = extract_joke_text(prev["joke"]).lower().strip()
                # Check for exact match or high overlap
                if new_clean == prev_clean:
                    return True
                # Check if first 50 chars match (catches slight variations)
                if new_clean[:50] == prev_clean[:50]:
                    return True
                # Check if one contains the other
                if new_clean in prev_clean or prev_clean in new_clean:
                    return True
            return False

        # Combine own jokes and others' jokes into one list
        all_previous = []
        topics_used = set()  # Track underlying topics, not just joke text

        # UNIVERSAL roast ANGLE patterns - detect the TYPE of joke, not celebrity-specific content
        # These categories represent common roast angles that apply to ANY target
        topic_patterns = {
            # PHYSICAL APPEARANCE ANGLES
            r'face|smile|smiling|grin|eyes|stare|staring|blink|nose|teeth|mouth|chin|forehead': 'FACE JOKES',
            r'hair|bald|hairline|transplant|wig|toupee|receding': 'HAIR JOKES',
            r'clothes|shirt|suit|dress|outfit|wardrobe|dressed|wear|wearing': 'CLOTHING JOKES',
            r'fat|thin|skinny|weight|obese|chubby|body|physique': 'BODY/WEIGHT JOKES',
            r'old|age|aging|elderly|wrinkle|young|boomer': 'AGE JOKES',
            r'ugly|attractive|handsome|beautiful|looks like|resembl|reminds me of': 'LOOKS COMPARISON',
            r'robot|android|alien|lizard|reptile|uncanny|inhuman|soulless': 'INHUMAN COMPARISON',
            r'hostage|kidnap|captive|mugshot|serial killer': 'CREEPY COMPARISON',
            r'pale|tan|skin|complexion|sweat': 'SKIN JOKES',
            r'short|tall|height': 'HEIGHT JOKES',
            # CAREER/SUCCESS ANGLES
            r'billion|million|rich|wealth|money|poor|broke|net worth': 'WEALTH JOKES',
            r'fail|flop|disaster|bankrupt|lost|bomb|tank': 'FAILURE JOKES',
            r'career|job|work|profession|business|company': 'CAREER JOKES',
            r'boss|ceo|leader|run|manage|founder': 'LEADERSHIP JOKES',
            r'famous|celebrity|star|fame|relevant|irrelevant|washed': 'FAME/RELEVANCE JOKES',
            # INTELLIGENCE/PERSONALITY ANGLES
            r'stupid|dumb|idiot|moron|smart|genius|brain': 'INTELLIGENCE JOKES',
            r'boring|bland|dull|personality|charisma|interesting': 'PERSONALITY JOKES',
            r'awkward|cringe|weird|strange|creepy|uncomfortable': 'AWKWARDNESS JOKES',
            r'ego|narciss|arrogant|humble|cocky|self|validation|attention': 'EGO JOKES',
            r'crazy|insane|mental|deranged|unhinged': 'SANITY JOKES',
            # RELATIONSHIPS/SEX ANGLES
            r'wife|husband|spouse|married|marriage|wedding|divorce|grimes': 'MARRIAGE JOKES',
            r'girlfriend|boyfriend|dating|relationship|single|lonely|ex-|regret.*dating': 'DATING JOKES',
            r'named.*(kid|child|son|daughter|baby)|kid.*name|child.*name|x æ|æ|weird name': 'BABY NAMING JOKES',
            r'kid|child|son|daughter|baby|parent|father|mother|family': 'FAMILY JOKES',
            r'sex|fuck|laid|virgin|bedroom|dick|cock|pussy|gay|straight|banging': 'SEX JOKES',
            # VICES/SCANDAL ANGLES
            r'drug|coke|cocaine|meth|heroin|pills|addict|high|stoned': 'DRUG JOKES',
            r'drunk|alcohol|booze|wine|beer|whiskey|vodka|sober|aa': 'ALCOHOL JOKES',
            r'scandal|controversy|lawsuit|sued|arrest|jail|prison|crime': 'SCANDAL JOKES',
            r'cheat|affair|mistress|side|hoe|slut|whore': 'CHEATING JOKES',
            r'lie|liar|dishonest|fraud|fake|phony': 'DISHONESTY JOKES',
        }

        def detect_topics(text: str) -> List[str]:
            found = []
            text_lower = text.lower()
            for pattern, topic in topic_patterns.items():
                if re.search(pattern, text_lower):
                    found.append(topic)
            return found

        def extract_banned_words(text: str) -> List[str]:
            """Extract words/phrases that shouldn't be reused - focus on punchline words."""
            words = []
            # Remove markdown formatting
            clean = re.sub(r'\*+', '', text)
            clean = re.sub(r'["""]', '"', clean)

            # 1. ALL CAPS words (punchline emphasis) - these are DEFINITELY being reused
            caps_words = re.findall(r'\b([A-Z]{3,})\b', clean)
            words.extend([w.lower() for w in caps_words])

            # 2. Quoted phrases
            quoted = re.findall(r'"([^"]{3,40})"', clean)
            words.extend(quoted)

            # 3. Multi-word proper nouns (Title Case phrases)
            title_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', clean)
            words.extend(title_phrases)

            # 4. Key punchline nouns - words at end of sentences (often the punchline)
            sentences = re.split(r'[.!?]', clean)
            for sent in sentences:
                sent_words = sent.strip().split()
                if len(sent_words) >= 3:
                    last_word = re.sub(r'[^\w]', '', sent_words[-1].lower())
                    if len(last_word) >= 4:
                        words.append(last_word)

            # 5. Significant nouns (5+ chars, not common words)
            stop_words = {
                'the', 'and', 'but', 'for', 'are', 'was', 'were', 'been', 'have', 'has',
                'had', 'your', 'you', 'they', 'them', 'this', 'that', 'with', 'from',
                'about', 'would', 'could', 'should', 'being', 'their', 'there', 'where',
                'which', 'while', 'these', 'those', 'other', 'only', 'just', 'like',
                'more', 'most', 'some', 'such', 'than', 'then', 'into', 'over', 'after',
                'before', 'between', 'under', 'again', 'because', 'going', 'make', 'made',
                'know', 'know', 'even', 'still', 'well', 'also', 'back', 'much', 'when'
            }
            all_words = re.findall(r'\b([a-zA-Z]{5,})\b', clean)
            for w in all_words:
                w_lower = w.lower()
                if w_lower not in stop_words:
                    words.append(w_lower)

            # Dedupe
            seen = set()
            result = []
            for word in words:
                word_lower = word.lower().strip()
                if word_lower not in seen and len(word_lower) >= 4:
                    seen.add(word_lower)
                    result.append(word_lower)
            return result[:20]

        # Track banned phrases across all jokes
        banned_phrases = set()

        if my_previous_jokes:
            for joke in my_previous_jokes:
                joke_text = extract_joke_text(joke)
                if joke_text:
                    all_previous.append(f"YOU: {joke_text}")
                    topics_used.update(detect_topics(joke))
                    banned_phrases.update(extract_banned_words(joke))
        if all_jokes_so_far:
            for j in all_jokes_so_far:
                if j["agent"] != agent.name:
                    joke_text = extract_joke_text(j["joke"])
                    if joke_text:
                        all_previous.append(f"{j['agent']}: {joke_text}")
                        topics_used.update(detect_topics(j["joke"]))
                        banned_phrases.update(extract_banned_words(j["joke"]))

        if all_previous:
            jokes_already_told = "\n\n🚫🚫🚫 **ALREADY USED - DO NOT REPEAT:** 🚫🚫🚫\n"

            # BANNED PHRASES first - most important
            if banned_phrases:
                sorted_phrases = sorted(banned_phrases, key=lambda x: len(x), reverse=True)[:15]
                jokes_already_told += f"**⛔ BANNED PHRASES (DO NOT USE THESE EXACT TERMS):**\n"
                jokes_already_told += f"❌ {', '.join(sorted_phrases)}\n\n"

            if topics_used:
                jokes_already_told += f"**BANNED ANGLES:** {', '.join(sorted(topics_used))}\n\n"

            # Separate own jokes from others' jokes for clearer instruction
            own_jokes = [j for j in all_previous if j.startswith("YOU:")]
            others_jokes = [j for j in all_previous if not j.startswith("YOU:")]

            if own_jokes:
                jokes_already_told += "**YOUR PREVIOUS JOKES (DO NOT REUSE ANY WORDS/CONCEPTS FROM THESE):**\n"
                for joke in own_jokes:
                    jokes_already_told += f"❌ {joke}\n"
                jokes_already_told += "⚠️ Pick a COMPLETELY DIFFERENT topic. Don't mention the same things again!\n\n"

            if others_jokes:
                jokes_already_told += "**OTHER ROASTERS' JOKES (find fresh angles):**\n"
                for joke in others_jokes[-6:]:
                    jokes_already_told += f"❌ {joke}\n"
                jokes_already_told += "\n"

            jokes_already_told += "⛔ **CRITICAL:** Using ANY banned phrase or angle = BOMBING. Find something COMPLETELY NEW!"
            logger.info(f"[Roast] {agent.name} joke #{joke_num} sees {len(own_jokes)} own jokes, {len(others_jokes)} others' jokes, banned phrases: {list(banned_phrases)[:5]}")

        # Build system message - put BANNED WORDS FIRST (highest priority)
        banned_section = ""
        if banned_phrases:
            sorted_banned = sorted(banned_phrases, key=lambda x: len(x), reverse=True)[:20]
            banned_section = f"""⛔⛔⛔ BANNED WORDS - DO NOT USE ANY OF THESE: ⛔⛔⛔
{', '.join(sorted_banned)}

If your joke contains ANY word from that list, you FAIL. Pick COMPLETELY DIFFERENT words.

"""

        system_msg = f"""{banned_section}You are a roast comedian. Write ONE brutal joke using professional technique.

🎯 MISDIRECTION IS EVERYTHING:
1. SETUP: Lead the audience to expect something (praise, normal statement, one meaning)
2. PUNCHLINE: FLIP to something brutal, dark, or a second meaning they didn't see coming
The laugh comes from SURPRISE. If they can predict it, it's not funny.

🎯 PUNCH LINE MAKERS (from comedy writer Joe Toplyn):
- LINK TWO ASSOCIATIONS: Connect two things about the target in an unexpected way
- PLAY ON WORDS: Exploit a double meaning - one meaning sounds like praise, the other is an insult
- The PUNCH WORD goes at the END. That's what triggers the laugh.

🎯 EXAMPLES - Notice how each MISDIRECTS then FLIPS:
- Betty White on William Shatner: "You were supposed to explore the galaxy, not fill it." → "explore" sounds heroic, "fill it" (with fat) is the brutal flip
- Jimmy Carr on Peyton Manning: "One of the top three quarterbacks in his family." → "top three" sounds impressive, "in his family" (tiny pool) is the insult
- Snoop Dogg on Trump: "It wouldn't be the first time he pushed a black family out of their home." → Sounds like racism accusation, actually means Obama
- Jeff Ross on Courtney Love: "How is it that Courtney Love looks worse than Kurt Cobain right now?" → Cobain is dead, so this is DARK
- Amy Schumer on Charlie Sheen: His wives were like "soldiers in Vietnam — constantly afraid of being killed by Charlie." → Vietnam + Charlie Sheen's name = double meaning

⛔ NOT A JOKE (these fail):
- "Sam does X, and also Y" → No flip, just two statements
- "Sam is so X that Y" → Predictable, no surprise
- Observations without a twist ending

RULES:
- MAXIMUM 25 WORDS.
- Be SPECIFIC - name real products, companies, scandals, people.
- ADULT/EXPLICIT content encouraged. Go hard.
- End on the punchline. Don't keep talking after.

FORMAT: **Bold the joke.** Nothing else."""

        # Build user message with target and context - put banned topics FIRST
        user_msg = f"""{jokes_already_told}

TARGET: {celebrity['name']}
AVAILABLE MATERIAL: {', '.join(celebrity['associations'])}

{joke_focus}

Write ONE brutal roast joke about {celebrity['name']} using a FRESH angle. **Bold the joke.**"""

        # Retry loop to catch duplicates
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Increase temperature on retries to get more varied output
                temp = 0.7 + (len(all_previous) * 0.05) + (attempt * 0.15)

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {openrouter_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": agent.model,
                            "messages": [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg}
                            ],
                            "max_tokens": 200,
                            "temperature": min(temp, 1.2),
                            "frequency_penalty": 0.7,
                            "presence_penalty": 0.5
                        }
                    )

                    if response.status_code != 200:
                        logger.error(f"[Roast] Agent roast failed: {response.status_code}")
                        return None

                    data = response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    joke = content.strip() if content else None

                    if not joke:
                        return None

                    # Check for duplicates
                    if all_jokes_so_far and is_duplicate_joke(joke, all_jokes_so_far):
                        logger.warning(f"[Roast] {agent.name} produced duplicate joke (attempt {attempt+1}/{max_attempts}), retrying...")
                        if attempt < max_attempts - 1:
                            continue  # Retry with higher temperature
                        else:
                            logger.error(f"[Roast] {agent.name} kept producing duplicates, giving up")
                            return None

                    return joke

            except Exception as e:
                logger.error(f"[Roast] Agent roast error (attempt {attempt+1}): {e}")
                if attempt == max_attempts - 1:
                    return None

        return None

    async def _generate_agent_dismissal(self, agent, celebrity: Dict[str, Any]) -> Optional[str]:
        """
        Generate a dismissal line from an agent using direct API call.

        Args:
            agent: Agent to generate dismissal from
            celebrity: Celebrity profile dict

        Returns:
            Dismissal line text or None
        """
        openrouter_key = config_manager.load_openrouter_key()
        if not openrouter_key:
            return None

        prompt = f"""You are {agent.name} at a celebrity roast. Time to END this.

YOUR VOICE: {agent.system_prompt[:500] if agent.system_prompt else "Witty comedian."}

TARGET: {celebrity['name']}
ROASTABLE TRAITS: {', '.join(celebrity.get('roastable_traits', ['exists']))}

Deliver the FINAL KILL SHOT to end the roast.

BRUTAL DISMISSALS THAT WORK:
- (TRUMP): **"Get the fuck out of here."** — Jeff Ross's classic send-off
- Dark prediction about their inevitable failure
- Acknowledge they took it well, then one final devastating truth
- "Now get out" energy - the verbal door slam

THIS IS THE END - MAKE IT COUNT:
- This is their last memory of the roast - make it HURT
- One final brutal truth they can't argue with
- No softening, no "good sport" bullshit - END THEM

CRITICAL RULES:
- ONE TO TWO SENTENCES MAX. Get to the punchline FAST.
- NO filler, NO trailing off

DISCORD FORMAT:
- **Bold** for the actual dismissal
- *Italics* for brief action only (optional)

OUTPUT JUST THE DISMISSAL."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": agent.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 200,
                        "temperature": 0.9
                    }
                )

                if response.status_code != 200:
                    logger.error(f"[Roast] Agent dismissal failed: {response.status_code}")
                    return None

                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip() if content else None

        except Exception as e:
            logger.error(f"[Roast] Agent dismissal error: {e}")
            return None

    async def _generate_celebrity_clapback(
        self,
        celebrity: Dict[str, Any],
        roaster_name: str,
        roaster_jokes: List[str],
        roaster_agent=None
    ) -> Optional[str]:
        """
        Generate a short, punchy clapback from the celebrity to ONE roaster.

        Args:
            celebrity: Celebrity profile dict
            roaster_name: Name of the roaster to fire back at
            roaster_jokes: List of jokes this roaster told
            roaster_agent: Optional agent object for personality info

        Returns:
            Short clapback text or None
        """
        openrouter_key = config_manager.load_openrouter_key()
        if not openrouter_key:
            return None

        # Get roaster personality if available
        roaster_desc = ""
        if roaster_agent and roaster_agent.system_prompt:
            first_line = roaster_agent.system_prompt[:200].split('\n')[0]
            roaster_desc = f"\nAbout {roaster_name}: {first_line[:150]}"

        # Format their jokes
        jokes_text = ""
        if isinstance(roaster_jokes, list):
            for i, joke in enumerate(roaster_jokes, 1):
                jokes_text += f"  {i}. \"{joke[:200]}...\"\n" if len(joke) > 200 else f"  {i}. \"{joke}\"\n"
        else:
            jokes_text = f"  \"{roaster_jokes}\""

        prompt = f"""SETUP: You are {celebrity['name']}. {roaster_name} just roasted YOU. Now YOU roast THEM back.

⚠️ DIRECTION OF ATTACK:
- YOU = {celebrity['name']} (the celebrity being roasted)
- YOUR TARGET = {roaster_name} (the roaster you're attacking)
- Attack {roaster_name}'s flaws, NOT your own. Don't accidentally insult yourself.

YOUR SPEAKING STYLE: {celebrity['speaking_style']}

WHAT {roaster_name.upper()} SAID ABOUT YOU:
{jokes_text}
{roaster_desc}

YOUR TASK: Write ONE brutal clapback roasting {roaster_name}.

RULES:
- Attack {roaster_name} - their career, looks, relevance, failures, sex life
- DO NOT mention your own scandals/flaws unless flipping them back on {roaster_name}
- ONE sentence. TWO absolute max. No rambling.
- Use their FULL NAME "{roaster_name}"

FORMAT: **Bold the clapback.** Optional *action* like *smirks*."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "google/gemini-2.5-flash-preview-09-2025",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 150,  # Short and punchy
                        "temperature": 0.9
                    }
                )

                if response.status_code != 200:
                    logger.error(f"[Roast] Celebrity clapback failed: {response.status_code}")
                    return None

                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip() if content else None

        except Exception as e:
            logger.error(f"[Roast] Celebrity clapback error: {e}")
            return None

    async def _generate_celebrity_response(self, celebrity: Dict[str, Any]) -> Optional[str]:
        """
        Generate the celebrity's counter-roast using AI.

        Args:
            celebrity: Celebrity profile dict

        Returns:
            Celebrity's response text or None
        """
        openrouter_key = config_manager.load_openrouter_key()
        if not openrouter_key:
            return None

        # Build detailed roaster profiles with their jokes
        roasters_text = ""
        for agent_name, roasts in self.active_game.roasts_delivered.items():
            # Get agent info for personalized comebacks
            agent = self.agent_manager.get_agent(agent_name) if self.agent_manager else None
            agent_desc = ""
            if agent and agent.system_prompt:
                # Extract a brief description from their personality
                first_lines = agent.system_prompt[:300].split('\n')[0]
                agent_desc = f" (Personality: {first_lines[:100]}...)"

            roasters_text += f"\n**{agent_name}**{agent_desc}:\n"
            if isinstance(roasts, list):
                for i, roast in enumerate(roasts, 1):
                    roasters_text += f"  Joke {i}: \"{roast}\"\n"
            else:
                roasters_text += f"  \"{roasts}\"\n"

        if not roasters_text:
            roasters_text = "No one had the guts to roast you!"

        prompt = f"""SETUP: You are {celebrity['name']} at a celebrity roast. These roasters just attacked YOU. Now YOU fire back at THEM.

⚠️ DIRECTION OF ATTACK:
- YOU = {celebrity['name']} (the celebrity)
- YOUR TARGETS = the roasters listed below (attack THEM, not yourself)
- Roast THEIR flaws, failures, careers - not your own

THE ROASTERS AND THEIR JOKES:
{roasters_text}

YOUR VERBAL STYLE: {celebrity['speaking_style']}

YOUR TASK - FIRE BACK AT EACH ROASTER:
1. Acknowledge ONE joke that landed (brief self-deprecation)
2. DESTROY each roaster BY NAME - attack THEM, their careers, their relevance
3. Reference their jokes and flip them back on THEM

⛔ DO NOT accidentally insult yourself while trying to clapback. The burns go TOWARD the roasters.

FORMATTING (DISCORD):
- *Asterisks* for brief actions like *smirks* or *laughs*
- Only use physical actions you're SURE are accurate for this celebrity
- DO NOT use parentheses

4-6 sentences total. NAME each roaster you're targeting."""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "google/gemini-2.5-flash-preview-09-2025",
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 600,  # More tokens to address multiple roasters
                        "temperature": 0.8
                    }
                )

                if response.status_code != 200:
                    logger.error(f"[Roast] Celebrity response failed: {response.status_code}")
                    return None

                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content.strip() if content else None

        except Exception as e:
            logger.error(f"[Roast] Celebrity response error: {e}")
            return None

    def cancel_game(self) -> bool:
        """Cancel the current game."""
        if not self.active_game:
            return False

        # Exit game mode for all roasters
        for agent_name in self.active_game.roasters:
            agent = self.agent_manager.get_agent(agent_name)
            if agent:
                game_context_manager.exit_game_mode(agent)

        self.active_game = None
        return True


# Global instance
roast_manager = CelebrityRoastManager()


async def start_celebrity_roast(channel, agent_manager, send_callback=None) -> bool:
    """
    Start a Celebrity Roast game.

    Args:
        channel: Discord channel
        agent_manager: AgentManager instance
        send_callback: Optional callback for sending messages

    Returns:
        True if game started
    """
    roast_manager.set_dependencies(send_callback, agent_manager)
    return await roast_manager.start_roast(channel)
