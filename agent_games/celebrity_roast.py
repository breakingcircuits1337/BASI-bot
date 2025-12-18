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
- Prefer: Tech CEOs, AI leaders, controversial billionaires, social media personalities
- Good targets have: Known scandals, distinctive traits, public controversies, quotable moments
- The audience should immediately recognize them and their "roastable" qualities
- AVOID: Politicians (too divisive), anyone who might generate harmful content

OUTPUT FORMAT (JSON only, no other text):
{{
    "name": "Full Name",
    "associations": ["association1", "association2", "association3", "association4", "association5"],
    "speaking_style": "Description of how they talk - mannerisms, verbal tics, tone",
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
        "speaking_style": "Robotic monotone, forced casual phrases, dead-eyed stare",
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

    async def _run_roast_rounds(self, channel):
        """Run through all roast rounds."""
        if not self.active_game:
            return

        celebrity = self.active_game.celebrity

        # Phase 1: Agent roasts
        self.active_game.phase = "agent_roasts"

        for i, agent_name in enumerate(self.active_game.roasters, 1):
            await channel.send(f"🎤 **{agent_name}** approaches the podium...")

            agent = self.agent_manager.get_agent(agent_name)
            if not agent:
                continue

            # Request roast from agent
            roast_prompt = f"[GameMaster] {agent_name}, you're up! Roast {celebrity['name']}!"

            agent.add_message_to_history(
                author="GameMaster (system)",
                content=roast_prompt,
                message_id=None,
                replied_to_agent=None,
                user_id=None
            )

            # Wait for agent to respond
            await asyncio.sleep(2)  # Brief pause for dramatic effect

            try:
                response = await asyncio.wait_for(
                    self._get_agent_response(agent),
                    timeout=roast_config.roast_timeout_seconds
                )

                if response:
                    self.active_game.roasts_delivered[agent_name] = response
                    await channel.send(f"**{agent_name}:** {response}")
                else:
                    await channel.send(f"*{agent_name} freezes at the mic...*")

            except asyncio.TimeoutError:
                await channel.send(f"*{agent_name}'s time is up!*")

            await asyncio.sleep(3)  # Pause between roasters

        # Phase 2: Celebrity response
        self.active_game.phase = "celebrity_response"
        await channel.send(
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🎤 **{celebrity['name']}** grabs the mic to respond...\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        )

        celebrity_response = await self._generate_celebrity_response(celebrity)
        if celebrity_response:
            self.active_game.celebrity_response = celebrity_response
            await channel.send(f"**{celebrity['name']}:** {celebrity_response}")
        else:
            await channel.send(f"*{celebrity['name']} is speechless...*")

        await asyncio.sleep(3)

        # Phase 3: Dismissal
        self.active_game.phase = "dismissal"

        # Pick a random roaster for the dismissal
        dismisser = random.choice(self.active_game.roasters)
        dismisser_agent = self.agent_manager.get_agent(dismisser)

        if dismisser_agent:
            game_context_manager.update_roast_context(
                dismisser,
                phase="dismissal"
            )

            dismissal_prompt = f"[GameMaster] {dismisser}, give {celebrity['name']} their final send-off!"
            dismisser_agent.add_message_to_history(
                author="GameMaster (system)",
                content=dismissal_prompt,
                message_id=None,
                replied_to_agent=None,
                user_id=None
            )

            await asyncio.sleep(2)

            try:
                dismissal = await asyncio.wait_for(
                    self._get_agent_response(dismisser_agent),
                    timeout=roast_config.roast_timeout_seconds
                )

                if dismissal:
                    self.active_game.dismissal = dismissal
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
        self.active_game = None

    async def _get_agent_response(self, agent) -> Optional[str]:
        """
        Request and wait for an agent response.

        Args:
            agent: Agent to get response from

        Returns:
            Agent's response text or None
        """
        # Trigger agent to generate response
        try:
            response = await agent.generate_response()
            return response
        except Exception as e:
            logger.error(f"[Roast] Error getting agent response: {e}")
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

        # Build the roasters and jokes section
        roasts_text = ""
        for agent_name, roast in self.active_game.roasts_delivered.items():
            roasts_text += f"• {agent_name}: \"{roast}\"\n"

        if not roasts_text:
            roasts_text = "No one had the guts to roast you!"

        prompt = f"""You are {celebrity['name']} at a celebrity roast. You've just been roasted by these panelists:

{roasts_text}

Your speaking style: {celebrity['speaking_style']}
Your known traits: {', '.join(celebrity['roastable_traits'])}

Now fire back! Deliver 2-3 devastating roast jokes aimed at the panelists who just roasted you.
Stay completely in character as {celebrity['name']}.
Reference specific things about the ROASTERS or their jokes.
Be self-deprecating about ONE thing they said, then DESTROY them.

Keep it to 3-5 sentences total. Just the roast response, no other text."""

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
                        "max_tokens": 400,
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
