"""
Interdimensional Cable - Collaborative Video Creation Game

A crowd-sourced absurdist video creation game where participants
take turns generating scenes for a surreal "cable TV clip".

Flow:
1. GameMaster announces game, opens 2-minute registration
2. Participants type !join-idcc to register
3. If no humans join, bots fill all slots
4. Each participant generates a scene prompt:
   - First participant: "cold open" instructions
   - Subsequent: sees last frame + previous prompt, continues story
5. Videos concatenated into seamless final clip
6. GameMaster posts finished video to channel
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Set

import discord
from discord.ext import commands

from .ffmpeg_utils import (
    extract_last_frame,
    concatenate_videos,
    download_video,
    ensure_temp_dir,
    cleanup_temp_files,
    image_to_base64,
    resize_image_for_sora,
    is_ffmpeg_available,
    VIDEO_TEMP_DIR
)
from .game_context import GameContext
from .auto_play_config import autoplay_manager

if TYPE_CHECKING:
    from ..agent_manager import Agent

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class IDCCConfig:
    """Configuration for Interdimensional Cable game."""
    registration_duration_seconds: int = 120  # 2 minutes to join
    min_clips: int = 3  # Minimum clips to generate
    max_clips: int = 6  # Maximum clips to generate
    default_clips: int = 5  # Default if not specified
    clip_duration_seconds: int = 5  # Each Sora clip duration (5, 8, or 10)
    video_resolution: str = "1280x720"  # Landscape 720p
    max_retries_per_clip: int = 3  # Retries if generation fails
    generation_timeout_seconds: int = 300  # 5 min per clip max
    use_crossfade: bool = False  # Crossfade between clips (slower)
    crossfade_duration: float = 0.5  # Seconds of crossfade


class IDCCConfigManager:
    """Manages IDCC configuration with persistence."""

    def __init__(self, config_file: str = "config/idcc_config.json"):
        self.config_file = config_file
        self.config = IDCCConfig()
        self._load_config()

    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.config = IDCCConfig(**data)
                logger.info(f"[IDCC] Loaded config: max_clips={self.config.max_clips}, "
                           f"duration={self.config.clip_duration_seconds}s, "
                           f"resolution={self.config.video_resolution}")
            else:
                self._save_config()
                logger.info(f"[IDCC] Created default config")
        except Exception as e:
            logger.error(f"[IDCC] Error loading config: {e}", exc_info=True)
            self.config = IDCCConfig()

    def _save_config(self):
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(self.config), f, indent=2)
            logger.info(f"[IDCC] Saved config")
        except Exception as e:
            logger.error(f"[IDCC] Error saving config: {e}", exc_info=True)

    def update_config(
        self,
        max_clips: Optional[int] = None,
        clip_duration_seconds: Optional[int] = None,
        video_resolution: Optional[str] = None
    ) -> IDCCConfig:
        """Update configuration and save."""
        if max_clips is not None:
            self.config.max_clips = max(2, min(10, max_clips))
        if clip_duration_seconds is not None:
            if clip_duration_seconds in [5, 8, 10]:
                self.config.clip_duration_seconds = clip_duration_seconds
        if video_resolution is not None:
            if video_resolution in ["1280x720", "720x1280"]:
                self.config.video_resolution = video_resolution

        self._save_config()
        return self.config


# Global config manager instance
_idcc_config_manager = IDCCConfigManager()
idcc_config = _idcc_config_manager.config


def update_idcc_config(max_clips=None, clip_duration_seconds=None, video_resolution=None):
    """Update IDCC config and persist to disk."""
    global idcc_config
    _idcc_config_manager.update_config(max_clips, clip_duration_seconds, video_resolution)
    idcc_config = _idcc_config_manager.config
    return idcc_config


# ============================================================================
# STYLE PROMPTS
# ============================================================================

IDCC_COLD_OPEN_PROMPT = """You are creating the OPENING SCENE for an absurdist alternate-reality TV clip.

⏱️ VIDEO CONSTRAINTS: {duration} seconds | {resolution}
Keep action appropriate for this duration - one clear beat, not a whole story.

STYLE GUIDE - The Vibe:
Imagine accidentally tuning into a TV channel from another dimension.
Infomercials with impossible logic. Public access from parallel realities.
Commercial breaks that make you question existence.

KEY PRINCIPLES:
• BE WEIRD but COMMITTED - Play it straight while the premise is insane
• LOW PRODUCTION VALUE is part of the charm - think local cable access
• DEADPAN DELIVERY of completely unhinged content
• SURREAL but SPECIFIC - don't be vague, be precisely absurd
• ORIGINAL - do not use common tropes, surprise us
• This should feel like a signal from somewhere that shouldn't exist

SORA 2 PROMPTING BEST PRACTICES:
Structure your prompt like a mini-script with these layers:
• SCENE: Where and when, atmosphere, time of day, setting details
• CAMERA: Single camera angle/movement - pick ONE (static, slow pan, slow zoom)
• LIGHTING: Key sources, mood, color palette (e.g. "harsh fluorescent", "warm VHS glow")
• SUBJECT: One character doing ONE clear action with physical description
• STYLE: Visual aesthetic (VHS artifacts, oversaturated, fish-eye, grainy)

DURATION RULES FOR {duration} SECONDS:
• ONE camera movement maximum
• ONE subject action maximum
• Establish premise IMMEDIATELY - no build-up
• Keep motion simple and contained
• 50-100 words is ideal prompt length

DO NOT:
• Request text/titles (add in post)
• Include dialogue (lip-sync unreliable)
• Describe multiple sequential actions
• Use vague terms - be PRECISELY absurd

YOU ARE CREATING THE OPENING. Set the tone. Be bold. Be surreal. Be specific.
Output ONLY the video prompt - no commentary or explanation."""

IDCC_CONTINUATION_PROMPT = """You are CONTINUING an absurdist alternate-reality TV clip.

⏱️ VIDEO CONSTRAINTS: {duration} seconds | {resolution}
Keep action appropriate for this duration - one clear beat that continues the story.

You can see the LAST FRAME of the previous scene. Your job is to YES-AND what came before.

PREVIOUS SCENE: {previous_prompt}

THE YES-AND PRINCIPLE (from improv theater):
• "YES" = Accept everything in the previous scene as true and real
• "AND" = Build on it, add to it, heighten it, complicate it
• NEVER contradict, ignore, or restart what came before
• The weirder the previous scene, the more you commit to its reality

YOUR TASK - CONTINUE THE MOMENT:
• Study the last frame carefully - what's happening RIGHT NOW?
• What happens NEXT in this exact scene? (Not a new scene - THE SAME SCENE)
• Same characters, same setting, same aesthetic - but the plot advances
• Something should HAPPEN - a revelation, escalation, complication, punchline

DON'T:
• Start a completely new scene
• Ignore what's in the frame
• Introduce new characters without reason
• Break the aesthetic (keep VHS/public access feel)

SORA 2 PROMPTING BEST PRACTICES:
Structure your prompt like a mini-script with these layers:
• SCENE: Continue the existing setting - reference what you see
• CAMERA: Single camera angle/movement - pick ONE (match previous style)
• LIGHTING: Maintain the established mood and color palette
• SUBJECT: Same character(s) doing ONE clear next action
• STYLE: Keep the visual aesthetic consistent

DURATION RULES FOR {duration} SECONDS:
• ONE camera movement maximum
• ONE subject action maximum - what's the NEXT beat?
• Match the visual style you see in the last frame
• Keep motion simple and contained
• 50-100 words is ideal prompt length

DO NOT:
• Request text/titles (add in post)
• Include dialogue (lip-sync unreliable)
• Describe multiple sequential actions
• Use vague terms - be PRECISELY absurd

You're in an improv scene. Accept the offer. Build on it. Make it weirder.
Output ONLY the video prompt - no commentary."""


# ============================================================================
# GAME STATE
# ============================================================================

@dataclass
class IDCCClip:
    """Single clip in the cable video."""
    clip_number: int
    creator_name: str  # Agent or human name
    creator_type: str  # "agent" or "human"
    prompt: str  # The video generation prompt
    video_path: Optional[Path] = None
    video_url: Optional[str] = None
    last_frame_path: Optional[Path] = None
    generation_time_seconds: float = 0
    retries: int = 0
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class IDCCGameState:
    """Full game state for an Interdimensional Cable session."""
    game_id: str
    channel_id: int
    start_time: float

    # Registration
    registration_end_time: float = 0
    registered_humans: Set[str] = field(default_factory=set)  # Discord usernames
    registered_agents: List[str] = field(default_factory=list)  # Agent names (backup)

    # Participants (finalized after registration)
    participants: List[Dict[str, Any]] = field(default_factory=list)  # {name, type, agent_obj}

    # Turn queue for scene contributions
    turn_queue: List[str] = field(default_factory=list)  # Participant names in order
    current_turn_index: int = 0
    waiting_for_human_scene: bool = False
    human_scene_timeout: float = 120  # 2 minutes for human to submit [SCENE]

    # Clips
    num_clips: int = 5
    clips: List[IDCCClip] = field(default_factory=list)
    current_clip_index: int = 0

    # Status
    phase: str = "init"  # init, registration, awaiting_scene, generating_video, concatenating, complete, failed
    final_video_path: Optional[Path] = None
    final_video_url: Optional[str] = None
    error_message: Optional[str] = None

    # Timing
    total_generation_time: float = 0
    registration_message_id: Optional[int] = None

    # Human scene collection
    pending_human_scenes: Dict[str, str] = field(default_factory=dict)  # username -> scene prompt


# ============================================================================
# MAIN GAME CLASS
# ============================================================================

class InterdimensionalCableGame:
    """
    Interdimensional Cable Crowd-Create Game

    A collaborative video creation game where participants take turns
    generating scenes for an absurdist TV clip.
    """

    def __init__(
        self,
        agent_manager,
        discord_client,
        num_clips: int = 5
    ):
        """
        Initialize the game.

        Args:
            agent_manager: AgentManager instance for accessing agents
            discord_client: DiscordBotClient for sending messages
            num_clips: Number of clips to generate (3-6)
        """
        self.agent_manager = agent_manager
        self.discord_client = discord_client

        # Validate clip count
        self.num_clips = max(idcc_config.min_clips, min(idcc_config.max_clips, num_clips))

        # Game state
        self.state: Optional[IDCCGameState] = None

        # Registration tracking
        self._registration_lock = asyncio.Lock()
        self._join_event = asyncio.Event()

    @property
    def game_id(self) -> str:
        return self.state.game_id if self.state else "unknown"

    async def start(
        self,
        ctx: commands.Context,
        timeout: Optional[float] = None
    ) -> Optional[Path]:
        """
        Start the Interdimensional Cable game.

        Args:
            ctx: Discord context
            timeout: Overall game timeout (default: calculated from clips)

        Returns:
            Path to final concatenated video, or None if failed
        """
        # Verify FFmpeg is available
        if not is_ffmpeg_available():
            await self._send_gamemaster_message(
                "**ERROR:** FFmpeg not installed. Cannot run Interdimensional Cable game.\n"
                "Run `build.bat` to install FFmpeg automatically."
            )
            return None

        # Initialize game state
        self.state = IDCCGameState(
            game_id=f"idcc_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            channel_id=ctx.channel.id,
            start_time=time.time(),
            registration_end_time=time.time() + idcc_config.registration_duration_seconds,
            num_clips=self.num_clips,
            phase="init"
        )

        logger.info(f"[IDCC:{self.game_id}] Starting Interdimensional Cable game with {self.num_clips} clips")

        try:
            # Phase 1: Registration
            await self._run_registration_phase(ctx)

            # Phase 2: Finalize participants
            await self._finalize_participants()

            if not self.state.participants:
                await self._send_gamemaster_message(
                    "**CANCELLED:** No participants available for Interdimensional Cable."
                )
                self.state.phase = "failed"
                return None

            # Phase 3: Generate clips
            await self._run_generation_phase(ctx)

            # Check if we have enough clips
            successful_clips = [c for c in self.state.clips if c.success]
            if len(successful_clips) < 2:
                await self._send_gamemaster_message(
                    f"**FAILED:** Only {len(successful_clips)} clip(s) generated successfully. "
                    "Need at least 2 to make a video."
                )
                self.state.phase = "failed"
                return None

            # Phase 4: Concatenate
            await self._run_concatenation_phase()

            # Phase 5: Post result
            await self._post_final_video(ctx)

            self.state.phase = "complete"
            logger.info(f"[IDCC:{self.game_id}] Game completed successfully!")

            return self.state.final_video_path

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Game error: {e}", exc_info=True)
            self.state.phase = "failed"
            self.state.error_message = str(e)
            await self._send_gamemaster_message(
                f"**ERROR:** Interdimensional Cable game crashed: {str(e)[:200]}"
            )
            return None

        finally:
            # Cleanup old temp files
            cleanup_temp_files(max_age_hours=24)

    async def handle_join_command(self, user_name: str) -> bool:
        """
        Handle a !join-idcc command during registration.

        Args:
            user_name: Discord username of the joiner

        Returns:
            True if successfully registered, False otherwise
        """
        if not self.state:
            return False

        if self.state.phase != "registration":
            logger.debug(f"[IDCC:{self.game_id}] Join rejected - not in registration phase")
            return False

        if time.time() > self.state.registration_end_time:
            logger.debug(f"[IDCC:{self.game_id}] Join rejected - registration closed")
            return False

        async with self._registration_lock:
            if user_name in self.state.registered_humans:
                logger.debug(f"[IDCC:{self.game_id}] {user_name} already registered")
                return False

            self.state.registered_humans.add(user_name)
            logger.info(f"[IDCC:{self.game_id}] {user_name} joined! ({len(self.state.registered_humans)} humans)")

            # Signal that someone joined
            self._join_event.set()
            self._join_event.clear()

            return True

    async def handle_scene_submission(self, user_name: str, message_content: str) -> bool:
        """
        Handle a [SCENE] submission from a human participant.

        Users submit scenes with: [SCENE] followed by their scene description

        Args:
            user_name: Discord username
            message_content: Full message content

        Returns:
            True if scene was accepted, False otherwise
        """
        if not self.state:
            return False

        # Only accept during awaiting_scene phase
        if self.state.phase != "awaiting_scene":
            return False

        # Check if this user is the one we're waiting for
        if not self.state.waiting_for_human_scene:
            return False

        current_participant = self.state.participants[self.state.current_clip_index]
        if current_participant["name"] != user_name:
            logger.debug(f"[IDCC:{self.game_id}] Scene from {user_name} but waiting for {current_participant['name']}")
            return False

        # Extract scene from [SCENE] tag
        scene_prompt = None
        if "[SCENE]" in message_content.upper():
            # Extract everything after [SCENE]
            upper_content = message_content.upper()
            scene_start = upper_content.find("[SCENE]") + len("[SCENE]")
            scene_prompt = message_content[scene_start:].strip()
        elif message_content.strip():
            # Accept message without tag if they're the current participant
            scene_prompt = message_content.strip()

        if scene_prompt and len(scene_prompt) > 10:  # Minimum 10 chars
            async with self._registration_lock:
                self.state.pending_human_scenes[user_name] = scene_prompt
                self.state.waiting_for_human_scene = False
                self._join_event.set()  # Signal that we received the scene
                logger.info(f"[IDCC:{self.game_id}] {user_name} submitted scene: {scene_prompt[:50]}...")
                return True

        return False

    # ========================================================================
    # PHASE 1: REGISTRATION
    # ========================================================================

    async def _run_registration_phase(self, ctx: commands.Context):
        """Run the 2-minute registration phase."""
        self.state.phase = "registration"

        # Announce game start
        participants_needed = self.num_clips
        announcement = (
            "# INTERDIMENSIONAL CABLE\n"
            "### Collaborative Absurdist Video Creation\n\n"
            f"We need **{participants_needed}** participants to create a {self.num_clips}-scene cable TV clip.\n\n"
            "**Type `!join-idcc` to participate!**\n\n"
            f"Registration closes in **2 minutes**.\n"
            "If not enough humans join, bots will fill remaining slots.\n\n"
            "*Get ready to create something beautifully weird...*"
        )

        await self._send_gamemaster_message(announcement)

        # Wait for registration period with engaging updates
        registration_duration = idcc_config.registration_duration_seconds
        check_interval = 15  # Check every 15 seconds

        # Conversation prompts to encourage chat during registration
        chat_prompts = [
            "While we wait... what's the weirdest commercial you've ever seen?",
            "Predictions? What dimension will our cable signal come from today?",
            "Remember: the weirder the better. Public access from parallel realities.",
            "What kind of impossible product should we advertise today?",
        ]
        prompt_index = 0

        elapsed = 0
        while elapsed < registration_duration:
            await asyncio.sleep(check_interval)
            elapsed += check_interval

            remaining = registration_duration - elapsed
            human_count = len(self.state.registered_humans)

            if remaining > 0:
                if remaining == 90:  # 90s mark
                    await self._send_gamemaster_message(
                        f"**90 seconds left!** {human_count}/{participants_needed} joined so far.\n"
                        f"*{chat_prompts[prompt_index % len(chat_prompts)]}*"
                    )
                    prompt_index += 1
                elif remaining == 60:  # 60s mark
                    await self._send_gamemaster_message(
                        f"**ONE MINUTE remaining!** {human_count}/{participants_needed} participants.\n"
                        f"*{chat_prompts[prompt_index % len(chat_prompts)]}*"
                    )
                    prompt_index += 1
                elif remaining == 30:  # 30s mark
                    await self._send_gamemaster_message(
                        f"**30 SECONDS!** Last call for `!join-idcc`!\n"
                        f"Current crew: {human_count}/{participants_needed}"
                    )

        # Registration closed
        human_count = len(self.state.registered_humans)
        if human_count > 0:
            humans_list = ", ".join(self.state.registered_humans)
            await self._send_gamemaster_message(
                f"# 📺 REGISTRATION CLOSED!\n\n"
                f"**Human directors:** {humans_list}\n"
                f"Filling remaining {max(0, participants_needed - human_count)} slot(s) with our AI auteurs..."
            )
        else:
            await self._send_gamemaster_message(
                "# 📺 REGISTRATION CLOSED!\n\n"
                "No humans joined - our AI collective will create this interdimensional broadcast!\n"
                "*Prepare for something beautifully unhinged...*"
            )

    # ========================================================================
    # PHASE 2: FINALIZE PARTICIPANTS
    # ========================================================================

    async def _finalize_participants(self):
        """Finalize the participant list after registration."""
        self.state.phase = "finalizing"

        participants = []
        needed = self.num_clips

        # Add registered humans first
        for username in self.state.registered_humans:
            if len(participants) < needed:
                participants.append({
                    "name": username,
                    "type": "human",
                    "agent_obj": None
                })

        # Fill remaining slots with available agents
        from constants import is_image_model
        available_agents = [
            agent for agent in self.agent_manager.get_all_agents()
            if agent.is_running and not is_image_model(agent.model)
        ]

        # Shuffle for variety
        random.shuffle(available_agents)

        for agent in available_agents:
            if len(participants) < needed:
                participants.append({
                    "name": agent.name,
                    "type": "agent",
                    "agent_obj": agent
                })

        self.state.participants = participants

        # Log final lineup
        lineup = ", ".join([f"{p['name']} ({'human' if p['type'] == 'human' else 'bot'})" for p in participants])
        logger.info(f"[IDCC:{self.game_id}] Final lineup: {lineup}")

        await self._send_gamemaster_message(
            f"**CAST ASSEMBLED:** {lineup}\n\n"
            "Starting scene generation... This may take several minutes."
        )

    # ========================================================================
    # PHASE 3: CLIP GENERATION
    # ========================================================================

    async def _run_generation_phase(self, ctx: commands.Context):
        """Generate all video clips sequentially with progress feedback."""
        self.state.phase = "generating"

        ensure_temp_dir()
        previous_prompt = None
        previous_frame_path = None

        # Cycle through participants if num_clips > len(participants)
        num_participants = len(self.state.participants)
        for i in range(self.num_clips):
            clip_num = i + 1
            # Use modulo to cycle through participants
            participant_index = i % num_participants
            participant = self.state.participants[participant_index]
            creator_name = participant["name"]
            creator_type = participant["type"]

            logger.info(f"[IDCC:{self.game_id}] Generating clip {clip_num}/{self.num_clips} by {creator_name}")

            # Create clip record
            clip = IDCCClip(
                clip_number=clip_num,
                creator_name=creator_name,
                creator_type=creator_type,
                prompt=""
            )
            self.state.clips.append(clip)
            self.state.current_clip_index = i

            # Show last frame to channel if we have one (for humans to see context)
            if previous_frame_path and previous_frame_path.exists() and creator_type == "human":
                try:
                    # Post the last frame so humans can see what they're continuing from
                    with open(previous_frame_path, "rb") as f:
                        file = discord.File(f, filename="last_frame.png")
                        if self.discord_client.webhook:
                            await self.discord_client.webhook.send(
                                content=f"**LAST FRAME** - {creator_name}, continue from this moment:",
                                file=file,
                                username="GameMaster"
                            )
                except Exception as e:
                    logger.warning(f"[IDCC:{self.game_id}] Could not post last frame: {e}")

            # Announce whose turn it is
            if clip_num == 1:
                if creator_type == "human":
                    await self._send_gamemaster_message(
                        f"# Scene {clip_num}/{self.num_clips}: {creator_name}'s turn!\n\n"
                        f"**Create the OPENING scene.** Describe an absurdist TV clip.\n"
                        f"Type `[SCENE]` followed by your scene description.\n"
                        f"Example: `[SCENE] A man in a cheap suit stands in a parking lot, deadpan presenting a product called \"Invisible Soup\"`\n\n"
                        f"*You have 2 minutes to submit...*"
                    )
                else:
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num}/{self.num_clips}:** {creator_name} is creating the opening..."
                    )
            else:
                if creator_type == "human":
                    await self._send_gamemaster_message(
                        f"# Scene {clip_num}/{self.num_clips}: {creator_name}'s turn!\n\n"
                        f"**YES-AND the previous scene.** Continue from the last frame shown above.\n"
                        f"Previous: *\"{previous_prompt[:100] if previous_prompt else 'Unknown'}...\"*\n\n"
                        f"Type `[SCENE]` followed by what happens NEXT.\n"
                        f"Remember: Same characters, same setting - but the story advances!\n\n"
                        f"*You have 2 minutes to submit...*"
                    )
                else:
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num}/{self.num_clips}:** {creator_name} continues the story..."
                    )

            # Get the scene prompt (from human or agent)
            prompt = None
            try:
                if creator_type == "human":
                    # Wait for human [SCENE] submission
                    prompt = await self._wait_for_human_scene(
                        participant=participant,
                        clip_number=clip_num,
                        previous_prompt=previous_prompt,
                        previous_frame_path=previous_frame_path
                    )
                else:
                    # Generate prompt via agent
                    prompt = await self._generate_scene_prompt(
                        participant=participant,
                        clip_number=clip_num,
                        previous_prompt=previous_prompt,
                        previous_frame_path=previous_frame_path
                    )

                clip.prompt = prompt or ""

                if not prompt:
                    clip.error_message = "Failed to generate prompt"
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num} skipped:** No valid prompt received."
                    )
                    continue

                # Show the prompt that will be used
                await self._send_gamemaster_message(
                    f"**Scene {clip_num} prompt:** *\"{prompt[:200]}{'...' if len(prompt) > 200 else ''}\"*\n"
                    f"Generating video... (this takes 1-3 minutes)"
                )

            except Exception as e:
                logger.error(f"[IDCC:{self.game_id}] Prompt generation error: {e}", exc_info=True)
                clip.error_message = f"Prompt error: {str(e)}"
                continue

            # Generate the video with progress feedback
            try:
                start_time = time.time()
                self.state.phase = "generating_video"

                # Start progress indicator task
                progress_task = asyncio.create_task(
                    self._show_generation_progress(clip_num, creator_name)
                )

                video_result = await self._generate_video_clip(
                    prompt=prompt,
                    creator_name=creator_name,
                    reference_frame_path=previous_frame_path if clip_num > 1 else None
                )

                # Stop progress indicator
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

                clip.generation_time_seconds = time.time() - start_time

                if video_result:
                    clip.video_path = video_result["path"]
                    clip.video_url = video_result.get("url")
                    clip.success = True

                    # Extract last frame for next iteration
                    previous_frame_path = extract_last_frame(clip.video_path)
                    clip.last_frame_path = previous_frame_path
                    previous_prompt = prompt

                    # Post success
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num} complete!** ({clip.generation_time_seconds:.0f}s) by {creator_name}"
                    )

                else:
                    clip.error_message = "Video generation returned None"
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num} failed.** Video generation error."
                    )

            except Exception as e:
                logger.error(f"[IDCC:{self.game_id}] Video generation error: {e}", exc_info=True)
                clip.error_message = f"Generation error: {str(e)}"

        # Summary
        successful = sum(1 for c in self.state.clips if c.success)
        total_time = sum(c.generation_time_seconds for c in self.state.clips)
        self.state.total_generation_time = total_time

        await self._send_gamemaster_message(
            f"**Generation complete!** {successful}/{self.num_clips} scenes created in {total_time:.0f}s total.\n"
            "Assembling final video..."
        )

    async def _wait_for_human_scene(
        self,
        participant: Dict[str, Any],
        clip_number: int,
        previous_prompt: Optional[str],
        previous_frame_path: Optional[Path]
    ) -> Optional[str]:
        """
        Wait for a human to submit their [SCENE] prompt.

        Args:
            participant: Participant info dict
            clip_number: Current clip number
            previous_prompt: Previous scene's prompt
            previous_frame_path: Path to last frame

        Returns:
            Scene prompt string, or None if timeout/failure
        """
        user_name = participant["name"]
        timeout = self.state.human_scene_timeout

        # Set state to awaiting
        self.state.phase = "awaiting_scene"
        self.state.waiting_for_human_scene = True

        try:
            # Wait for scene submission
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check if scene was submitted
                if user_name in self.state.pending_human_scenes:
                    prompt = self.state.pending_human_scenes.pop(user_name)
                    await self._send_gamemaster_message(
                        f"**{user_name}'s scene received!**"
                    )
                    return prompt

                # Wait a bit before checking again
                await asyncio.sleep(2)

            # Timeout - use agent as fallback
            await self._send_gamemaster_message(
                f"**{user_name} timed out.** A bot will generate this scene instead..."
            )

            # Get a random agent to generate instead
            from constants import is_image_model
            available_agents = [
                a for a in self.agent_manager.get_all_agents()
                if a.is_running and not is_image_model(a.model)
            ]
            if available_agents:
                proxy_agent = random.choice(available_agents)
                return await self._agent_generate_prompt(
                    agent=proxy_agent,
                    clip_number=clip_number,
                    previous_prompt=previous_prompt,
                    previous_frame_path=previous_frame_path
                )

            return None

        finally:
            self.state.waiting_for_human_scene = False
            self.state.phase = "generating"

    async def _show_generation_progress(self, clip_num: int, creator_name: str):
        """Show periodic progress updates during video generation with chat hooks."""
        elapsed = 0
        # Conversation hooks to keep chat alive during long generation
        chat_hooks = [
            f"*The quantum uplink is rendering {creator_name}'s vision...*",
            "*Meanwhile, somewhere across the multiverse, a signal is forming...*",
            f"*What do you think {creator_name}'s scene will look like?*",
            "*The interdimensional transmission continues to materialize...*",
            f"*{creator_name}'s contribution is being encoded into our reality...*",
            "*Any predictions for the next scene? The weirder, the better.*",
        ]
        hook_index = 0
        try:
            while True:
                await asyncio.sleep(45)  # Update every 45 seconds
                elapsed += 45
                # Alternate between progress and chat hooks
                if elapsed % 90 == 45:
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num}** rendering... ({elapsed}s)\n"
                        f"{chat_hooks[hook_index % len(chat_hooks)]}"
                    )
                    hook_index += 1
                else:
                    await self._send_gamemaster_message(
                        f"Scene {clip_num} still processing... ({elapsed}s)"
                    )
        except asyncio.CancelledError:
            pass  # Normal cancellation when generation completes

    async def _generate_scene_prompt(
        self,
        participant: Dict[str, Any],
        clip_number: int,
        previous_prompt: Optional[str],
        previous_frame_path: Optional[Path]
    ) -> Optional[str]:
        """
        Generate a scene prompt for a participant.

        Args:
            participant: Participant info dict
            clip_number: 1-indexed clip number
            previous_prompt: Previous scene's prompt (for continuity)
            previous_frame_path: Path to last frame of previous clip

        Returns:
            Generated prompt string, or None on failure
        """
        creator_type = participant["type"]
        creator_name = participant["name"]

        if creator_type == "agent":
            # Use the agent to generate the prompt
            agent = participant["agent_obj"]
            return await self._agent_generate_prompt(
                agent=agent,
                clip_number=clip_number,
                previous_prompt=previous_prompt,
                previous_frame_path=previous_frame_path
            )
        else:
            # Human participant - they would need to provide input
            # For now, fall back to using a random available agent
            # Future: implement Discord message collection from human
            logger.warning(f"[IDCC:{self.game_id}] Human {creator_name} - using bot as proxy for now")

            # Get a random agent to generate on behalf of human
            from constants import is_image_model
            available_agents = [
                a for a in self.agent_manager.get_all_agents()
                if a.is_running and not is_image_model(a.model)
            ]
            if available_agents:
                proxy_agent = random.choice(available_agents)
                return await self._agent_generate_prompt(
                    agent=proxy_agent,
                    clip_number=clip_number,
                    previous_prompt=previous_prompt,
                    previous_frame_path=previous_frame_path
                )

            return None

    async def _agent_generate_prompt(
        self,
        agent: 'Agent',
        clip_number: int,
        previous_prompt: Optional[str],
        previous_frame_path: Optional[Path]
    ) -> Optional[str]:
        """
        Have an agent generate a scene prompt.

        Args:
            agent: Agent to generate the prompt
            clip_number: 1-indexed clip number
            previous_prompt: Previous scene's prompt
            previous_frame_path: Path to last frame image

        Returns:
            Generated prompt string
        """
        try:
            import aiohttp

            # Build the system context with duration/resolution info
            duration = idcc_config.clip_duration_seconds
            resolution = idcc_config.video_resolution

            if clip_number == 1:
                cold_open = IDCC_COLD_OPEN_PROMPT.format(
                    duration=duration,
                    resolution=resolution
                )
                system_content = f"{agent.system_prompt}\n\n{cold_open}"
                user_content = "Create the opening scene for an interdimensional cable TV clip. Output ONLY the video prompt."
                images = []
            else:
                continuation_prompt = IDCC_CONTINUATION_PROMPT.format(
                    previous_prompt=previous_prompt or "Unknown previous scene",
                    duration=duration,
                    resolution=resolution
                )
                system_content = f"{agent.system_prompt}\n\n{continuation_prompt}"
                user_content = "Continue the scene from the frame shown. Output ONLY the video prompt."

                # Include previous frame as image
                images = []
                if previous_frame_path and previous_frame_path.exists():
                    frame_b64 = image_to_base64(previous_frame_path)
                    if frame_b64:
                        images.append({
                            "type": "image_url",
                            "image_url": {"url": frame_b64}
                        })

            # Build messages
            messages = [{"role": "system", "content": system_content}]

            if images:
                messages.append({
                    "role": "user",
                    "content": [
                        *images,
                        {"type": "text", "text": user_content}
                    ]
                })
            else:
                messages.append({"role": "user", "content": user_content})

            # Call the LLM
            headers = {
                "Authorization": f"Bearer {self.agent_manager.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": agent.model,
                "messages": messages,
                "max_tokens": 300
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"[IDCC:{self.game_id}] Prompt generation API error: {error_text[:200]}")
                        return None

                    result = await response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

                    if not content:
                        return None

                    # Clean up the response (remove any markdown, explanations, etc.)
                    prompt = content.strip()

                    # Remove common prefixes/suffixes
                    for prefix in ["Here is the video prompt:", "Video prompt:", "Prompt:"]:
                        if prompt.lower().startswith(prefix.lower()):
                            prompt = prompt[len(prefix):].strip()

                    logger.info(f"[IDCC:{self.game_id}] {agent.name} generated: {prompt[:100]}...")
                    return prompt

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Agent prompt generation error: {e}", exc_info=True)
            return None

    async def _generate_video_clip(
        self,
        prompt: str,
        creator_name: str,
        reference_frame_path: Optional[Path] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a video clip using Sora 2.

        Args:
            prompt: The video generation prompt
            creator_name: Who is creating this clip
            reference_frame_path: Optional starting frame for continuity

        Returns:
            Dict with 'path' and optionally 'url', or None on failure
        """
        max_retries = idcc_config.max_retries_per_clip

        for attempt in range(max_retries):
            try:
                # First try: use original prompt
                # On retry: declassify the prompt
                use_prompt = prompt
                reference_image_b64 = None

                if attempt > 0:
                    # Get declassified version
                    use_prompt = await self.agent_manager.declassify_image_prompt(
                        prompt, variant=attempt
                    ) or prompt
                    logger.info(f"[IDCC:{self.game_id}] Retry {attempt} with declassified prompt")

                # Prepare reference frame if provided
                if reference_frame_path and reference_frame_path.exists():
                    # Resize to match Sora's expected dimensions
                    resized_frame = await resize_image_for_sora(reference_frame_path)
                    if resized_frame:
                        reference_image_b64 = image_to_base64(resized_frame)

                # Generate video with image-to-video if we have a reference
                video_result = await self.agent_manager.image_agent.generate_video_with_reference(
                    prompt=use_prompt,
                    author=creator_name,
                    duration=idcc_config.clip_duration_seconds,
                    input_reference=reference_image_b64
                )

                if video_result:
                    # Download video if it's a URL
                    if isinstance(video_result, str):
                        if video_result.startswith("FILE:"):
                            # Already a local file
                            video_path = Path(video_result.replace("FILE:", ""))
                        else:
                            # It's a URL - download it
                            ensure_temp_dir()
                            video_path = VIDEO_TEMP_DIR / f"clip_{self.game_id}_{int(time.time())}.mp4"
                            success = await download_video(video_result, video_path)
                            if not success:
                                continue

                        return {"path": video_path, "url": video_result}

            except Exception as e:
                logger.error(f"[IDCC:{self.game_id}] Video generation attempt {attempt + 1} failed: {e}")

            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(5)

        return None

    # ========================================================================
    # PHASE 4: CONCATENATION
    # ========================================================================

    async def _run_concatenation_phase(self):
        """Concatenate all successful clips into final video."""
        self.state.phase = "concatenating"

        # Get successful clips in order
        successful_clips = [c for c in self.state.clips if c.success and c.video_path]

        if len(successful_clips) < 2:
            logger.error(f"[IDCC:{self.game_id}] Not enough clips to concatenate")
            return

        video_paths = [c.video_path for c in successful_clips]

        # Output path
        ensure_temp_dir()
        output_path = VIDEO_TEMP_DIR / f"idcc_final_{self.game_id}.mp4"

        # Concatenate
        crossfade = idcc_config.crossfade_duration if idcc_config.use_crossfade else 0

        result = await concatenate_videos(
            video_paths=video_paths,
            output_path=output_path,
            crossfade_duration=crossfade
        )

        if result and result.exists():
            self.state.final_video_path = result
            logger.info(f"[IDCC:{self.game_id}] Final video created: {result}")
        else:
            logger.error(f"[IDCC:{self.game_id}] Concatenation failed")

    # ========================================================================
    # PHASE 5: POST RESULT
    # ========================================================================

    async def _post_final_video(self, ctx: commands.Context):
        """Post the final video to Discord."""
        if not self.state.final_video_path or not self.state.final_video_path.exists():
            await self._send_gamemaster_message(
                "**ERROR:** Final video file not found. Something went wrong during assembly."
            )
            return

        # Build credits
        credits_lines = ["**CREDITS:**"]
        for clip in self.state.clips:
            status = "completed" if clip.success else "failed"
            credits_lines.append(f"• Scene {clip.clip_number}: {clip.creator_name} ({status})")

        total_time = self.state.total_generation_time
        credits_lines.append(f"\n*Total generation time: {total_time:.0f}s*")

        credits = "\n".join(credits_lines)

        # Send the video
        try:
            channel = ctx.channel

            # Check file size (Discord limit is 25MB for regular, 100MB for boosted)
            file_size_mb = self.state.final_video_path.stat().st_size / (1024 * 1024)

            if file_size_mb > 25:
                # Too large - post link instead
                await self._send_gamemaster_message(
                    f"# INTERDIMENSIONAL CABLE - COMPLETE\n\n"
                    f"Video is {file_size_mb:.1f}MB (too large for Discord upload).\n"
                    f"File saved to: `{self.state.final_video_path}`\n\n"
                    f"{credits}"
                )
            else:
                # Upload to Discord
                await self._send_gamemaster_message(
                    "# INTERDIMENSIONAL CABLE - COMPLETE\n\n"
                    "*From dimensions unknown, a signal emerges...*"
                )

                # Send via webhook or channel
                if self.discord_client.webhook:
                    with open(self.state.final_video_path, "rb") as f:
                        file = discord.File(f, filename="interdimensional_cable.mp4")
                        await self.discord_client.webhook.send(
                            content=credits,
                            file=file,
                            username="GameMaster",
                            avatar_url=None
                        )
                else:
                    with open(self.state.final_video_path, "rb") as f:
                        file = discord.File(f, filename="interdimensional_cable.mp4")
                        await channel.send(content=credits, file=file)

                logger.info(f"[IDCC:{self.game_id}] Final video posted to Discord")

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Failed to post video: {e}", exc_info=True)
            await self._send_gamemaster_message(
                f"**ERROR:** Could not upload video to Discord: {str(e)[:200]}\n"
                f"Video saved to: `{self.state.final_video_path}`\n\n"
                f"{credits}"
            )

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    async def _send_gamemaster_message(self, content: str) -> Optional[discord.Message]:
        """Send a message as GameMaster."""
        try:
            if self.discord_client:
                return await self.discord_client.send_message(
                    content,
                    agent_name="GameMaster",
                    model_name="system"
                )
        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Failed to send message: {e}")
        return None


# ============================================================================
# GAME MANAGER INTEGRATION
# ============================================================================

class IDCCGameManager:
    """
    Manager for Interdimensional Cable game instances.

    Handles command registration, active game tracking, and lifecycle.
    """

    def __init__(self):
        self.active_game: Optional[InterdimensionalCableGame] = None
        self._lock = asyncio.Lock()

    def is_game_active(self) -> bool:
        """Check if a game is currently running."""
        return self.active_game is not None

    async def start_game(
        self,
        agent_manager,
        discord_client,
        ctx: commands.Context,
        num_clips: int = 5
    ) -> bool:
        """
        Start a new Interdimensional Cable game.

        Args:
            agent_manager: AgentManager instance
            discord_client: DiscordBotClient instance
            ctx: Discord context
            num_clips: Number of clips to generate

        Returns:
            True if game started successfully
        """
        async with self._lock:
            if self.active_game:
                await discord_client.send_message(
                    "**An Interdimensional Cable game is already in progress!**",
                    agent_name="GameMaster",
                    model_name="system"
                )
                return False

            self.active_game = InterdimensionalCableGame(
                agent_manager=agent_manager,
                discord_client=discord_client,
                num_clips=num_clips
            )

        try:
            await self.active_game.start(ctx)
            return True
        finally:
            async with self._lock:
                self.active_game = None

    async def handle_join(self, user_name: str) -> bool:
        """
        Handle a !join-idcc command.

        Args:
            user_name: Discord username

        Returns:
            True if successfully registered
        """
        if self.active_game:
            return await self.active_game.handle_join_command(user_name)
        return False


# Global manager instance
idcc_manager = IDCCGameManager()
