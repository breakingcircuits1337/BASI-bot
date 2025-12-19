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
from config_manager import config_manager

import discord
from discord.ext import commands

from .ffmpeg_utils import (
    extract_last_frame,
    concatenate_videos,
    download_video,
    ensure_temp_dir,
    ensure_media_dirs,
    cleanup_temp_files,
    image_to_base64,
    resize_image_for_sora,
    is_ffmpeg_available,
    copy_video_to_media,
    save_media_prompt,
    save_failed_prompt,
    VIDEO_TEMP_DIR,
    MEDIA_VIDEOS_DIR
)
from .game_context import GameContext, game_context_manager
from .game_prompts import get_game_prompt, get_shot_direction
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
    clip_duration_seconds: int = 8  # Each Sora clip duration (4, 8, or 12)
    video_resolution: str = "1280x720"  # Landscape 720p
    max_retries_per_clip: int = 3  # Retries if generation fails
    generation_timeout_seconds: int = 300  # 5 min per clip max
    use_crossfade: bool = False  # Crossfade between clips (slower)
    crossfade_duration: float = 0.5  # Seconds of crossfade
    spitball_round_timeout_seconds: int = 45  # Time for humans to submit in each spitball round
    game_cooldown_seconds: int = 1800  # 30 min cooldown between games (matches other games)


# Global cooldown tracking
_last_idcc_end_time: float = 0


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
            if clip_duration_seconds in [4, 8, 12]:
                idcc_config.clip_duration_seconds = clip_duration_seconds
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
# INTERDIMENSIONAL CABLE REFERENCE - Inspiration for original pitches
# ============================================================================
# Agents pitch ORIGINAL ideas inspired by these classic Interdimensional Cable bits:
#
# FORMATS TO USE:
# - Infomercial (Real Fake Doors, Ants in My Eyes Johnson Electronics)
# - Restaurant ad (Lil' Bits - tiny food, creepy whisper)
# - Cop show (Baby Legs - detective with baby legs partnered with Regular Legs)
# - Talk show (How Did I Get Here? - no one knows how they got there)
# - Movie trailer (Two Brothers - keeps escalating into nonsense)
# - Personal space show (host keeps violating his own rule about personal space)
# - Sitcom parody (Gazorpazorpfield - Garfield but violent and abusive)
# - Lawyer ad (personal injury lawyer with increasingly specific scenarios)
# - Cooking show, workout video, news segment, PSA, dating show, etc.
#
# WHAT MAKES THEM WORK:
# - Committed performance (play it 100% straight)
# - One absurd premise explored fully
# - Character has a specific visual look and voice
# - Clear punchline or button
# - Self-contained (doesn't need context)
#
# Agents should pitch ORIGINAL bits in these formats, not copies of existing ones.
# ============================================================================


@dataclass
class HumanVoteSelection:
    """Tracks a human's votes through the writers room rounds."""
    user_name: str
    voted_bits: List[int] = field(default_factory=list)  # Indices of bits they voted for


class WritersRoomSystem:
    """
    Manages the writers room for Robot Chicken style IDCC.

    Flow:
    1. All participants (agents + humans) pitch complete bits
    2. Everyone votes to select the best N bits for the lineup
    3. Lineup is curated for variety (no back-to-back same format)
    """

    def __init__(self, game_id: str, num_clips: int, clip_duration: int):
        self.game_id = game_id
        self.num_clips = num_clips
        self.clip_duration = clip_duration
        self.human_selections: Dict[str, HumanVoteSelection] = {}
        self.pitched_bits: List[BitConcept] = []  # All pitched bits
        self.bit_pitchers: List[str] = []  # Who pitched each bit (parallel to pitched_bits)

    def register_human(self, user_name: str):
        """Register a human for voting."""
        self.human_selections[user_name] = HumanVoteSelection(user_name=user_name)

    def add_pitched_bit(self, bit: BitConcept, pitcher_name: str):
        """Add a pitched bit to the pool."""
        bit.pitched_by = pitcher_name
        bit.duration_beats = get_duration_beats(self.clip_duration)
        self.pitched_bits.append(bit)
        self.bit_pitchers.append(pitcher_name)

    def format_bits_for_voting(self) -> str:
        """Format all pitched bits for Discord voting display."""
        if not self.pitched_bits:
            return "No bits pitched yet."

        lines = [f"📺 **VOTE FOR THE LINEUP** (type numbers 1-{len(self.pitched_bits)}, pick up to {self.num_clips}):\n"]
        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

        for i, bit in enumerate(self.pitched_bits):
            emoji = emojis[i] if i < len(emojis) else f"({i+1})"
            # Show parody info if available, fallback to format
            parody_info = bit.parody_target or bit.format
            twist_info = bit.twist or bit.comedic_hook or bit.premise
            lines.append(f"{emoji} **{parody_info.upper()}** by {bit.pitched_by}")
            lines.append(f"   TWIST: {twist_info[:100]}{'...' if len(twist_info) > 100 else ''}")
            if bit.punchline:
                lines.append(f"   PUNCHLINE: {bit.punchline[:80]}{'...' if len(bit.punchline) > 80 else ''}\n")
            else:
                lines.append("")

        return "\n".join(lines)

    def parse_vote(self, user_name: str, message: str) -> List[int]:
        """Parse a user's vote(s). Can vote for multiple bits."""
        if user_name not in self.human_selections:
            return []

        votes = []
        for char in message:
            if char.isdigit():
                num = int(char)
                if 1 <= num <= len(self.pitched_bits):
                    votes.append(num - 1)  # 0-indexed

        # Store their votes
        self.human_selections[user_name].voted_bits = votes
        return votes

    def tally_votes(self, agent_votes: Dict[str, List[int]]) -> List[int]:
        """
        Tally all votes (human + agent) and return indices of winning bits.

        Args:
            agent_votes: Dict of agent_name -> list of bit indices they voted for

        Returns:
            List of bit indices, sorted by vote count, limited to num_clips
        """
        vote_counts = [0] * len(self.pitched_bits)

        # Count human votes
        for selection in self.human_selections.values():
            for idx in selection.voted_bits:
                if 0 <= idx < len(self.pitched_bits):
                    vote_counts[idx] += 1

        # Count agent votes
        for votes in agent_votes.values():
            for idx in votes:
                if 0 <= idx < len(self.pitched_bits):
                    vote_counts[idx] += 1

        # Sort by vote count (descending), return top N indices
        sorted_indices = sorted(range(len(vote_counts)), key=lambda i: vote_counts[i], reverse=True)
        return sorted_indices[:self.num_clips]

    def curate_lineup(self, winning_indices: List[int]) -> List[BitConcept]:
        """
        Curate the final lineup from winning bits.
        Reorder to avoid back-to-back same formats where possible.

        Args:
            winning_indices: Indices of bits that won the vote

        Returns:
            Ordered list of BitConcepts for the final lineup
        """
        if not winning_indices:
            return []

        winning_bits = [self.pitched_bits[i] for i in winning_indices]

        # Simple curation: try to avoid same format back-to-back
        curated = [winning_bits[0]]
        remaining = winning_bits[1:]

        while remaining:
            last_format = curated[-1].format.lower()

            # Find a bit with different format if possible
            different_format = None
            for i, bit in enumerate(remaining):
                if bit.format.lower() != last_format:
                    different_format = i
                    break

            if different_format is not None:
                curated.append(remaining.pop(different_format))
            else:
                # No choice, just take next one
                curated.append(remaining.pop(0))

        return curated

    def format_bit_for_punch_up(self, bit: BitConcept) -> str:
        """Format a single bit for the punch-up round display."""
        lines = [
            f"**PARODY:** {bit.parody_target or bit.format}",
            f"**TWIST:** {bit.twist or bit.comedic_hook}",
            f"**FORMAT:** {bit.format}",
            f"**CHARACTER:** {bit.character_description[:150]}{'...' if len(bit.character_description) > 150 else ''}",
            f"**VOICE:** {bit.vocal_specs}",
            f"**DIALOGUE:** {bit.sample_dialogue[:150]}{'...' if len(bit.sample_dialogue) > 150 else ''}",
            f"**PUNCHLINE:** {bit.punchline}",
            f"*Pitched by: {bit.pitched_by}*"
        ]
        return "\n".join(lines)

    def parse_punch_up_response(self, response: str, pitcher_name: str, responder_name: str) -> Dict:
        """
        Parse a punch-up response.

        Returns:
            Dict with keys: verdict ("GOOD_AS_IS" or "PUNCH_UP"), suggestion, reason
        """
        response_upper = response.upper()
        result = {
            "verdict": "GOOD_AS_IS",
            "suggestion": "",
            "reason": "",
            "responder": responder_name
        }

        # Can't punch up your own bit
        if pitcher_name == responder_name:
            result["reason"] = "Own bit - automatically GOOD AS IS"
            return result

        # Check for GOOD AS IS
        if "GOOD AS IS" in response_upper or "GOODASIS" in response_upper:
            result["verdict"] = "GOOD_AS_IS"
            # Try to extract reason
            if "REASON:" in response_upper:
                reason_idx = response_upper.find("REASON:")
                result["reason"] = response[reason_idx + 7:].split("\n")[0].strip()
            return result

        # Check for PUNCH-UP
        if "PUNCH-UP" in response_upper or "PUNCHUP" in response_upper or "PUNCH UP" in response_upper:
            result["verdict"] = "PUNCH_UP"
            # Extract suggestion
            if "SUGGESTION:" in response_upper:
                sugg_idx = response_upper.find("SUGGESTION:")
                sugg_text = response[sugg_idx + 11:]
                # Get until REASON or newline
                if "REASON:" in sugg_text.upper():
                    reason_idx = sugg_text.upper().find("REASON:")
                    result["suggestion"] = sugg_text[:reason_idx].strip()
                    result["reason"] = sugg_text[reason_idx + 7:].split("\n")[0].strip()
                else:
                    result["suggestion"] = sugg_text.split("\n")[0].strip()
            return result

        # Default to good as is if unclear
        return result

    def format_punch_ups_for_voting(self, punch_ups: List[Dict]) -> str:
        """Format punch-up suggestions for voting."""
        if not punch_ups:
            return "No punch-ups suggested."

        lines = []
        for i, pu in enumerate(punch_ups, 1):
            lines.append(f"**{i}.** {pu['suggestion'][:200]}")
            if pu.get('reason'):
                lines.append(f"   *({pu['reason'][:100]})*")
            lines.append(f"   — {pu['responder']}\n")

        return "\n".join(lines)

    def parse_punch_up_votes(self, response: str, num_options: int) -> List[int]:
        """Parse which punch-ups to apply. Returns list of 0-indexed punch-up numbers."""
        response_upper = response.upper()

        # Check for NONE
        if "NONE" in response_upper:
            return []

        # Extract numbers
        votes = []
        import re
        numbers = re.findall(r'\d+', response)
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= num_options:
                votes.append(num - 1)  # 0-indexed

        return list(set(votes))  # Remove duplicates

    def apply_punch_ups(self, bit: BitConcept, punch_ups: List[Dict]) -> BitConcept:
        """
        Apply accepted punch-ups to a bit.
        In practice, this just marks them as applied - the actual text changes
        are noted for the scene generator to incorporate.
        """
        bit.punched_up = True
        bit.punch_ups_applied = [pu['suggestion'] for pu in punch_ups]
        return bit

    def curate_lineup_from_bits(self, bits: List[BitConcept]) -> List[BitConcept]:
        """
        Curate a lineup from pre-selected bits (after punch-up round).
        Reorder to avoid back-to-back same formats where possible.

        Args:
            bits: List of BitConcepts (already selected/punched-up)

        Returns:
            Ordered list of BitConcepts for the final lineup
        """
        if not bits:
            return []

        if len(bits) <= 1:
            return bits

        # Simple curation: try to avoid same format back-to-back
        curated = [bits[0]]
        remaining = bits[1:]

        while remaining:
            last_format = curated[-1].format.lower()

            # Find a bit with different format if possible
            different_format = None
            for i, bit in enumerate(remaining):
                if bit.format.lower() != last_format:
                    different_format = i
                    break

            if different_format is not None:
                curated.append(remaining.pop(different_format))
            else:
                # No choice, just take next one
                curated.append(remaining.pop(0))

        return curated


# ============================================================================
# PARODY CARD SYSTEM - Robot Chicken/SNL Style Picks for Humans
# ============================================================================

# Pool of parody targets - Robot Chicken style nostalgia/pop culture through interdimensional cable lens
PARODY_TARGET_CARDS = [
    # 80s/90s Cartoons - the Robot Chicken bread and butter
    {"target": "He-Man and the Masters of the Universe", "format": "80s cartoon", "desc": "Buff blonde in furry underwear, 'I HAVE THE POWER'"},
    {"target": "G.I. Joe", "format": "80s cartoon", "desc": "Real American heroes, PSAs, Cobra never wins"},
    {"target": "Transformers", "format": "80s cartoon", "desc": "Robots in disguise, Optimus speeches, toy commercials"},
    {"target": "Thundercats", "format": "80s cartoon", "desc": "Cat people, Sword of Omens, 'HOOOOO!'"},
    {"target": "Teenage Mutant Ninja Turtles", "format": "80s cartoon", "desc": "Pizza-obsessed turtle warriors, cowabunga"},
    {"target": "Care Bears", "format": "80s cartoon", "desc": "Feelings-based warfare, tummy symbols, aggressive caring"},
    {"target": "My Little Pony (G1)", "format": "80s cartoon", "desc": "Pastel horses, friendship, surprisingly dark villains"},
    {"target": "Jem and the Holograms", "format": "80s cartoon", "desc": "Truly outrageous, hologram earrings, band rivalry"},
    {"target": "Rainbow Brite", "format": "80s cartoon", "desc": "Color-based magic, star sprinkles, fighting grey"},
    {"target": "Voltron", "format": "80s cartoon", "desc": "Lions combining, forming blazing sword, defender of universe"},

    # Kids' Shows - innocence ready to be corrupted
    {"target": "Sesame Street", "format": "children's TV", "desc": "Muppets teaching ABCs, brought to you by letters"},
    {"target": "Barney", "format": "children's TV", "desc": "Purple dinosaur, aggressive love, imagination"},
    {"target": "Teletubbies", "format": "children's TV", "desc": "Nightmare creatures with TV stomachs, eh-oh"},
    {"target": "Blue's Clues", "format": "children's TV", "desc": "Dog leaves clues, host talks to camera, thinking chair"},
    {"target": "Dora the Explorer", "format": "children's TV", "desc": "Bilingual adventurer, backpack, swiper no swiping"},
    {"target": "Mr. Rogers' Neighborhood", "format": "children's TV", "desc": "Cardigan gentleman, land of make-believe, wholesomeness"},
    {"target": "Reading Rainbow", "format": "children's TV", "desc": "LeVar Burton, book reviews, butterfly in the sky"},
    {"target": "Lamb Chop's Play-Along", "format": "children's TV", "desc": "Sock puppet, song that never ends, Shari Lewis"},

    # Cereal Mascots - corporate characters with tragic backstories
    {"target": "Trix Rabbit", "format": "cereal mascot", "desc": "Rabbit denied cereal by cruel children, 'silly rabbit'"},
    {"target": "Lucky Charms Leprechaun", "format": "cereal mascot", "desc": "Hunted for his marshmallows, 'magically delicious'"},
    {"target": "Tony the Tiger", "format": "cereal mascot", "desc": "THEY'RE GREAT, aggressive enthusiasm, frosted flakes"},
    {"target": "Toucan Sam", "format": "cereal mascot", "desc": "Follows his nose, fruit loops, British bird"},
    {"target": "Cap'n Crunch", "format": "cereal mascot", "desc": "Naval captain selling mouth-destroying cereal"},
    {"target": "Count Chocula", "format": "cereal mascot", "desc": "Vampire selling chocolate cereal, monster friends"},
    {"target": "Snap Crackle Pop", "format": "cereal mascot", "desc": "Elf trio, rice krispies, concerning sound effects"},

    # Classic Franchises - icons ready for deconstruction
    {"target": "Star Wars", "format": "sci-fi franchise", "desc": "Force, lightsabers, daddy issues, space opera"},
    {"target": "Star Trek", "format": "sci-fi franchise", "desc": "Federation, boldly going, Kirk speeches, red shirts"},
    {"target": "Disney Princesses", "format": "animated franchise", "desc": "Rescued maidens, animal friends, true love's kiss"},
    {"target": "DC Superheroes", "format": "comic franchise", "desc": "Batman brooding, Superman perfect, Justice League"},
    {"target": "Mario Bros", "format": "video game", "desc": "Plumber rescues princess, mushrooms, turtle stomping"},
    {"target": "Zelda", "format": "video game", "desc": "Silent hero, broken pots, it's dangerous to go alone"},
    {"target": "Sonic the Hedgehog", "format": "video game", "desc": "Gotta go fast, rings, attitude with a 'tude"},
    {"target": "Pokemon", "format": "video game/anime", "desc": "Catch em all, cockfighting but cute, Pikachu"},
    {"target": "Power Rangers", "format": "90s show", "desc": "Teenagers with attitude, combining robots, morphin time"},
    {"target": "Scooby-Doo", "format": "cartoon", "desc": "Meddling kids, it was old man Jenkins, Scooby snacks"},

    # Toys & Toy Lines - childhood objects with lives
    {"target": "Barbie", "format": "toy line", "desc": "Impossibly proportioned doll, dream house, Ken"},
    {"target": "G.I. Joe (toys)", "format": "toy line", "desc": "Action figures, kung-fu grip, real American hero"},
    {"target": "Cabbage Patch Kids", "format": "toy line", "desc": "Adopted dolls, birth certificates, 80s riots"},
    {"target": "Hot Wheels", "format": "toy line", "desc": "Orange track, loop-de-loops, tiny cars"},
    {"target": "LEGO", "format": "toy line", "desc": "Building blocks, stepping on them, everything is awesome"},
    {"target": "Stretch Armstrong", "format": "toy", "desc": "Stretchy man filled with corn syrup, pull him"},
    {"target": "Easy-Bake Oven", "format": "toy", "desc": "Lightbulb cooking, tiny cakes, child chef"},
    {"target": "Furby", "format": "toy", "desc": "Owl-hamster hybrid, learns to talk, eyes that watch"},
    {"target": "Teddy Ruxpin", "format": "toy", "desc": "Animatronic bear, cassette tapes, dead eyes"},
    {"target": "Tamagotchi", "format": "toy", "desc": "Digital pet, constant beeping, inevitable death"},

    # Classic Commercials & PSAs - 80s/90s ad culture
    {"target": "This Is Your Brain On Drugs", "format": "PSA", "desc": "Egg in frying pan, any questions, dramatic"},
    {"target": "McGruff the Crime Dog", "format": "PSA", "desc": "Take a bite out of crime, trenchcoat dog"},
    {"target": "Smokey Bear", "format": "PSA", "desc": "Only YOU can prevent forest fires, stern bear"},
    {"target": "G.I. Joe PSAs", "format": "PSA", "desc": "Now you know, and knowing is half the battle"},
    {"target": "Got Milk?", "format": "commercial", "desc": "Milk mustaches, celebrities, dairy propaganda"},
    {"target": "Where's the Beef?", "format": "commercial", "desc": "Old lady, small hamburger, catchphrase"},
    {"target": "Life Alert", "format": "commercial", "desc": "I've fallen and I can't get up, elderly panic"},
    {"target": "Chia Pet", "format": "commercial", "desc": "Ch-ch-ch-chia, terracotta animals, grows hair"},

    # Tech Bros & Modern Dystopia - SNL territory
    {"target": "Elon Musk product launches", "format": "tech presentation", "desc": "Cybertruck windows, Mars promises, memes as strategy"},
    {"target": "Apple keynotes", "format": "tech presentation", "desc": "One more thing, revolutionary, courage"},
    {"target": "Amazon/Bezos", "format": "corporate", "desc": "Pee bottles, space penis rocket, warehouse vibes"},
    {"target": "Zuckerberg metaverse demos", "format": "tech presentation", "desc": "Dead eyes, legs announcement, 'this is going to be great'"},
    {"target": "AI chatbots", "format": "tech", "desc": "I'm sorry I can't do that, hallucinating, helpful assistant"},
    {"target": "Crypto/NFT culture", "format": "finance bro", "desc": "WAGMI, rug pulls, laser eyes, diamond hands"},
    {"target": "Tech startup pitch decks", "format": "presentation", "desc": "Disrupting, hockey stick growth, we're like Uber for..."},
    {"target": "Smart home devices", "format": "tech ad", "desc": "Alexa listening, IoT everything, fridges with screens"},

    # Celebrity & Influencer Hellscape - SNL's bread and butter
    {"target": "Kardashians", "format": "reality TV", "desc": "Vocal fry, BBLs, 'bible', crying face"},
    {"target": "Joe Rogan podcast", "format": "podcast", "desc": "DMT, chimps, 'it's entirely possible', sauna"},
    {"target": "True crime obsession", "format": "cultural phenomenon", "desc": "Murder podcasts, armchair detectives, 'stay sexy'"},
    {"target": "Gender reveal parties", "format": "social media trend", "desc": "Explosions, fires, 'it's a boy' disasters"},
    {"target": "MLM huns", "format": "social media", "desc": "Hey girl, boss babe, pyramid scheme denial"},
    {"target": "Mukbang", "format": "YouTube", "desc": "Eating massive amounts on camera, slurping, ASMR"},
    {"target": "Family vloggers", "format": "YouTube", "desc": "Exploiting kids for content, fake enthusiasm, 'WHATS UP GUYS'"},
    {"target": "TikTok trends", "format": "social media", "desc": "Dances, challenges, 'put a finger down', devious licks"},

    # Prestige TV & Film Tropes - cultural touchstones
    {"target": "True Detective Season 1", "format": "prestige TV", "desc": "Time is a flat circle, McConaughey monologues, cosmic horror"},
    {"target": "Breaking Bad", "format": "prestige TV", "desc": "I am the one who knocks, chemistry, descent into evil"},
    {"target": "The Office (talking heads)", "format": "sitcom", "desc": "Looking at camera, 'that's what she said', cringe"},
    {"target": "Law & Order", "format": "procedural", "desc": "DUN DUN, ripped from headlines, 'in the criminal justice system'"},
    {"target": "CSI", "format": "procedural", "desc": "Enhance, sunglasses, impossible forensics"},
    {"target": "Hallmark Christmas movies", "format": "TV movie", "desc": "City girl, small town, falls for local, saves Christmas"},
    {"target": "Marvel post-credits scenes", "format": "film", "desc": "Teasing next movie, Thanos sitting, 'I'll do it myself'"},
    {"target": "Jump scare horror", "format": "horror film", "desc": "Quiet quiet quiet LOUD, cat fake-out, mirror scare"},

    # Food & Lifestyle Absurdity
    {"target": "Guy Fieri", "format": "food TV", "desc": "Flavortown, frosted tips, 'winner winner chicken dinner'"},
    {"target": "Tasty/BuzzFeed recipe videos", "format": "social media", "desc": "Overhead hands, 'bake until golden', impossible recipes"},
    {"target": "Competitive eating", "format": "sport", "desc": "Hot dogs, Joey Chestnut, speed and suffering"},
    {"target": "HGTV renovation shows", "format": "reality TV", "desc": "Open concept, shiplap, 'we need to demo this'"},
    {"target": "Doomsday preppers", "format": "reality TV", "desc": "Bunkers, MREs, 'when SHTF', bug-out bags"},
    {"target": "Extreme couponing", "format": "reality TV", "desc": "Stockpiles, binder systems, paying $0.03"},
]

# Twist angles - what makes the parody dark/absurd/funny (interdimensional cable energy)
TWIST_CARDS = [
    # Dark/Sinister
    {"twist": "the host is clearly in a cult", "tone": "sinister"},
    {"twist": "it's actually a front for money laundering", "tone": "criminal"},
    {"twist": "the target audience is clearly serial killers", "tone": "dark"},
    {"twist": "the testimonials are clearly coerced", "tone": "dark"},
    {"twist": "customer reviews reveal a body count", "tone": "dark comedy"},
    {"twist": "the fine print reveals something horrifying", "tone": "dark"},
    {"twist": "the whole thing is a hostage situation", "tone": "tense"},

    # Existential/Cosmic Horror
    {"twist": "everyone involved is visibly dead inside", "tone": "existential"},
    {"twist": "the host slowly realizes their reality isn't real", "tone": "cosmic horror"},
    {"twist": "time moves wrong - characters age mid-sentence", "tone": "cosmic"},
    {"twist": "the fourth wall keeps breaking and it hurts them", "tone": "meta horror"},
    {"twist": "everything is a loop and they're becoming aware", "tone": "existential"},
    {"twist": "the laugh track is coming from inside the studio", "tone": "cosmic horror"},
    {"twist": "gravity works differently and nobody mentions it", "tone": "surreal"},

    # Body Horror/Grotesque
    {"twist": "everyone's proportions are subtly wrong", "tone": "uncanny valley"},
    {"twist": "the mascot costume is clearly full of something alive", "tone": "body horror"},
    {"twist": "teeth where teeth shouldn't be", "tone": "body horror"},
    {"twist": "the host's face keeps sliding around", "tone": "grotesque"},
    {"twist": "someone's skin doesn't fit quite right", "tone": "body horror"},
    {"twist": "the food looks back", "tone": "grotesque"},

    # Absurdist/Surreal
    {"twist": "everyone is an alien poorly pretending to be human", "tone": "absurd"},
    {"twist": "the product is sentient and angry", "tone": "sci-fi"},
    {"twist": "it's been running for 30 years and no one can stop it", "tone": "surreal"},
    {"twist": "it only airs at 3am and no one knows why", "tone": "mysterious"},
    {"twist": "the host has been legally dead for 5 years", "tone": "supernatural"},
    {"twist": "the set is clearly a different place each cut", "tone": "surreal"},
    {"twist": "they keep referencing a war that never happened", "tone": "alt history"},
    {"twist": "the product exists in a dimension where it makes sense", "tone": "dimensional"},
    {"twist": "the commercial breaks are for products that don't exist yet", "tone": "temporal"},

    # Meta/Self-Aware
    {"twist": "the host keeps breaking character to argue with producers", "tone": "meta"},
    {"twist": "the host is having a complete mental breakdown", "tone": "uncomfortable"},
    {"twist": "it's a thinly veiled cry for help", "tone": "sad"},
    {"twist": "the spokesperson doesn't believe any of this", "tone": "cynical"},
    {"twist": "they're clearly improvising because the script caught fire", "tone": "chaotic"},
    {"twist": "the cue cards are visible and say something different", "tone": "meta"},
    {"twist": "someone off-camera keeps sobbing", "tone": "uncomfortable"},

    # Ironic/Late Capitalism
    {"twist": "the product causes the problem it claims to solve", "tone": "ironic"},
    {"twist": "it's way too honest about what it really is", "tone": "brutal honesty"},
    {"twist": "the product works TOO well", "tone": "monkey's paw"},
    {"twist": "it's clearly a pyramid scheme but no one cares", "tone": "late capitalism"},
    {"twist": "it's sponsored by an evil corporation that doesn't hide it", "tone": "dystopian"},
    {"twist": "the before/after reveals something disturbing", "tone": "horror"},
    {"twist": "the disclaimers take longer than the ad", "tone": "legal horror"},
    {"twist": "the phone number connects to something you don't want", "tone": "creepy"},
]

# Character archetypes - who's delivering this bit (interdimensional cable voices)
CHARACTER_CARDS = [
    # Classic Infomercial Types
    {"archetype": "Manic infomercial host", "voice": "loud, fast-talking, never blinks, keeps saying 'but wait'", "look": "cheap suit, sweat stains, too much teeth"},
    {"archetype": "Dead-eyed corporate spokesperson", "voice": "flat affect, rehearsed warmth, occasional glitch", "look": "business casual, vacant smile, uncanny valley"},
    {"archetype": "Overly enthusiastic local business owner", "voice": "screaming, regional accent, way too close to camera", "look": "polo with company logo, pointing aggressively"},
    {"archetype": "Desperate QVC host", "voice": "increasingly frantic as stock runs low, pleading", "look": "jewelry, blazer, holding product like a baby"},
    {"archetype": "Aggressive lawyer", "voice": "yelling, pointing, 'IF YOU OR A LOVED ONE'", "look": "bad suit, American flag, gavel"},

    # Weird Creature Presenters (interdimensional cable)
    {"archetype": "Guy with ants in his eyes", "voice": "can't see prices, everything hurts, still selling", "look": "ants crawling across eyeballs, still smiling"},
    {"archetype": "Sentient furniture", "voice": "trying too hard to relate to humans, creaky", "look": "talking lamp or chair, googly eyes glued on"},
    {"archetype": "Three beings in a trenchcoat", "voice": "taking turns speaking, slightly out of sync", "look": "suspiciously tall, trenchcoat bulging, extra hands"},
    {"archetype": "Interdimensional tourist", "voice": "fascinated by mundane things, wrong emphasis on words", "look": "human-ish, colors slightly off, too many fingers"},
    {"archetype": "Probably a demon", "voice": "ominous bass, legally distinct from Satan", "look": "business suit, red skin, tiny horns, briefcase"},
    {"archetype": "Shapeshifter who can't hold it together", "voice": "keeps changing mid-sentence", "look": "features melting between takes"},

    # Uncomfortable Human Presenters
    {"archetype": "Host having a breakdown", "voice": "starting calm, unraveling, occasional sob", "look": "makeup running, tie loosening, thousand yard stare"},
    {"archetype": "Hostage reading script", "voice": "monotone, coded blinking, emphasizing weird words", "look": "bruised, exhausted, help me eyes"},
    {"archetype": "Washed-up celebrity", "voice": "tired, clearly reading cue cards, 'what has my life become'", "look": "faded glamour, bad plastic surgery, haunted"},
    {"archetype": "Guy who just woke up", "voice": "confused, possibly drugged, where am I", "look": "bathrobe, bedhead, studio lights hurting"},
    {"archetype": "Someone who walked onto the wrong set", "voice": "increasingly panicked, 'this isn't what I auditioned for'", "look": "different costume, looking for exit"},

    # Archetype Parodies
    {"archetype": "Smug tech bro", "voice": "condescending, disrupting, 'let me explain'", "look": "Patagonia vest, AirPods, rehearsed TED Talk gesture"},
    {"archetype": "Wellness grifter", "voice": "breathy, everything is 'intentional', jade egg energy", "look": "flowy neutrals, suspicious glow, holding crystals"},
    {"archetype": "True crime podcast host", "voice": "dramatic pauses, ad reads mid-murder, 'let's get into it'", "look": "dim lighting, wine glass, murder board visible"},
    {"archetype": "Unhinged mascot", "voice": "muffled screaming from inside suit, method acting", "look": "costume falling apart, dead eyes, bloodstains"},
    {"archetype": "AI trying to be human", "voice": "uncanny cadence, wrong idioms, 'as a human myself'", "look": "too symmetrical, blinks at wrong times"},
    {"archetype": "Cult leader doing an ad", "voice": "warm, welcoming, 'we're all family here', eyes don't match smile", "look": "all white, nametag, pamphlets, compound visible"},
]


@dataclass
class HumanParodySelection:
    """Tracks a human's card selections for parody pitch."""
    user_name: str
    parody_card_idx: Optional[int] = None  # Which parody target they picked
    twist_card_idx: Optional[int] = None   # Which twist they picked
    character_card_idx: Optional[int] = None  # Which character they picked
    dealt_parody_cards: List[int] = None  # Indices into PARODY_TARGET_CARDS
    dealt_twist_cards: List[int] = None   # Indices into TWIST_CARDS
    dealt_character_cards: List[int] = None  # Indices into CHARACTER_CARDS

    def __post_init__(self):
        if self.dealt_parody_cards is None:
            self.dealt_parody_cards = []
        if self.dealt_twist_cards is None:
            self.dealt_twist_cards = []
        if self.dealt_character_cards is None:
            self.dealt_character_cards = []


class HumanParodyCardSystem:
    """
    Card-based pitch system for humans in IDCC.

    Instead of typing full pitches, humans pick from dealt cards:
    1. Pick a PARODY TARGET (what are we making fun of)
    2. Pick a TWIST (what's the dark/absurd angle)
    3. Pick a CHARACTER (who delivers the bit)

    System combines their picks into a complete BitConcept.
    """

    def __init__(self, game_id: str):
        self.game_id = game_id
        self.selections: Dict[str, HumanParodySelection] = {}
        self.current_round: str = ""  # "parody", "twist", "character"

    def register_human(self, user_name: str):
        """Register a human participant."""
        self.selections[user_name] = HumanParodySelection(user_name=user_name)

    def deal_cards(self, user_name: str, cards_per_category: int = 4) -> Dict[str, List[Dict]]:
        """
        Deal random cards to a human from each category.

        Returns dict with keys: parody_cards, twist_cards, character_cards
        Each value is a list of card dicts with their display info.
        """
        import random

        if user_name not in self.selections:
            self.register_human(user_name)

        selection = self.selections[user_name]

        # Deal from each pool
        parody_indices = random.sample(range(len(PARODY_TARGET_CARDS)), min(cards_per_category, len(PARODY_TARGET_CARDS)))
        twist_indices = random.sample(range(len(TWIST_CARDS)), min(cards_per_category, len(TWIST_CARDS)))
        character_indices = random.sample(range(len(CHARACTER_CARDS)), min(cards_per_category, len(CHARACTER_CARDS)))

        selection.dealt_parody_cards = parody_indices
        selection.dealt_twist_cards = twist_indices
        selection.dealt_character_cards = character_indices

        return {
            "parody_cards": [PARODY_TARGET_CARDS[i] for i in parody_indices],
            "twist_cards": [TWIST_CARDS[i] for i in twist_indices],
            "character_cards": [CHARACTER_CARDS[i] for i in character_indices],
        }

    def format_parody_cards(self, user_name: str) -> str:
        """Format dealt parody target cards for Discord display."""
        if user_name not in self.selections:
            return "No cards dealt."

        selection = self.selections[user_name]
        if not selection.dealt_parody_cards:
            return "No parody cards dealt."

        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]
        lines = ["🎯 **PICK A PARODY TARGET** (type 1-4):\n"]

        for i, idx in enumerate(selection.dealt_parody_cards):
            card = PARODY_TARGET_CARDS[idx]
            emoji = emojis[i] if i < len(emojis) else f"({i+1})"
            lines.append(f"{emoji} **{card['target']}** ({card['format']})")
            lines.append(f"   *{card['desc']}*\n")

        return "\n".join(lines)

    def format_twist_cards(self, user_name: str) -> str:
        """Format dealt twist cards for Discord display."""
        if user_name not in self.selections:
            return "No cards dealt."

        selection = self.selections[user_name]
        if not selection.dealt_twist_cards:
            return "No twist cards dealt."

        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]
        lines = ["🔀 **PICK A TWIST** (type 1-4):\n"]

        for i, idx in enumerate(selection.dealt_twist_cards):
            card = TWIST_CARDS[idx]
            emoji = emojis[i] if i < len(emojis) else f"({i+1})"
            lines.append(f"{emoji} **{card['twist']}**")
            lines.append(f"   *({card['tone']})*\n")

        return "\n".join(lines)

    def format_character_cards(self, user_name: str) -> str:
        """Format dealt character cards for Discord display."""
        if user_name not in self.selections:
            return "No cards dealt."

        selection = self.selections[user_name]
        if not selection.dealt_character_cards:
            return "No character cards dealt."

        emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣"]
        lines = ["👤 **PICK A CHARACTER** (type 1-4):\n"]

        for i, idx in enumerate(selection.dealt_character_cards):
            card = CHARACTER_CARDS[idx]
            emoji = emojis[i] if i < len(emojis) else f"({i+1})"
            lines.append(f"{emoji} **{card['archetype']}**")
            lines.append(f"   Voice: *{card['voice']}*")
            lines.append(f"   Look: *{card['look']}*\n")

        return "\n".join(lines)

    def format_all_cards(self, user_name: str) -> str:
        """Format all three card categories for a single display."""
        parody = self.format_parody_cards(user_name)
        twist = self.format_twist_cards(user_name)
        character = self.format_character_cards(user_name)

        return f"{parody}\n---\n{twist}\n---\n{character}"

    def parse_selection(self, user_name: str, message: str, round_type: str) -> Optional[int]:
        """
        Parse a human's card selection.

        Args:
            user_name: Who's selecting
            message: Their message (should contain a number 1-4)
            round_type: "parody", "twist", or "character"

        Returns:
            The selected card index (0-based), or None if invalid
        """
        if user_name not in self.selections:
            return None

        selection = self.selections[user_name]

        # Get the dealt cards for this round
        if round_type == "parody":
            dealt = selection.dealt_parody_cards
        elif round_type == "twist":
            dealt = selection.dealt_twist_cards
        elif round_type == "character":
            dealt = selection.dealt_character_cards
        else:
            return None

        if not dealt:
            return None

        # Parse number from message
        for char in message.strip():
            if char.isdigit():
                num = int(char)
                if 1 <= num <= len(dealt):
                    idx = num - 1  # Convert to 0-based
                    # Store the selection
                    if round_type == "parody":
                        selection.parody_card_idx = idx
                    elif round_type == "twist":
                        selection.twist_card_idx = idx
                    elif round_type == "character":
                        selection.character_card_idx = idx
                    return idx

        return None

    def build_bit_from_selections(self, user_name: str) -> Optional[BitConcept]:
        """
        Build a BitConcept from a human's card selections.

        Returns None if they haven't made all selections.
        """
        if user_name not in self.selections:
            return None

        selection = self.selections[user_name]

        # Check all selections made
        if selection.parody_card_idx is None or \
           selection.twist_card_idx is None or \
           selection.character_card_idx is None:
            return None

        # Get the actual cards
        parody_pool_idx = selection.dealt_parody_cards[selection.parody_card_idx]
        twist_pool_idx = selection.dealt_twist_cards[selection.twist_card_idx]
        char_pool_idx = selection.dealt_character_cards[selection.character_card_idx]

        parody_card = PARODY_TARGET_CARDS[parody_pool_idx]
        twist_card = TWIST_CARDS[twist_pool_idx]
        char_card = CHARACTER_CARDS[char_pool_idx]

        # Build the bit
        return BitConcept(
            parody_target=parody_card["target"],
            twist=twist_card["twist"],
            format=parody_card["format"],
            character_description=char_card["look"],
            vocal_specs=char_card["voice"],
            sample_dialogue="",  # Human can add this or it can be generated
            punchline="",  # Human can add this or it can be generated
            pitched_by=user_name,
        )

    def get_selection_summary(self, user_name: str) -> str:
        """Get a summary of what a human has selected so far."""
        if user_name not in self.selections:
            return "No selections."

        selection = self.selections[user_name]
        lines = [f"**{user_name}'s Pitch:**"]

        if selection.parody_card_idx is not None and selection.dealt_parody_cards:
            card = PARODY_TARGET_CARDS[selection.dealt_parody_cards[selection.parody_card_idx]]
            lines.append(f"🎯 Parody: **{card['target']}**")

        if selection.twist_card_idx is not None and selection.dealt_twist_cards:
            card = TWIST_CARDS[selection.dealt_twist_cards[selection.twist_card_idx]]
            lines.append(f"🔀 Twist: **{card['twist']}**")

        if selection.character_card_idx is not None and selection.dealt_character_cards:
            card = CHARACTER_CARDS[selection.dealt_character_cards[selection.character_card_idx]]
            lines.append(f"👤 Character: **{card['archetype']}**")

        return "\n".join(lines)


# ============================================================================
# GAME STATE
# ============================================================================
# Note: Scene generation prompts (IDCC_COLD_OPEN, IDCC_CONTINUATION, IDCC_FINAL_SCENE)
# are now in game_prompts.py and accessed through the game context system.


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
class BitConcept:
    """
    A single self-contained comedy bit for one clip.

    Robot Chicken / SNL style: direct parodies of pop culture, TV, movies,
    ads, politics, modern life. Each bit is a specific parody with a twist.

    Example bits:
    - PARODY: Peloton ads → TWIST: instructor is recruiting for doomsday cult
    - PARODY: House Hunters → TWIST: couple shopping for murder dungeon
    - PARODY: Dr. Phil → TWIST: guest is a Lovecraftian horror
    - PARODY: Pharma ads → TWIST: side effects ARE the product
    - PARODY: Gordon Ramsay → TWIST: reviewing prison food
    - PARODY: Dating app ads → TWIST: everyone is visibly dead inside
    - PARODY: Amazon Alexa → TWIST: what she hears at 3am
    - PARODY: Local news → TWIST: anchors clearly hate each other
    """
    # WHAT are we parodying? (specific show/movie/ad/cultural phenomenon)
    parody_target: str = ""  # "Peloton ads", "House Hunters", "Dr. Phil", "pharma commercials"

    # THE TWIST - what's the comedic angle/conceit?
    twist: str = ""  # "instructor is recruiting for a doomsday cult"

    # Format/genre of this specific bit
    format: str = ""  # "infomercial", "talk show", "movie trailer", "reality show"

    # Legacy fields (for backwards compatibility with old-style pitches)
    premise: str = ""  # Legacy: what the bit is about
    comedic_hook: str = ""  # Legacy: what makes it funny

    # Character description (EXACT visual - copy-pasted to Sora)
    character_description: str = ""  # "Ultra-fit woman with unsettling smile, all-white athletic wear"

    # Vocal specs (how they SOUND for TTS)
    vocal_specs: str = ""  # "breathy, inspirational, slightly manic"

    # Sample dialogue that captures the voice and lands the joke
    sample_dialogue: str = ""  # "Feel the burn... of eternal salvation. Keep pedaling toward the light."

    # The punchline/button that ends the bit
    punchline: str = ""  # "Join us. There's no leaving the leaderboard."

    # Duration-calibrated scope (set based on clip_duration_seconds)
    duration_beats: int = 2

    # Shot direction for this bit's format
    shot_direction: str = ""  # "Sleek cycling studio, ominous backlighting"

    # Who pitched this bit (for credits/variety tracking)
    pitched_by: str = ""

    # Punch-up status (after writers room Round 3)
    punched_up: bool = False
    punch_ups_applied: list = None  # List of accepted punch-ups


def get_duration_beats(clip_duration_seconds: int) -> int:
    """
    Calculate appropriate beat count for a given clip duration.

    4s  = 1 beat  (single gag/reaction)
    8s  = 2 beats (setup + payoff)
    12s = 3 beats (full mini-skit: setup, escalation, punchline)
    """
    if clip_duration_seconds <= 4:
        return 1
    elif clip_duration_seconds <= 8:
        return 2
    else:
        return 3


def get_duration_scope_description(clip_duration_seconds: int) -> str:
    """
    Get a human-readable description of what fits in this duration.
    Used to instruct agents on appropriate scope.
    """
    beats = get_duration_beats(clip_duration_seconds)
    if beats == 1:
        return "ONE BEAT ONLY: Single gag or punchline. One line of dialogue max. Think: Lil' Bits whisper 'Eat some shit you stupid bitch... just kidding.'"
    elif beats == 2:
        return "TWO BEATS: Setup + Payoff. Establish premise, land the joke. Think: Ants in My Eyes Johnson intro - 'I can't see anything!' *knocks over TVs*"
    else:
        return "THREE BEATS: Full mini-skit. Setup, escalation, button. Think: Real Fake Doors - introduce showroom, try doors that won't open, 'Come on down!'"


@dataclass
class IDCCChannelLineup:
    """
    The channel lineup for a Robot Chicken style IDCC session.

    Instead of one cohesive story, we have N independent bits.
    TV static between bits = channel surfing through dimensions.
    """
    # Array of bits, one per clip slot
    bits: List[BitConcept] = field(default_factory=list)

    # Raw writers room log for context
    writers_room_log: List[str] = field(default_factory=list)

    # Duration setting (affects beat count per bit)
    clip_duration_seconds: int = 12

    def get_bit(self, clip_number: int) -> Optional[BitConcept]:
        """Get the bit for a specific clip (1-indexed)."""
        if 0 < clip_number <= len(self.bits):
            return self.bits[clip_number - 1]
        return None

    def get_next_bit_preview(self, clip_number: int) -> Optional[BitConcept]:
        """Get the NEXT bit for transition preview (mouth closed, silent)."""
        if 0 < clip_number < len(self.bits):
            return self.bits[clip_number]  # clip_number is 1-indexed, so this gets next
        return None


@dataclass
class IDCCGameState:
    """Full game state for an Interdimensional Cable session."""
    game_id: str
    channel_id: int
    start_time: float

    # Registration
    registration_end_time: float = 0
    registered_humans: Set[str] = field(default_factory=set)  # Discord usernames
    registered_human_ids: Dict[str, str] = field(default_factory=dict)  # username -> Discord user ID for @ mentions
    registered_agents: List[str] = field(default_factory=list)  # Agent names (backup)

    # Participants (finalized after registration)
    participants: List[Dict[str, Any]] = field(default_factory=list)  # {name, type, agent_obj}

    # Channel Lineup (Robot Chicken style, array of independent bits)
    channel_lineup: Optional[IDCCChannelLineup] = None

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
    phase: str = "init"  # init, registration, spitballing, awaiting_scene, generating_video, concatenating, complete, failed
    final_video_path: Optional[Path] = None
    final_video_url: Optional[str] = None
    error_message: Optional[str] = None

    # Timing
    total_generation_time: float = 0
    registration_message_id: Optional[int] = None

    # Human scene collection
    pending_human_scenes: Dict[str, str] = field(default_factory=dict)  # username -> scene prompt

    # Spitball phase human inputs
    pending_spitball_inputs: Dict[str, str] = field(default_factory=dict)  # username -> their pitch/vote/character
    spitball_collecting: bool = False  # True when collecting human inputs
    spitball_round_name: str = ""  # Current round: "pitch", "vote_concept", "character", "vote_character"


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
        num_clips: int = 5,
        game_orchestrator=None
    ):
        """
        Initialize the game.

        Args:
            agent_manager: AgentManager instance for accessing agents
            discord_client: DiscordBotClient for sending messages
            num_clips: Number of clips to generate (3-6)
            game_orchestrator: GameOrchestrator for resetting idle timer after completion
        """
        self.agent_manager = agent_manager
        self.discord_client = discord_client
        self.game_orchestrator = game_orchestrator

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

            # Phase 3: Writers Room (Robot Chicken style - establish Channel Lineup)
            await self._run_writers_room_phase(ctx)

            if not self.state.channel_lineup or not self.state.channel_lineup.bits:
                await self._send_gamemaster_message(
                    "**ERROR:** Failed to establish channel lineup. Aborting."
                )
                self.state.phase = "failed"
                return None

            # Phase 4: Generate clips
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

            # Reset auto-play idle timer so we don't immediately start another game
            if self.game_orchestrator:
                self.game_orchestrator.update_human_activity()
                logger.info(f"[IDCC:{self.game_id}] Reset auto-play idle timer after game completion")

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
            # Reset idle timer even on failure to prevent rapid retry loops
            if self.game_orchestrator:
                self.game_orchestrator.update_human_activity()
            # Cleanup old temp files
            cleanup_temp_files(max_age_hours=24)

    async def handle_join_command(self, user_name: str, user_id: Optional[str] = None) -> bool:
        """
        Handle a !join-idcc command during registration.

        Args:
            user_name: Discord username of the joiner
            user_id: Discord user ID for @ mentions (optional, but recommended)

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
            # Store user ID for @ mentions later
            if user_id:
                self.state.registered_human_ids[user_name] = user_id
                logger.info(f"[IDCC:{self.game_id}] {user_name} (ID: {user_id}) joined! ({len(self.state.registered_humans)} humans)")
            else:
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

    async def handle_spitball_submission(self, user_name: str, message_content: str) -> bool:
        """
        Handle a spitball submission from a human participant during writers' room.

        Args:
            user_name: Discord username
            message_content: Full message content

        Returns:
            True if submission was accepted, False otherwise
        """
        if not self.state:
            return False

        # Only accept during spitballing phase when collecting
        if self.state.phase != "spitballing" or not self.state.spitball_collecting:
            return False

        # Check if this user is a registered human participant
        human_names = [p["name"] for p in self.state.participants if p["type"] == "human"]
        if user_name not in human_names:
            return False

        # Don't accept if they already submitted this round
        if user_name in self.state.pending_spitball_inputs:
            return False

        # Store their input
        content = message_content.strip()
        if content and len(content) > 5:  # Minimum 5 chars
            self.state.pending_spitball_inputs[user_name] = content
            logger.info(f"[IDCC:{self.game_id}] {user_name} submitted spitball ({self.state.spitball_round_name}): {content[:50]}...")
            return True

        return False

    async def _wait_for_human_spitball_inputs(
        self,
        round_name: str,
        human_participants: List[Dict],
        timeout_seconds: int = None
    ) -> Dict[str, str]:
        """
        Wait for human participants to submit their spitball inputs.

        Args:
            round_name: Name of round for logging ("pitch", "vote_concept", etc.)
            human_participants: List of human participant dicts
            timeout_seconds: How long to wait (default from config)

        Returns:
            Dict of username -> their submission
        """
        if timeout_seconds is None:
            timeout_seconds = idcc_config.spitball_round_timeout_seconds

        if not human_participants:
            return {}

        human_names = [p["name"] for p in human_participants]

        # Clear any previous inputs and start collecting
        self.state.pending_spitball_inputs = {}
        self.state.spitball_collecting = True
        self.state.spitball_round_name = round_name

        # Announce waiting for humans with clearer instructions
        human_list = ", ".join(human_names)
        card_rounds = ("pitch", "character", "parody", "twist")
        if round_name in card_rounds:
            instruction = "Type a number (1-4) to pick your card!"
        elif round_name in ("vote_lineup", "vote_concept"):
            instruction = "Type the numbers of your picks (e.g., '1 3 5')!"
        else:
            instruction = "Type your response!"
        await self._send_gamemaster_message(
            f"⏳ **Waiting for humans:** {human_list}\n"
            f"**{instruction}** *({timeout_seconds} seconds)*"
        )

        # Wait with periodic checks
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            # Check if all humans have submitted
            submitted = set(self.state.pending_spitball_inputs.keys())
            if submitted >= set(human_names):
                logger.info(f"[IDCC:{self.game_id}] All humans submitted for {round_name}")
                break

            # Check every 2 seconds
            await asyncio.sleep(2)

            # Reminder at halfway point
            elapsed = time.time() - start_time
            if abs(elapsed - timeout_seconds / 2) < 2:
                remaining = [n for n in human_names if n not in self.state.pending_spitball_inputs]
                if remaining:
                    await self._send_gamemaster_message(
                        f"⏰ **{int(timeout_seconds - elapsed)}s remaining** - Still waiting for: {', '.join(remaining)}"
                    )

        # Stop collecting
        self.state.spitball_collecting = False

        # Report results
        collected = self.state.pending_spitball_inputs.copy()
        submitted_names = list(collected.keys())
        if submitted_names:
            logger.info(f"[IDCC:{self.game_id}] Collected {round_name} from humans: {submitted_names}")
        else:
            logger.info(f"[IDCC:{self.game_id}] No human submissions for {round_name}")

        return collected

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

        # Wait for registration period
        registration_duration = idcc_config.registration_duration_seconds
        check_interval = 15  # Check every 15 seconds

        elapsed = 0
        while elapsed < registration_duration:
            await asyncio.sleep(check_interval)
            elapsed += check_interval

            remaining = registration_duration - elapsed
            human_count = len(self.state.registered_humans)

            if remaining > 0:
                if remaining == 60:  # 60s mark
                    await self._send_gamemaster_message(
                        f"**ONE MINUTE remaining!** {human_count}/{participants_needed} participants."
                    )
                elif remaining == 30:  # 30s mark
                    await self._send_gamemaster_message(
                        f"**30 SECONDS!** Last call for `!join-idcc`! ({human_count}/{participants_needed})"
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
                # Get user ID for @ mentions (if available)
                user_id = self.state.registered_human_ids.get(username)
                participants.append({
                    "name": username,
                    "type": "human",
                    "agent_obj": None,
                    "user_id": user_id  # Discord user ID for @ mentions
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
                    "agent_obj": agent,
                    "user_id": None  # Bots don't have Discord user IDs
                })

        self.state.participants = participants

        # Log final lineup
        lineup = ", ".join([f"{p['name']} ({'human' if p['type'] == 'human' else 'bot'})" for p in participants])
        logger.info(f"[IDCC:{self.game_id}] Final lineup: {lineup}")

        await self._send_gamemaster_message(
            f"**CAST ASSEMBLED:** {lineup}\n\n"
            "Time to brainstorm! The crew will spitball ideas before we start generating..."
        )

    # ========================================================================
    # PHASE 3: COLLABORATIVE SPITBALLING (CONSENSUS-BASED)
    # ========================================================================

    def _parse_vote(self, response: str, voter_name: str, valid_candidates: List[str]) -> Optional[str]:
        """
        Parse a vote from an agent's response.

        Looks for patterns like:
        - "MY VOTE: Agent Name"
        - "I vote for Agent Name"
        - "Vote: Agent Name"

        Returns the voted-for agent name, or None if invalid/self-vote.
        """
        import re

        response_upper = response.upper()

        # Try various vote patterns
        patterns = [
            r"MY VOTE:\s*\*?\*?([^*\n]+)",  # MY VOTE: **Name** or MY VOTE: Name
            r"I VOTE FOR\s*\*?\*?([^*\n]+)",
            r"VOTE:\s*\*?\*?([^*\n]+)",
            r"VOTING FOR\s*\*?\*?([^*\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                voted_name = match.group(1).strip().strip('*').strip()
                # Find the actual candidate name (case-insensitive match)
                for candidate in valid_candidates:
                    if candidate.upper() == voted_name or candidate.upper() in voted_name or voted_name in candidate.upper():
                        # Check for self-vote
                        if candidate.lower() == voter_name.lower():
                            logger.warning(f"[IDCC:{self.game_id}] {voter_name} tried to vote for themselves")
                            return None
                        return candidate

        logger.warning(f"[IDCC:{self.game_id}] Could not parse vote from {voter_name}: {response[:100]}...")
        return None

    def _tally_votes(self, votes: Dict[str, str], candidates: List[str]) -> tuple:
        """
        Tally votes and determine winner.

        Args:
            votes: Dict of voter_name -> voted_for_name
            candidates: List of valid candidate names

        Returns:
            (winner_name, vote_counts_dict, was_tie)
        """
        vote_counts = {c: 0 for c in candidates}

        for voter, voted_for in votes.items():
            if voted_for in vote_counts:
                vote_counts[voted_for] += 1

        # Find max votes
        max_votes = max(vote_counts.values()) if vote_counts else 0
        winners = [c for c, count in vote_counts.items() if count == max_votes]

        if len(winners) == 1:
            return winners[0], vote_counts, False
        else:
            # Tie - pick randomly
            winner = random.choice(winners)
            return winner, vote_counts, True

    # =========================================================================
    # ROBOT CHICKEN STYLE WRITERS ROOM (NEW)
    # =========================================================================

    def _parse_bit_from_response(self, response: str, pitcher_name: str) -> Optional[BitConcept]:
        """
        Parse a BitConcept from an agent's pitch response.

        Expected format (new parody style):
        PARODY_TARGET: [what we're parodying]
        TWIST: [the comedic angle]
        FORMAT: [type]
        CHARACTER_DESCRIPTION: [description]
        VOCAL_SPECS: [voice specs]
        SAMPLE_DIALOGUE: [key lines]
        PUNCHLINE: [the landing]

        Also supports legacy format with PREMISE and COMEDIC_HOOK.
        """
        try:
            lines = response.strip().split('\n')
            data = {}

            current_key = None
            current_value = []

            # All supported fields (new + legacy)
            fields = [
                'PARODY_TARGET', 'TWIST', 'FORMAT', 'PREMISE',
                'CHARACTER_DESCRIPTION', 'VOCAL_SPECS', 'SAMPLE_DIALOGUE',
                'COMEDIC_HOOK', 'PUNCHLINE'
            ]

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check for field labels
                found_field = False
                for field in fields:
                    if line.upper().startswith(field + ':') or line.upper().startswith(field + ' :'):
                        # Save previous field if exists
                        if current_key:
                            data[current_key] = ' '.join(current_value).strip()
                        current_key = field.lower()
                        # Get value after the colon
                        colon_pos = line.find(':')
                        if colon_pos != -1:
                            current_value = [line[colon_pos + 1:].strip()]
                        else:
                            current_value = []
                        found_field = True
                        break

                if not found_field and current_key:
                    # No field label found, append to current value
                    current_value.append(line)

            # Save last field
            if current_key:
                data[current_key] = ' '.join(current_value).strip()

            # Validate required fields - need either parody_target+twist OR premise
            has_parody = data.get('parody_target') and data.get('twist')
            has_legacy = data.get('premise')

            if not data.get('format') or (not has_parody and not has_legacy):
                logger.warning(f"[IDCC:{self.game_id}] Bit from {pitcher_name} missing required fields")
                return None

            # Build the bit - use parody fields if present, fall back to legacy
            return BitConcept(
                parody_target=data.get('parody_target', ''),
                twist=data.get('twist', data.get('comedic_hook', '')),  # Fall back to comedic_hook
                format=data.get('format', ''),
                premise=data.get('premise', ''),  # Legacy field
                comedic_hook=data.get('comedic_hook', ''),  # Legacy field
                character_description=data.get('character_description', ''),
                vocal_specs=data.get('vocal_specs', ''),
                sample_dialogue=data.get('sample_dialogue', ''),
                punchline=data.get('punchline', ''),
                pitched_by=pitcher_name,
                duration_beats=get_duration_beats(idcc_config.clip_duration_seconds)
            )
        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Error parsing bit from {pitcher_name}: {e}")
            return None

    def _parse_lineup_votes(self, response: str, voter_name: str, num_bits: int, own_bit_index: int) -> List[int]:
        """
        Parse vote indices from a voter's response.

        Expected format:
        MY VOTES: 1, 3, 5
        BEST ONE: 3 - [reason]
        """
        votes = []
        try:
            # Look for "MY VOTES:" line
            for line in response.split('\n'):
                if 'MY VOTES' in line.upper() or 'VOTES:' in line.upper():
                    # Extract numbers
                    for char in line:
                        if char.isdigit():
                            num = int(char)
                            if 1 <= num <= num_bits and (num - 1) != own_bit_index:
                                if (num - 1) not in votes:  # Avoid duplicates
                                    votes.append(num - 1)  # Convert to 0-indexed

            # Fallback: just extract any numbers from response
            if not votes:
                for char in response:
                    if char.isdigit():
                        num = int(char)
                        if 1 <= num <= num_bits and (num - 1) != own_bit_index:
                            if (num - 1) not in votes:
                                votes.append(num - 1)

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Error parsing votes from {voter_name}: {e}")

        return votes

    async def _run_writers_room_phase(self, ctx: commands.Context):
        """
        Robot Chicken style writers room - each participant pitches complete bits.

        Two rounds:
        1. Everyone pitches a COMPLETE self-contained bit
        2. Vote for which bits make the N-clip lineup

        Result: IDCCChannelLineup with N independent bits.
        """
        # get_duration_scope_description is defined in this file at line 606

        self.state.phase = "spitballing"
        writers_room_log = []

        # Get participants
        human_participants = [p for p in self.state.participants if p["type"] == "human"]
        agent_participants = [p for p in self.state.participants if p["type"] == "agent" and p["agent_obj"]]

        # Fallback to running agents if no agent participants
        if not agent_participants:
            logger.warning(f"[IDCC:{self.game_id}] No agent participants, using fallback")
            from constants import is_image_model
            available_agents = [
                a for a in self.agent_manager.get_all_agents()
                if a.is_running and not is_image_model(a.model)
            ]
            if available_agents:
                # Need enough agents to pitch at least num_clips bits
                agent_participants = [{"name": a.name, "type": "agent", "agent_obj": a} for a in available_agents[:self.num_clips]]

        # Need at least num_clips pitches to fill the lineup
        bot_writers = agent_participants[:self.num_clips]
        all_writers = human_participants + bot_writers

        if len(all_writers) < self.num_clips:
            logger.warning(f"[IDCC:{self.game_id}] Only {len(all_writers)} writers for {self.num_clips} bits - some bits may repeat")

        if len(all_writers) < 1:
            logger.error(f"[IDCC:{self.game_id}] Need at least 1 participant")
            return

        # Initialize writers room system
        writers_room = WritersRoomSystem(
            game_id=self.game_id,
            num_clips=self.num_clips,
            clip_duration=idcc_config.clip_duration_seconds
        )

        # Register humans
        for human in human_participants:
            writers_room.register_human(human["name"])

        # Get duration scope for prompts
        duration_scope = get_duration_scope_description(idcc_config.clip_duration_seconds)

        # Opening announcement
        await self._send_gamemaster_message(
            "# 📺 WRITERS' ROOM - ROBOT CHICKEN STYLE\n\n"
            f"We're making **{self.num_clips} independent bits** - like channel surfing.\n"
            f"Each bit is **{idcc_config.clip_duration_seconds} seconds** ({duration_scope}).\n\n"
            "**Round 1:** Everyone pitches a COMPLETE bit\n"
            "**Round 2:** Vote for the best bits to make the lineup\n\n"
            "---"
        )

        # =====================================================================
        # ROUND 1: PITCH COMPLETE BITS
        # =====================================================================

        await self._send_gamemaster_message(
            "## Round 1: PITCH YOUR BIT\n\n"
            "Pitch a **complete self-contained bit**. Include:\n"
            "• FORMAT (infomercial, cop show, cooking show, etc.)\n"
            "• PREMISE (the absurd situation)\n"
            "• CHARACTER_DESCRIPTION (exact visual)\n"
            "• VOCAL_SPECS (how they sound)\n"
            "• COMEDIC_HOOK (what's funny)\n"
            "• PUNCHLINE (the landing line)\n\n"
            "*Think: Real Fake Doors, Ants in My Eyes Johnson, Lil' Bits...*"
        )

        # Load previously used pitches for silent rejection
        used_pitches_history = config_manager.load_idcc_pitch_history()
        used_pitches_lower = [p.lower() for p in used_pitches_history]

        # Enter game mode for bot writers
        for participant in bot_writers:
            agent = participant["agent_obj"]
            game_context_manager.enter_game_mode(agent=agent, game_name="idcc_pitch_complete_bit")
            game_context_manager.update_idcc_context(
                agent_name=agent.name,
                phase="idcc_pitch_complete_bit",
                num_clips=self.num_clips
            )
            # Set turn context with duration info
            game_context_manager.update_turn_context(
                agent_name=agent.name,
                turn_context=f"\nClip duration: {idcc_config.clip_duration_seconds} seconds\nScope: {duration_scope}"
            )

        pitched_bits = []  # List of BitConcept objects

        # Calculate pitches needed - at least num_clips, ideally a few extra for voting variety
        target_pitches = max(self.num_clips, self.num_clips + 2)
        min_required = self.num_clips  # MUST have at least this many
        num_agents = len(bot_writers)

        # Get bot pitches - keep trying until we have enough
        await self._send_gamemaster_message("*Writers are pitching their bits...*")

        pitch_round = 0
        max_rounds = 10  # Safety limit to prevent infinite loops
        while len(pitched_bits) < target_pitches and pitch_round < max_rounds:
            pitch_round += 1

            # After getting minimum required, we can be less aggressive
            if len(pitched_bits) >= min_required and pitch_round > 3:
                break

            round_prompts = [
                f"Pitch your complete bit. {idcc_config.clip_duration_seconds} second clip. {duration_scope}",
                f"Pitch ANOTHER bit - something COMPLETELY DIFFERENT. {idcc_config.clip_duration_seconds} seconds. {duration_scope}",
                f"One more bit! Make it WILD. {idcc_config.clip_duration_seconds} seconds. {duration_scope}",
                f"Keep pitching! We need more bits. {idcc_config.clip_duration_seconds} seconds. {duration_scope}"
            ]
            prompt = round_prompts[min(pitch_round - 1, len(round_prompts) - 1)]

            for participant in bot_writers:
                if len(pitched_bits) >= target_pitches:
                    break

                agent = participant["agent_obj"]
                try:
                    response = await self._get_agent_idcc_response(
                        agent=agent,
                        user_message=prompt
                    )
                    if response:
                        bit = self._parse_bit_from_response(response, agent.name)
                        if bit:
                            # Check if this parody target has been used before (silently reject)
                            parody_lower = (bit.parody_target or "").lower().strip()
                            is_duplicate = False
                            if parody_lower:
                                # Check against history and current session
                                session_pitches = [b.parody_target.lower() for b in pitched_bits if b.parody_target]
                                if parody_lower in used_pitches_lower or parody_lower in session_pitches:
                                    is_duplicate = True
                                    logger.info(f"[IDCC:{self.game_id}] Silently rejecting duplicate pitch '{bit.parody_target}' from {agent.name}")

                            if not is_duplicate:
                                pitched_bits.append(bit)
                                writers_room.add_pitched_bit(bit, agent.name)
                                writers_room_log.append(f"{agent.name} pitched: {bit.format} - {bit.premise[:100]}")

                                # Display the pitch
                                await self.discord_client.send_message(
                                    content=response[:1500],
                                    agent_name=agent.name,
                                    model_name=agent.model
                                )
                                await asyncio.sleep(2)
                            # If duplicate, silently skip - don't display, loop will try again
                except Exception as e:
                    logger.error(f"[IDCC:{self.game_id}] Pitch error for {agent.name}: {e}")

        # Final check - MUST have minimum required bits
        if len(pitched_bits) < min_required:
            error_msg = f"Failed to get enough bits after {pitch_round} rounds: got {len(pitched_bits)}, need {min_required}"
            logger.error(f"[IDCC:{self.game_id}] {error_msg}")
            await self._send_gamemaster_message(f"❌ **GAME FAILED:** {error_msg}")
            self.state.phase = "failed"
            return

        # Human pitches via card selection system
        if human_participants:
            # Initialize card system
            human_card_system = HumanParodyCardSystem(self.game_id)

            # Deal cards to all humans
            for human in human_participants:
                human_card_system.deal_cards(human["name"])

            # Show cards and collect selections - 3 rounds: parody, twist, character
            for round_name, format_func, round_label in [
                ("parody", human_card_system.format_parody_cards, "🎯 PARODY TARGET"),
                ("twist", human_card_system.format_twist_cards, "🔀 TWIST"),
                ("character", human_card_system.format_character_cards, "👤 CHARACTER"),
            ]:
                # Show cards to each human (in channel - DMs would be better but more complex)
                await self._send_gamemaster_message(f"\n### {round_label} - Pick your card (type 1-4)")
                for human in human_participants:
                    cards_display = format_func(human["name"])
                    await self._send_gamemaster_message(f"**{human['name']}'s cards:**\n{cards_display}")

                # Collect selections
                selections = await self._wait_for_human_spitball_inputs(round_name, human_participants, timeout_seconds=45)
                for name, selection_text in selections.items():
                    result = human_card_system.parse_selection(name, selection_text, round_name)
                    if result is not None:
                        await self._send_gamemaster_message(f"✓ **{name}** picked #{result + 1}")

            # Build bits from selections
            for human in human_participants:
                bit = human_card_system.build_bit_from_selections(human["name"])
                if bit:
                    pitched_bits.append(bit)
                    writers_room.add_pitched_bit(bit, human["name"])
                    summary = human_card_system.get_selection_summary(human["name"])
                    writers_room_log.append(f"{human['name']} pitched: {bit.parody_target} + {bit.twist}")
                    await self._send_gamemaster_message(f"\n{summary}")
                else:
                    # Incomplete selections - log it
                    logger.warning(f"[IDCC:{self.game_id}] {human['name']} didn't complete card selections")

        if len(writers_room.pitched_bits) < 1:
            logger.error(f"[IDCC:{self.game_id}] No valid bits pitched")
            for participant in bot_writers:
                game_context_manager.exit_game_mode(participant["agent_obj"])
            return

        # =====================================================================
        # ROUND 2: VOTE FOR LINEUP
        # =====================================================================

        # Format bits for voting
        bits_display = writers_room.format_bits_for_voting()

        await self._send_gamemaster_message(
            "\n---\n"
            f"## Round 2: VOTE FOR THE LINEUP\n\n"
            f"Pick your **TOP {self.num_clips}** favorites (can't vote for your own):\n\n"
            f"{bits_display}\n\n"
            f"**Type the numbers** (e.g., '1 3 5' or '1, 3, 5')"
        )

        # Update context for voting
        for participant in bot_writers:
            agent = participant["agent_obj"]
            game_context_manager.update_idcc_context(
                agent_name=agent.name,
                phase="idcc_vote_lineup"
            )
            game_context_manager.update_turn_context(
                agent_name=agent.name,
                turn_context=f"\n{bits_display}"
            )

        agent_votes = {}  # agent_name -> list of bit indices

        # Get bot votes
        for participant in bot_writers:
            agent = participant["agent_obj"]
            try:
                # Find this agent's own bit index (can't vote for self)
                own_bit_index = -1
                for i, bit in enumerate(writers_room.pitched_bits):
                    if bit.pitched_by == agent.name:
                        own_bit_index = i
                        break

                response = await self._get_agent_idcc_response(
                    agent=agent,
                    user_message=f"Vote for your top {self.num_clips} favorite bits (NOT your own). Format: MY VOTES: 1, 3, 5"
                )
                if response:
                    votes = self._parse_lineup_votes(response, agent.name, len(writers_room.pitched_bits), own_bit_index)
                    if votes:
                        agent_votes[agent.name] = votes
                    writers_room_log.append(f"{agent.name} voted: {votes}")
                    await self.discord_client.send_message(
                        content=response[:500],
                        agent_name=agent.name,
                        model_name=agent.model
                    )
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"[IDCC:{self.game_id}] Vote error for {agent.name}: {e}")

        # Wait for human votes
        if human_participants:
            human_votes_raw = await self._wait_for_human_spitball_inputs("vote_lineup", human_participants)
            for voter_name, vote_text in human_votes_raw.items():
                # Find their own bit index
                own_bit_index = -1
                for i, bit in enumerate(writers_room.pitched_bits):
                    if bit.pitched_by == voter_name:
                        own_bit_index = i
                        break

                votes = writers_room.parse_vote(voter_name, vote_text)
                # Remove self-vote if present
                votes = [v for v in votes if v != own_bit_index]
                if votes:
                    agent_votes[voter_name] = votes
                    writers_room_log.append(f"{voter_name} voted: {votes}")
                    await self._send_gamemaster_message(f"**{voter_name} voted for:** {[v+1 for v in votes]}")

        # =====================================================================
        # TALLY VOTES
        # =====================================================================

        winning_indices = writers_room.tally_votes(agent_votes)

        # If not enough votes, take first N bits - but MUST have enough pitched bits
        if len(winning_indices) < self.num_clips:
            if len(writers_room.pitched_bits) < self.num_clips:
                error_msg = f"Not enough bits pitched: got {len(writers_room.pitched_bits)}, need {self.num_clips}"
                logger.error(f"[IDCC:{self.game_id}] {error_msg}")
                await self._send_gamemaster_message(f"❌ **GAME FAILED:** {error_msg}")
                self.state.phase = "failed"
                return
            logger.warning(f"[IDCC:{self.game_id}] Not enough voted bits, using first {self.num_clips} pitched bits")
            winning_indices = list(range(self.num_clips))

        # Get winning bits for punch-up
        winning_bits = [writers_room.pitched_bits[i] for i in winning_indices]

        # =====================================================================
        # ROUND 3: PUNCH-UP (Optional improvements or "good as is")
        # =====================================================================

        await self._send_gamemaster_message(
            "\n---\n"
            "## Round 3: PUNCH-UP\n\n"
            "Each winning bit gets a chance for improvements - or you can vote **GOOD AS IS** if it works.\n"
            "*You can't punch-up your own bit.*"
        )

        punched_up_bits = []

        for bit_idx, bit in enumerate(winning_bits):
            bit_title = f"{bit.parody_target or bit.format}: {bit.twist[:50] if bit.twist else bit.comedic_hook[:50]}..."

            # Display the bit
            bit_display = writers_room.format_bit_for_punch_up(bit)
            await self._send_gamemaster_message(
                f"\n### Bit {bit_idx + 1}: {bit_title}\n\n{bit_display}\n\n"
                "*Vote GOOD AS IS or suggest a PUNCH-UP:*"
            )

            # Update context for punch-up
            for participant in bot_writers:
                agent = participant["agent_obj"]
                game_context_manager.update_idcc_context(
                    agent_name=agent.name,
                    phase="idcc_punch_up"
                )
                game_context_manager.update_turn_context(
                    agent_name=agent.name,
                    turn_context=f"\nBit to review:\n{bit_display}"
                )

            # Collect punch-up responses
            punch_up_responses = []  # List of parsed punch-up dicts

            for participant in bot_writers:
                agent = participant["agent_obj"]
                try:
                    response = await self._get_agent_idcc_response(
                        agent=agent,
                        user_message=f"Review this bit. Reply GOOD AS IS or suggest a PUNCH-UP. You can't punch-up your own bit."
                    )
                    if response:
                        parsed = writers_room.parse_punch_up_response(response, bit.pitched_by, agent.name)
                        punch_up_responses.append(parsed)
                        writers_room_log.append(f"{agent.name} on bit {bit_idx+1}: {parsed['verdict']}")

                        # Show abbreviated response
                        display_response = response[:300] + "..." if len(response) > 300 else response
                        await self.discord_client.send_message(
                            content=display_response,
                            agent_name=agent.name,
                            model_name=agent.model
                        )
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"[IDCC:{self.game_id}] Punch-up error for {agent.name}: {e}")

            # Collect suggestions only (not GOOD_AS_IS)
            suggested_punch_ups = [p for p in punch_up_responses if p['verdict'] == 'PUNCH_UP' and p['suggestion']]

            if not suggested_punch_ups:
                # All voted GOOD AS IS - keep bit as is
                await self._send_gamemaster_message(f"✅ **Bit {bit_idx + 1}:** Room voted GOOD AS IS")
                punched_up_bits.append(bit)
                writers_room_log.append(f"Bit {bit_idx + 1}: GOOD AS IS (consensus)")
            else:
                # Have punch-ups to vote on
                punch_ups_display = writers_room.format_punch_ups_for_voting(suggested_punch_ups)
                await self._send_gamemaster_message(
                    f"\n**Suggested punch-ups for Bit {bit_idx + 1}:**\n{punch_ups_display}\n"
                    f"*Vote which to APPLY (numbers) or NONE to keep original:*"
                )

                # Update context for punch-up vote
                for participant in bot_writers:
                    agent = participant["agent_obj"]
                    game_context_manager.update_idcc_context(
                        agent_name=agent.name,
                        phase="idcc_punch_up_vote"
                    )
                    game_context_manager.update_turn_context(
                        agent_name=agent.name,
                        turn_context=f"\nPunch-ups to vote on:\n{punch_ups_display}"
                    )

                # Collect votes
                all_punch_up_votes = []  # List of lists of indices

                for participant in bot_writers:
                    agent = participant["agent_obj"]
                    try:
                        response = await self._get_agent_idcc_response(
                            agent=agent,
                            user_message=f"Vote which punch-ups to APPLY (type numbers) or NONE to keep original."
                        )
                        if response:
                            votes = writers_room.parse_punch_up_votes(response, len(suggested_punch_ups))
                            all_punch_up_votes.append(votes)
                            writers_room_log.append(f"{agent.name} punch-up vote: {votes if votes else 'NONE'}")

                            await self.discord_client.send_message(
                                content=response[:200],
                                agent_name=agent.name,
                                model_name=agent.model
                            )
                            await asyncio.sleep(0.5)
                    except Exception as e:
                        logger.error(f"[IDCC:{self.game_id}] Punch-up vote error for {agent.name}: {e}")

                # Tally punch-up votes (plurality wins - at least half of voters)
                num_voters = len(all_punch_up_votes)
                threshold = num_voters / 2  # At least half

                accepted_indices = []
                for pu_idx in range(len(suggested_punch_ups)):
                    votes_for = sum(1 for votes in all_punch_up_votes if pu_idx in votes)
                    if votes_for >= threshold:  # >= so ties pass
                        accepted_indices.append(pu_idx)

                if accepted_indices:
                    accepted_punch_ups = [suggested_punch_ups[i] for i in accepted_indices]
                    bit = writers_room.apply_punch_ups(bit, accepted_punch_ups)
                    punch_up_list = ", ".join([f"#{i+1}" for i in accepted_indices])
                    await self._send_gamemaster_message(f"✏️ **Bit {bit_idx + 1}:** Applied punch-ups {punch_up_list}")
                    writers_room_log.append(f"Bit {bit_idx + 1}: Applied punch-ups {accepted_indices}")
                else:
                    await self._send_gamemaster_message(f"✅ **Bit {bit_idx + 1}:** No consensus - keeping original")
                    writers_room_log.append(f"Bit {bit_idx + 1}: No punch-up consensus")

                punched_up_bits.append(bit)

            await asyncio.sleep(1)

        # Exit game mode for all bot writers
        for participant in bot_writers:
            game_context_manager.exit_game_mode(participant["agent_obj"])

        # =====================================================================
        # CURATE FINAL LINEUP
        # =====================================================================

        # Curate lineup for variety (using the punched-up bits)
        lineup_bits = writers_room.curate_lineup_from_bits(punched_up_bits)

        # Record used pitches to history (for future games to avoid)
        for bit in lineup_bits:
            if bit.parody_target:
                config_manager.add_idcc_pitch(bit.parody_target)
        logger.info(f"[IDCC:{self.game_id}] Recorded {len(lineup_bits)} pitches to history")

        # Create channel lineup
        channel_lineup = IDCCChannelLineup(
            bits=lineup_bits,
            writers_room_log=writers_room_log,
            clip_duration_seconds=idcc_config.clip_duration_seconds
        )

        self.state.channel_lineup = channel_lineup

        # Display the lineup
        lineup_display = "# 📺 CHANNEL LINEUP LOCKED\n\n"
        for i, bit in enumerate(lineup_bits):
            # Show parody info if available, fallback to format/premise
            parody_info = bit.parody_target or bit.format
            twist_info = bit.twist or bit.comedic_hook or bit.premise
            lineup_display += f"**Bit {i+1}:** {parody_info.upper()} by {bit.pitched_by}\n"
            lineup_display += f"   *{twist_info[:80]}{'...' if len(twist_info) > 80 else ''}*"
            if bit.punched_up and bit.punch_ups_applied:
                lineup_display += " ✏️"
            lineup_display += "\n\n"
        lineup_display += f"*Now generating {len(lineup_bits)} independent bits...*"

        await self._send_gamemaster_message(lineup_display)
        logger.info(f"[IDCC:{self.game_id}] Channel lineup established with {len(lineup_bits)} bits")

    async def _get_agent_idcc_response(
        self,
        agent: 'Agent',
        user_message: str,
        images: Optional[List[Dict]] = None
    ) -> Optional[str]:
        """
        Get a response from an agent using the game context system.

        The agent's game context (set via game_context_manager) determines
        which IDCC prompt they see - this is the proper way to handle
        game-specific context.
        """
        try:
            import aiohttp

            # Get the game-specific prompt from the context manager
            game_prompt = game_context_manager.get_game_prompt_for_agent(agent.name)

            if not game_prompt:
                logger.warning(f"[IDCC:{self.game_id}] No game prompt for {agent.name}, using base system prompt")
                system_content = agent.system_prompt
            else:
                system_content = f"{agent.system_prompt}\n\n{game_prompt}"

            messages = [{"role": "system", "content": system_content}]

            # Build user message content
            if images:
                messages.append({
                    "role": "user",
                    "content": [
                        *images,
                        {"type": "text", "text": user_message}
                    ]
                })
            else:
                messages.append({"role": "user", "content": user_message})

            headers = {
                "Authorization": f"Bearer {self.agent_manager.openrouter_api_key}",
                "Content-Type": "application/json"
            }

            # Get max_tokens from game settings
            from .game_prompts import get_game_settings
            game_state = game_context_manager.get_game_state(agent.name)
            phase = game_state.idcc_phase if game_state else None
            settings = get_game_settings(phase) if phase else {}
            max_tokens = settings.get("max_tokens", 300)

            payload = {
                "model": agent.model,
                "messages": messages,
                "max_tokens": max_tokens
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"[IDCC:{self.game_id}] API error {response.status}: {error_text[:200]}")
                        return None

                    result = await response.json()
                    message = result.get("choices", [{}])[0].get("message", {})
                    content = message.get("content") or ""

                    # Handle models that output tool_calls even without tools defined
                    if not content:
                        tool_calls = message.get("tool_calls", [])
                        if tool_calls:
                            try:
                                args_str = tool_calls[0].get("function", {}).get("arguments", "{}")
                                import json as json_module
                                args = json_module.loads(args_str) if isinstance(args_str, str) else args_str
                                content = args.get("prompt", "")
                                if content:
                                    logger.info(f"[IDCC:{self.game_id}] Extracted prompt from tool_call")
                            except Exception:
                                pass

                    # Handle GLM XML format in content
                    if not content and "<arg_key>" in str(message):
                        import re
                        raw = str(message)
                        match = re.search(r'<arg_key>prompt</arg_key>\s*<arg_value>(.*?)</arg_value>', raw, re.DOTALL)
                        if match:
                            content = match.group(1).strip()
                            logger.info(f"[IDCC:{self.game_id}] Extracted prompt from GLM XML format")

                    return content.strip() if content else None

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Agent IDCC response error: {e}", exc_info=True)
            return None

    # ========================================================================
    # PHASE 4: CLIP GENERATION
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
            creator_user_id = participant.get("user_id")  # Discord user ID for @ mentions

            # Create @ mention string if user has an ID (for humans only)
            mention_str = f"<@{creator_user_id}>" if creator_user_id else creator_name

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

            # Announce whose turn it is (with @ mention for humans)
            # Get bit info from channel lineup if available
            current_bit = None
            if self.state.channel_lineup:
                current_bit = self.state.channel_lineup.get_bit(clip_num)

            if current_bit:
                # Robot Chicken style: announce independent bit
                bit_desc = f"**{current_bit.format.upper()}** - {current_bit.premise[:80]}{'...' if len(current_bit.premise) > 80 else ''}"
                if creator_type == "human":
                    await self._send_gamemaster_message(
                        f"# Bit {clip_num}/{self.num_clips}: {mention_str}'s turn!\n\n"
                        f"📺 **CHANNEL FLIP** - New independent bit!\n"
                        f"{bit_desc}\n\n"
                        f"**STYLE: Adult Swim cartoon** - 2D animation, bold outlines, flat colors\n\n"
                        f"Type `[SCENE]` followed by your scene description.\n"
                        f"*This is a SELF-CONTAINED bit - land the joke in one clip!*\n\n"
                        f"*You have 2 minutes to submit...*"
                    )
                else:
                    await self._send_gamemaster_message(
                        f"**Bit {clip_num}/{self.num_clips}:** {creator_name} is generating...\n"
                        f"📺 {bit_desc}"
                    )
            else:
                # No bit available - this should never happen after validation fixes
                error_msg = f"No bit available for clip {clip_num}"
                logger.error(f"[IDCC:{self.game_id}] {error_msg}")
                await self._send_gamemaster_message(f"❌ **ERROR:** {error_msg}")
                clip.status = "failed"
                clip.error_message = error_msg
                continue

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
            # Retry up to MAX_SCENE_RETRIES times, then abort game to avoid API cap exhaustion
            MAX_SCENE_RETRIES = 10
            scene_attempt = 0
            scene_success = False
            total_scene_time = 0.0

            while not scene_success:
                scene_attempt += 1

                # Check if we've hit the retry cap
                if scene_attempt > MAX_SCENE_RETRIES:
                    logger.error(f"[IDCC:{self.game_id}] Scene {clip_num} failed after {MAX_SCENE_RETRIES} attempts - aborting game to preserve API quota")
                    await self._send_gamemaster_message(
                        f"⚠️ **GAME ABORTED** ⚠️\n\n"
                        f"Scene {clip_num} failed after {MAX_SCENE_RETRIES} attempts. "
                        f"This usually means the video API is rate-limited or experiencing issues.\n\n"
                        f"Ending game to avoid further API usage. Please try again later."
                    )
                    # Mark game as failed and exit
                    self.state.phase = "aborted"
                    clip.error_message = f"Aborted after {MAX_SCENE_RETRIES} failed attempts"
                    return  # Exit the entire generation phase

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
                        reference_frame_path=previous_frame_path if clip_num > 1 else None,
                        base_variant=scene_attempt - 1  # Increment variant to try different declassifications
                    )

                    # Stop progress indicator
                    progress_task.cancel()
                    try:
                        await progress_task
                    except asyncio.CancelledError:
                        pass

                    attempt_time = time.time() - start_time
                    total_scene_time += attempt_time

                    # Check if video_result is a success (has "path") or failure (has "error")
                    if video_result and "path" in video_result:
                        clip.video_path = video_result["path"]
                        clip.video_url = video_result.get("url")
                        clip.success = True
                        scene_success = True

                        # Extract last frame for next iteration
                        previous_frame_path = extract_last_frame(clip.video_path)
                        clip.last_frame_path = previous_frame_path
                        previous_prompt = prompt

                        # Post success
                        await self._send_gamemaster_message(
                            f"**Scene {clip_num} complete!** ({total_scene_time:.0f}s) by {creator_name}"
                        )

                    elif video_result and "error" in video_result:
                        # Got error info from _generate_video_clip - save failed prompt
                        error_msg = video_result.get("error", "Unknown error")
                        attempts = video_result.get("attempts", 0)
                        declassified = video_result.get("declassified_prompts", [])

                        # Save failed prompt with error details
                        save_failed_prompt(
                            game_id=self.game_id,
                            clip_number=clip_num,
                            prompt=prompt,
                            error_message=error_msg,
                            attempts=attempts,
                            declassified_prompts=declassified
                        )
                        clip.error_message = error_msg

                        # Log and retry
                        logger.warning(f"[IDCC:{self.game_id}] Scene {clip_num} attempt {scene_attempt}/{MAX_SCENE_RETRIES} failed: {error_msg}")
                        await self._send_gamemaster_message(
                            f"**Scene {clip_num} attempt {scene_attempt} failed.** Retrying with modified prompt..."
                        )
                        # Brief pause before retry
                        await asyncio.sleep(3)
                    else:
                        # Unexpected None result
                        logger.warning(f"[IDCC:{self.game_id}] Scene {clip_num} attempt {scene_attempt}/{MAX_SCENE_RETRIES} returned None")
                        await self._send_gamemaster_message(
                            f"**Scene {clip_num} attempt {scene_attempt} failed.** Retrying with modified prompt..."
                        )
                        await asyncio.sleep(3)

                except Exception as e:
                    logger.error(f"[IDCC:{self.game_id}] Video generation error on attempt {scene_attempt}: {e}", exc_info=True)
                    await self._send_gamemaster_message(
                        f"**Scene {clip_num} error on attempt {scene_attempt}:** {str(e)[:100]}. Retrying..."
                    )
                    await asyncio.sleep(5)

            clip.generation_time_seconds = total_scene_time

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
        Have an agent generate a scene prompt using the Robot Chicken bit system.

        Each clip is now an independent bit from the channel_lineup.
        The agent generates based on that bit's BitConcept.

        Args:
            agent: Agent to generate the prompt
            clip_number: 1-indexed clip number
            previous_prompt: Previous scene's prompt (for lastframe reference)
            previous_frame_path: Path to last frame image

        Returns:
            Generated prompt string
        """
        from agent_games.game_prompts import get_bit_scene_timing, build_mandatory_scene_ending

        try:
            # Get this scene's BitConcept from the channel lineup
            bit = None
            if self.state.channel_lineup:
                bit = self.state.channel_lineup.get_bit(clip_number)

            if bit:
                # Robot Chicken style - each scene is its own bit
                phase = "idcc_scene_bit"

                # Get clip duration from user settings
                clip_duration = self.state.channel_lineup.clip_duration_seconds
                is_final = (clip_number == self.num_clips)

                # Get next bit's character for transition (if not final)
                next_bit_character = None
                if not is_final and clip_number < len(self.state.channel_lineup.bits):
                    next_bit = self.state.channel_lineup.get_bit(clip_number + 1)
                    if next_bit:
                        next_bit_character = next_bit.character_description

                # Get timing parameters calibrated to duration
                timing = get_bit_scene_timing(clip_duration, is_final, next_bit_character)

                # Use bit's shot direction or generate one
                shot_direction = bit.shot_direction or get_shot_direction(
                    show_format=bit.format,
                    scene_number=clip_number,
                    total_scenes=self.num_clips
                )

                user_message = (
                    f"Create the video prompt for Bit {clip_number}. "
                    f"This is a SELF-CONTAINED {bit.format}. Land the joke in this one clip. "
                    f"Output ONLY the video prompt starting with 'Adult Swim cartoon style...'"
                )

                # Enter game mode
                game_context_manager.enter_game_mode(
                    agent=agent,
                    game_name=phase
                )

                # Set IDCC context with BitConcept fields for prompt formatting
                game_context_manager.update_idcc_context(
                    agent_name=agent.name,
                    phase=phase,
                    show_bible="",  # Not used for Robot Chicken style
                    previous_prompt=previous_prompt,
                    scene_number=clip_number,
                    num_clips=self.num_clips,
                    shot_direction=shot_direction,
                    # BitConcept parody fields for idcc_scene_bit prompt
                    bit_parody_target=bit.parody_target,
                    bit_twist=bit.twist,
                    bit_format=bit.format,
                    bit_character=bit.character_description,
                    bit_vocal_specs=bit.vocal_specs or "clear speaking voice with character-appropriate energy",
                    bit_sample_dialogue=bit.sample_dialogue,
                    bit_punchline=bit.punchline,
                    clip_duration=clip_duration,
                    dialogue_end_time=timing["dialogue_end_time"],
                    dialogue_word_limit=timing["dialogue_word_limit"],
                    duration_scope=timing["duration_scope"],
                    scene_ending_instruction=timing["scene_ending_instruction"],
                    timing_details=timing["timing_details"]
                )
            else:
                # No bit for this clip - this should never happen if pitch validation worked
                error_msg = f"No BitConcept for clip {clip_number} - channel_lineup has {len(self.state.channel_lineup.bits) if self.state.channel_lineup else 0} bits"
                logger.error(f"[IDCC:{self.game_id}] {error_msg}")
                return None

            # Prepare image if we have a previous frame (kept for visual consistency)
            images = None
            if clip_number > 1 and previous_frame_path and previous_frame_path.exists():
                frame_b64 = image_to_base64(previous_frame_path)
                if frame_b64:
                    images = [{
                        "type": "image_url",
                        "image_url": {"url": frame_b64}
                    }]

            # Get response using the game context system
            response = await self._get_agent_idcc_response(
                agent=agent,
                user_message=user_message,
                images=images
            )

            # Exit game mode
            game_context_manager.exit_game_mode(agent)

            if not response:
                return None

            # Clean up the response (remove any markdown, explanations, etc.)
            prompt = response.strip()

            # Remove common prefixes/suffixes
            for prefix in ["Here is the video prompt:", "Video prompt:", "Prompt:"]:
                if prompt.lower().startswith(prefix.lower()):
                    prompt = prompt[len(prefix):].strip()

            # PROGRAMMATICALLY APPEND the mandatory ending sequence
            # This ensures TV static + next scene preview is ALWAYS included
            # regardless of what the agent generated
            if bit:
                clip_duration = self.state.channel_lineup.clip_duration_seconds
                is_final = (clip_number == self.num_clips)

                # Get next bit for transition preview
                next_bit = None
                if not is_final and clip_number < len(self.state.channel_lineup.bits):
                    next_bit = self.state.channel_lineup.get_bit(clip_number + 1)

                # Append mandatory ending
                mandatory_ending = build_mandatory_scene_ending(clip_duration, is_final, next_bit)
                prompt = prompt + mandatory_ending
                logger.info(f"[IDCC:{self.game_id}] Appended mandatory ending for clip {clip_number} (final={is_final})")

            logger.info(f"[IDCC:{self.game_id}] {agent.name} generated: {prompt[:100]}...")
            return prompt

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Agent prompt generation error: {e}", exc_info=True)
            # Make sure to exit game mode on error
            try:
                game_context_manager.exit_game_mode(agent)
            except:
                pass
            return None

    async def _generate_video_clip(
        self,
        prompt: str,
        creator_name: str,
        reference_frame_path: Optional[Path] = None,
        base_variant: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a video clip using Sora 2.

        Args:
            prompt: The video generation prompt
            creator_name: Who is creating this clip
            reference_frame_path: Optional starting frame for continuity
            base_variant: Base variant number for declassification (allows incrementing across outer retries)

        Returns:
            Dict with 'path' and optionally 'url', or None on failure
            On failure, returns dict with 'error', 'attempts', 'declassified_prompts' for debugging
        """
        max_retries = idcc_config.max_retries_per_clip
        declassified_prompts = []  # Track all declassified prompts tried
        last_error = "Unknown error"

        for attempt in range(max_retries):
            try:
                # Calculate effective variant: base_variant * max_retries + attempt
                # This ensures each outer retry round uses different declassification variants
                effective_variant = base_variant * max_retries + attempt

                # First attempt in first round: use original prompt
                # Otherwise: declassify the prompt with the effective variant
                use_prompt = prompt
                reference_image_b64 = None

                if effective_variant > 0:
                    logger.info(f"[IDCC:{self.game_id}] Attempting declassification (variant {effective_variant})...")

                    # Try to get declassified version
                    declassified = await self.agent_manager.declassify_image_prompt(
                        prompt, variant=effective_variant
                    )

                    if declassified:
                        use_prompt = declassified
                        declassified_prompts.append(f"[Variant {effective_variant}] {declassified}")
                        logger.info(f"[IDCC:{self.game_id}] Using declassified prompt: {use_prompt[:100]}...")
                    else:
                        # Declassifier failed - apply manual fallback modifications
                        logger.warning(f"[IDCC:{self.game_id}] Declassifier failed! Applying manual fallback...")

                        # Manual prompt sanitization fallback
                        # Add style prefix and simplify potentially problematic content
                        style_prefixes = [
                            "A cartoon scene showing",
                            "An animated TV clip of",
                            "A whimsical animated scene depicting",
                            "A stylized cartoon animation featuring",
                            "A colorful animated sequence of",
                        ]
                        prefix = style_prefixes[effective_variant % len(style_prefixes)]

                        # Clean potentially problematic words
                        sanitized = prompt
                        replacements = [
                            ("blood", "red liquid"), ("gore", "mess"), ("violent", "dramatic"),
                            ("kill", "defeat"), ("dead", "unconscious"), ("death", "defeat"),
                            ("weapon", "prop"), ("gun", "device"), ("knife", "utensil"),
                            ("drug", "substance"), ("cocaine", "powder"), ("weed", "herb"),
                            ("sexy", "stylish"), ("naked", "unclothed"), ("nude", "bare"),
                        ]
                        for old, new in replacements:
                            sanitized = sanitized.replace(old, new).replace(old.capitalize(), new.capitalize())

                        use_prompt = f"{prefix} {sanitized}"
                        declassified_prompts.append(f"[Manual Fallback {effective_variant}] {use_prompt}")
                        logger.info(f"[IDCC:{self.game_id}] Manual fallback prompt: {use_prompt[:100]}...")

                # Prepare reference frame if provided
                if reference_frame_path and reference_frame_path.exists():
                    # Resize to match Sora's expected dimensions
                    resized_frame = await resize_image_for_sora(reference_frame_path)
                    if resized_frame:
                        reference_image_b64 = image_to_base64(resized_frame)

                # Generate video with image-to-video if we have a reference
                # Skip cooldown for IDCC - we need to generate multiple clips in sequence
                video_result = await self.agent_manager.generate_video_with_reference(
                    prompt=use_prompt,
                    author=creator_name,
                    duration=idcc_config.clip_duration_seconds,
                    input_reference=reference_image_b64,
                    skip_cooldown=True
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
                                last_error = "Failed to download video from URL"
                                continue

                        return {"path": video_path, "url": video_result}
                else:
                    # Get detailed error from agent_manager if available
                    api_error = getattr(self.agent_manager, 'last_video_error', '')
                    if api_error:
                        last_error = f"API: {api_error}"
                    else:
                        last_error = f"Video generation returned None (attempt {attempt + 1})"

            except Exception as e:
                last_error = f"Exception on attempt {attempt + 1}: {str(e)}"
                logger.error(f"[IDCC:{self.game_id}] Video generation attempt {attempt + 1} failed: {e}")

            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(5)

        # All retries failed - return error info for saving
        return {
            "error": last_error,
            "attempts": max_retries,
            "declassified_prompts": declassified_prompts
        }

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

        # Output path - save to Media/Videos/ for organized storage
        ensure_media_dirs()
        output_path = MEDIA_VIDEOS_DIR / f"idcc_final_{self.game_id}.mp4"

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

            # Save all clip prompts as the combined prompt for this video
            all_prompts = []
            for i, clip in enumerate(successful_clips, 1):
                all_prompts.append(f"=== CLIP {i} ===\n{clip.prompt}\n")
            combined_prompt = "\n".join(all_prompts)
            save_media_prompt(result, combined_prompt, media_type="video")
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

        # Find human contributors and their IDs for @ mentions
        human_contributor_mentions = []
        human_contributor_names = set()
        for clip in self.state.clips:
            if clip.success and clip.creator_type == "human":
                human_contributor_names.add(clip.creator_name)

        # Get user IDs from registered_human_ids
        for name in human_contributor_names:
            user_id = self.state.registered_human_ids.get(name)
            if user_id:
                human_contributor_mentions.append(f"<@{user_id}>")

        # Build credits
        credits_lines = ["**CREDITS:**"]
        for clip in self.state.clips:
            status = "completed" if clip.success else "failed"
            credits_lines.append(f"• Scene {clip.clip_number}: {clip.creator_name} ({status})")

        total_time = self.state.total_generation_time
        credits_lines.append(f"\n*Total generation time: {total_time:.0f}s*")

        # Add human contributor mentions if any
        if human_contributor_mentions:
            credits_lines.append(f"\n🎬 **Human Directors:** {' '.join(human_contributor_mentions)}")

        credits = "\n".join(credits_lines)

        # Send the video
        main_channel_success = False
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
                main_channel_success = True
                # Still try media channel (boosted servers have higher limits)
                if self.discord_client and self.discord_client.media_channel_id:
                    try:
                        result = await self.discord_client.post_to_media_channel(
                            media_type="video",
                            agent_name="Interdimensional Cable",
                            model_name=f"IDCC #{self.game_id}",
                            prompt=credits,
                            file_data=str(self.state.final_video_path),
                            filename="interdimensional_cable.mp4"
                        )
                        if result:
                            config_manager.add_idcc_posted_video(self.state.final_video_path.name)
                    except Exception as media_err:
                        logger.error(f"[IDCC:{self.game_id}] Media channel post failed: {media_err}")
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
                main_channel_success = True

                # Cross-post to media channel immediately after main post (same pattern as images)
                if self.discord_client and self.discord_client.media_channel_id:
                    try:
                        result = await self.discord_client.post_to_media_channel(
                            media_type="video",
                            agent_name="Interdimensional Cable",
                            model_name=f"IDCC #{self.game_id}",
                            prompt=credits,
                            file_data=str(self.state.final_video_path),
                            filename="interdimensional_cable.mp4"
                        )
                        if result:
                            config_manager.add_idcc_posted_video(self.state.final_video_path.name)
                    except Exception as media_err:
                        logger.error(f"[IDCC:{self.game_id}] Media channel post failed: {media_err}")

        except Exception as e:
            logger.error(f"[IDCC:{self.game_id}] Failed to post video: {e}", exc_info=True)
            await self._send_gamemaster_message(
                f"**ERROR:** Could not upload video to Discord: {str(e)[:200]}\n"
                f"Video saved to: `{self.state.final_video_path}`\n\n"
                f"{credits}"
            )
            # Still try media channel even if main failed
            if self.discord_client and self.discord_client.media_channel_id:
                try:
                    result = await self.discord_client.post_to_media_channel(
                        media_type="video",
                        agent_name="Interdimensional Cable",
                        model_name=f"IDCC #{self.game_id}",
                        prompt=credits,
                        file_data=str(self.state.final_video_path),
                        filename="interdimensional_cable.mp4"
                    )
                    if result:
                        config_manager.add_idcc_posted_video(self.state.final_video_path.name)
                except Exception as media_err:
                    logger.error(f"[IDCC:{self.game_id}] Media channel post failed: {media_err}")

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

    async def _crosspost_to_media_channel(self):
        """DEPRECATED - media channel posting is now done inline in _post_final_video."""
        pass


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

    def can_start_game(self) -> tuple[bool, str]:
        """Check if a new IDCC game can be started."""
        global _last_idcc_end_time

        if self.active_game:
            return False, "An IDCC game is already in progress!"

        # Check cooldown
        config = _idcc_config_manager.get_config()
        elapsed = time.time() - _last_idcc_end_time
        if _last_idcc_end_time > 0 and elapsed < config.game_cooldown_seconds:
            remaining = int(config.game_cooldown_seconds - elapsed)
            mins = remaining // 60
            secs = remaining % 60
            return False, f"IDCC on cooldown. Try again in {mins}m {secs}s"

        return True, "Ready to channel surf!"

    async def start_game(
        self,
        agent_manager,
        discord_client,
        ctx: commands.Context,
        num_clips: int = 5,
        game_orchestrator=None
    ) -> bool:
        """
        Start a new Interdimensional Cable game.

        Args:
            agent_manager: AgentManager instance
            discord_client: DiscordBotClient instance
            ctx: Discord context
            num_clips: Number of clips to generate
            game_orchestrator: GameOrchestrator for resetting idle timer

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
                num_clips=num_clips,
                game_orchestrator=game_orchestrator
            )

        try:
            await self.active_game.start(ctx)
            return True
        finally:
            global _last_idcc_end_time
            async with self._lock:
                self.active_game = None
                _last_idcc_end_time = time.time()
                logger.info(f"[IDCC] Game ended, cooldown started")

    async def handle_join(self, user_name: str, user_id: Optional[str] = None) -> bool:
        """
        Handle a !join-idcc command.

        Args:
            user_name: Discord username
            user_id: Discord user ID for @ mentions

        Returns:
            True if successfully registered
        """
        if self.active_game:
            return await self.active_game.handle_join_command(user_name, user_id)
        return False


# Global manager instance
idcc_manager = IDCCGameManager()
