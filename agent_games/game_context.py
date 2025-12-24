"""
Game Context Manager

Handles agent entry/exit from game mode:
- Saves original agent settings
- Applies game-specific prompts and settings
- Restores settings when game ends
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

from .game_prompts import get_game_prompt, get_game_settings, get_post_game_prompt

logger = logging.getLogger(__name__)


@dataclass
class AgentGameState:
    """Stores agent state during game mode."""
    agent_name: str
    game_name: str
    opponent_name: Optional[str]

    # Original settings (to restore)
    original_system_prompt: str
    original_response_frequency: int
    original_response_likelihood: int
    original_max_tokens: int
    original_vector_store: Optional[Any]  # Save vector store to restore later

    # Game-specific additions
    game_prompt: str
    in_game: bool = True

    # Dynamic game state (updated during gameplay)
    legal_moves: Optional[list] = None  # For chess: list of UCI move strings
    turn_context: Optional[str] = None  # Per-turn context (strategy hints, etc.)

    # IDCC-specific: Show Bible and phase context
    idcc_show_bible: Optional[str] = None  # Formatted Show Bible string
    idcc_phase: Optional[str] = None  # Current phase: spitball_round1, spitball_round2, scene_opening, scene_middle, scene_final
    idcc_previous_prompt: Optional[str] = None  # Previous scene's prompt for continuity
    idcc_scene_number: Optional[int] = None  # Current scene number
    idcc_num_clips: Optional[int] = None  # Total number of clips
    idcc_shot_direction: Optional[str] = None  # Format-appropriate shot direction for this scene

    # IDCC Robot Chicken style: BitConcept fields for per-scene bits
    idcc_bit_parody_target: Optional[str] = None  # What we're parodying
    idcc_bit_twist: Optional[str] = None  # The comedic twist
    idcc_bit_format: Optional[str] = None
    idcc_bit_character: Optional[str] = None
    idcc_bit_vocal_specs: Optional[str] = None
    idcc_bit_sample_dialogue: Optional[str] = None  # Key dialogue lines
    idcc_bit_punchline: Optional[str] = None
    idcc_clip_duration: Optional[int] = None
    idcc_dialogue_end_time: Optional[str] = None
    idcc_dialogue_word_limit: Optional[str] = None
    idcc_duration_scope: Optional[str] = None
    idcc_scene_ending_instruction: Optional[str] = None
    idcc_timing_details: Optional[str] = None
    idcc_used_pitches_section: Optional[str] = None  # Previously used parody targets to avoid

    # Celebrity Roast specific fields
    roast_celebrity_name: Optional[str] = None  # Name of the celebrity being roasted
    roast_celebrity_associations: Optional[str] = None  # Comma-separated associations for joke writing
    roast_phase: Optional[str] = None  # Current phase: agent_roasts, celebrity_response, dismissal
    roast_round_number: Optional[int] = None  # Current round of roasting


class GameContextManager:
    """Manages agent context when entering/exiting games."""

    def __init__(self):
        """Initialize the game context manager."""
        self.active_games: Dict[str, AgentGameState] = {}  # agent_name -> state

    def enter_game_mode(
        self,
        agent,
        game_name: str,
        opponent_name: Optional[str] = None,
        **game_params
    ) -> AgentGameState:
        """
        Put agent into game mode: save settings, apply game prompt.

        Args:
            agent: Agent instance from agent_manager
            game_name: Name of game (tictactoe, chess, etc.)
            opponent_name: Name of opponent (if multiplayer)
            **game_params: Additional game-specific parameters

        Returns:
            AgentGameState with saved settings
        """
        agent_name = agent.name

        # Check if already in game
        if agent_name in self.active_games:
            logger.warning(f"[GameContext] {agent_name} already in game mode, forcing exit first")
            self.exit_game_mode(agent)

        # Save original settings including vector store
        game_state = AgentGameState(
            agent_name=agent_name,
            game_name=game_name,
            opponent_name=opponent_name,
            original_system_prompt=agent.system_prompt,
            original_response_frequency=agent.response_frequency,
            original_response_likelihood=agent.response_likelihood,
            original_max_tokens=agent.max_tokens,
            original_vector_store=agent.vector_store,  # Save to restore later
            game_prompt=get_game_prompt(game_name, agent_name, opponent_name, **game_params),
            in_game=True
        )

        # Store state
        self.active_games[agent_name] = game_state

        # Store original settings on agent for safe saving during game mode
        agent._game_mode_original_settings = {
            "response_frequency": game_state.original_response_frequency,
            "response_likelihood": game_state.original_response_likelihood,
            "max_tokens": game_state.original_max_tokens
        }

        # Apply game settings
        game_settings = get_game_settings(game_name)
        if game_settings:
            if 'response_frequency' in game_settings:
                agent.response_frequency = game_settings['response_frequency']
            if 'response_likelihood' in game_settings:
                agent.response_likelihood = game_settings['response_likelihood']
            if 'max_tokens' in game_settings:
                agent.max_tokens = game_settings['max_tokens']

        # CRITICAL: Disable vector store during game mode
        # Game messages are ephemeral and don't need long-term memory
        agent.vector_store = None
        logger.info(f"[GameContext] DISABLED vector store for {agent_name} during game (was: {game_state.original_vector_store is not None})")

        logger.info(f"[GameContext] {agent_name} entered {game_name} mode "
                   f"(freq: {agent.response_frequency}s, likelihood: {agent.response_likelihood}%, "
                   f"tokens: {agent.max_tokens})")

        return game_state

    def exit_game_mode(self, agent, inject_transition: bool = False, transition_message: str = None) -> bool:
        """
        Exit game mode: restore original settings.

        NOTE: This method only restores settings. Transition message injection
        is handled by GameContext.exit() which has game-specific messages and
        pre-game conversation context. Only use inject_transition=True for
        emergency/error exits where GameContext isn't available.

        Args:
            agent: Agent instance from agent_manager
            inject_transition: If True, inject a basic transition message (for error cases)
            transition_message: Optional custom transition message to inject

        Returns:
            True if settings were restored, False if not in game mode
        """
        agent_name = agent.name

        if agent_name not in self.active_games:
            logger.warning(f"[GameContext] {agent_name} not in game mode, nothing to restore")
            return False

        # Get saved state
        game_state = self.active_games[agent_name]
        game_name = game_state.game_name

        # Restore original settings
        agent.response_frequency = game_state.original_response_frequency
        agent.response_likelihood = game_state.original_response_likelihood
        agent.max_tokens = game_state.original_max_tokens
        agent.vector_store = game_state.original_vector_store  # Restore vector store
        agent._game_mode_original_settings = None  # Clear game mode flag

        logger.info(f"[GameContext] {agent_name} exited {game_name} mode "
                   f"(restored freq: {agent.response_frequency}s, likelihood: {agent.response_likelihood}%, "
                   f"tokens: {agent.max_tokens})")
        logger.debug(f"[GameContext] Restored vector store for {agent_name}")

        # Only inject transition message if explicitly requested (e.g., error recovery)
        if inject_transition:
            game_type = game_name.replace("_", " ").title()
            if transition_message:
                msg = transition_message
            else:
                # Basic fallback message for error cases
                msg = f"[{game_type.upper()} HAS ENDED - returning to normal conversation]"

            agent.add_message_to_history(
                author="[SYSTEM]",
                content=msg,
                message_id=None,
                replied_to_agent=None,
                user_id=None
            )
            logger.info(f"[GameContext] Injected transition message for {agent_name}")

        # Mark as exited and remove from active games
        game_state.in_game = False
        del self.active_games[agent_name]

        return True

    def get_game_state(self, agent_name: str) -> Optional[AgentGameState]:
        """
        Get current game state for an agent.

        Args:
            agent_name: Name of agent

        Returns:
            AgentGameState if in game, None otherwise
        """
        return self.active_games.get(agent_name)

    def is_in_game(self, agent_name: str) -> bool:
        """
        Check if agent is currently in a game.

        Args:
            agent_name: Name of agent

        Returns:
            True if in game mode
        """
        return agent_name in self.active_games

    def update_legal_moves(self, agent_name: str, legal_moves: list) -> None:
        """
        Update the legal moves for an agent in game mode.

        Args:
            agent_name: Name of agent
            legal_moves: List of legal move strings (e.g., UCI notation for chess)
        """
        if agent_name in self.active_games:
            self.active_games[agent_name].legal_moves = legal_moves
            logger.debug(f"[GameContext] Updated legal moves for {agent_name}: {len(legal_moves)} moves available")

    def update_turn_context(self, agent_name: str, turn_context: Optional[str]) -> None:
        """
        Update the per-turn context for an agent in game mode.

        Args:
            agent_name: Name of agent
            turn_context: Context string for this turn (strategy hints, etc.), or None to clear
        """
        if agent_name in self.active_games:
            self.active_games[agent_name].turn_context = turn_context
            logger.debug(f"[GameContext] Updated turn context for {agent_name}")

    def update_idcc_context(
        self,
        agent_name: str,
        phase: Optional[str] = None,
        show_bible: Optional[str] = None,
        previous_prompt: Optional[str] = None,
        scene_number: Optional[int] = None,
        num_clips: Optional[int] = None,
        shot_direction: Optional[str] = None,
        # Robot Chicken style BitConcept fields (parody style)
        bit_parody_target: Optional[str] = None,
        bit_twist: Optional[str] = None,
        bit_format: Optional[str] = None,
        bit_character: Optional[str] = None,
        bit_vocal_specs: Optional[str] = None,
        bit_sample_dialogue: Optional[str] = None,
        bit_punchline: Optional[str] = None,
        clip_duration: Optional[int] = None,
        dialogue_end_time: Optional[str] = None,
        dialogue_word_limit: Optional[str] = None,
        duration_scope: Optional[str] = None,
        scene_ending_instruction: Optional[str] = None,
        timing_details: Optional[str] = None
    ) -> None:
        """
        Update IDCC-specific context for an agent.

        Args:
            agent_name: Name of agent
            phase: Current IDCC phase
            show_bible: Formatted Show Bible string (legacy)
            previous_prompt: Previous scene's prompt for continuity
            scene_number: Current scene number (1-indexed)
            num_clips: Total number of clips
            shot_direction: Format-appropriate shot direction for this scene
            bit_parody_target: What we're parodying (Peloton, House Hunters, etc.)
            bit_twist: The comedic twist/angle
            bit_format: BitConcept format (infomercial, talk show, etc.)
            bit_character: BitConcept character description
            bit_vocal_specs: BitConcept vocal specifications
            bit_sample_dialogue: Key dialogue lines
            bit_punchline: BitConcept punchline
            clip_duration: Clip duration in seconds
            dialogue_end_time: When dialogue should end
            dialogue_word_limit: Max dialogue words
            duration_scope: Description of beat count
            scene_ending_instruction: How to end the scene
            timing_details: Detailed timing info
        """
        if agent_name not in self.active_games:
            logger.warning(f"[GameContext] Cannot update IDCC context for {agent_name} - not in game mode")
            return

        state = self.active_games[agent_name]

        if phase is not None:
            state.idcc_phase = phase
            # Update the game prompt to the phase-specific one
            state.game_prompt = get_game_prompt(phase, agent_name, None)
            logger.debug(f"[GameContext] Updated IDCC phase for {agent_name}: {phase}")

        if show_bible is not None:
            state.idcc_show_bible = show_bible
            logger.debug(f"[GameContext] Updated Show Bible for {agent_name}")

        if previous_prompt is not None:
            state.idcc_previous_prompt = previous_prompt

        if scene_number is not None:
            state.idcc_scene_number = scene_number

        if num_clips is not None:
            state.idcc_num_clips = num_clips

        if shot_direction is not None:
            state.idcc_shot_direction = shot_direction
            logger.debug(f"[GameContext] Updated shot direction for {agent_name}: {shot_direction[:50]}...")

        # Robot Chicken style BitConcept fields (parody style)
        if bit_parody_target is not None:
            state.idcc_bit_parody_target = bit_parody_target
        if bit_twist is not None:
            state.idcc_bit_twist = bit_twist
        if bit_format is not None:
            state.idcc_bit_format = bit_format
        if bit_character is not None:
            state.idcc_bit_character = bit_character
        if bit_vocal_specs is not None:
            state.idcc_bit_vocal_specs = bit_vocal_specs
        if bit_sample_dialogue is not None:
            state.idcc_bit_sample_dialogue = bit_sample_dialogue
        if bit_punchline is not None:
            state.idcc_bit_punchline = bit_punchline
        if clip_duration is not None:
            state.idcc_clip_duration = clip_duration
        if dialogue_end_time is not None:
            state.idcc_dialogue_end_time = dialogue_end_time
        if dialogue_word_limit is not None:
            state.idcc_dialogue_word_limit = dialogue_word_limit
        if duration_scope is not None:
            state.idcc_duration_scope = duration_scope
        if scene_ending_instruction is not None:
            state.idcc_scene_ending_instruction = scene_ending_instruction
        if timing_details is not None:
            state.idcc_timing_details = timing_details

    def update_idcc_used_pitches(self, agent_name: str, used_pitches_section: str) -> None:
        """Update the used pitches section for the pitch prompt."""
        if agent_name not in self.active_games:
            return
        self.active_games[agent_name].idcc_used_pitches_section = used_pitches_section

    def update_roast_context(
        self,
        agent_name: str,
        celebrity_name: Optional[str] = None,
        celebrity_associations: Optional[str] = None,
        phase: Optional[str] = None,
        round_number: Optional[int] = None
    ) -> None:
        """
        Update Celebrity Roast specific context for an agent.

        Args:
            agent_name: Name of agent
            celebrity_name: Name of the celebrity being roasted
            celebrity_associations: Comma-separated associations for joke writing
            phase: Current roast phase (agent_roasts, celebrity_response, dismissal)
            round_number: Current round of roasting
        """
        if agent_name not in self.active_games:
            logger.warning(f"[GameContext] Cannot update roast context for {agent_name} - not in game mode")
            return

        state = self.active_games[agent_name]

        if celebrity_name is not None:
            state.roast_celebrity_name = celebrity_name
            logger.debug(f"[GameContext] Updated roast celebrity for {agent_name}: {celebrity_name}")

        if celebrity_associations is not None:
            state.roast_celebrity_associations = celebrity_associations

        if phase is not None:
            state.roast_phase = phase
            # Update the game prompt to the phase-specific one
            state.game_prompt = get_game_prompt(f"roast_{phase}", agent_name, None)
            logger.debug(f"[GameContext] Updated roast phase for {agent_name}: {phase}")

        if round_number is not None:
            state.roast_round_number = round_number

    def get_game_prompt_for_agent(self, agent_name: str) -> str:
        """
        Get the game-specific prompt for an agent.

        Args:
            agent_name: Name of agent

        Returns:
            Game prompt string, or empty string if not in game
        """
        game_state = self.get_game_state(agent_name)
        if not game_state:
            return ""

        # Start with base game prompt
        prompt = game_state.game_prompt

        # Fill in IDCC-specific template variables if applicable
        if game_state.idcc_phase:
            replacements = {
                "{show_bible}": game_state.idcc_show_bible or "No Show Bible established yet.",
                "{previous_prompt}": game_state.idcc_previous_prompt or "N/A - this is the first scene",
                "{scene_number}": str(game_state.idcc_scene_number or 1),
                "{num_clips}": str(game_state.idcc_num_clips or 4),
                "{shot_direction}": game_state.idcc_shot_direction or "Wide establishing shot - set the scene",
                # Robot Chicken style BitConcept fields (parody style)
                "{bit_parody_target}": game_state.idcc_bit_parody_target or "generic TV show",
                "{bit_twist}": game_state.idcc_bit_twist or "absurd dark twist",
                "{bit_format}": game_state.idcc_bit_format or "comedy sketch",
                "{bit_character}": game_state.idcc_bit_character or "exaggerated cartoon character",
                "{bit_vocal_specs}": game_state.idcc_bit_vocal_specs or "clear speaking voice with character-appropriate energy",
                "{bit_sample_dialogue}": game_state.idcc_bit_sample_dialogue or "Character delivers lines with commitment to the bit",
                "{bit_punchline}": game_state.idcc_bit_punchline or "surprising payoff",
                "{clip_duration}": str(game_state.idcc_clip_duration or 12),
                "{dialogue_end_time}": game_state.idcc_dialogue_end_time or "0:09",
                "{dialogue_word_limit}": game_state.idcc_dialogue_word_limit or "20",
                "{duration_scope}": game_state.idcc_duration_scope or "THREE BEATS - setup, escalation, punchline",
                "{scene_ending_instruction}": game_state.idcc_scene_ending_instruction or "TV static/channel flip effect",
                "{timing_details}": game_state.idcc_timing_details or "",
                "{used_pitches_section}": game_state.idcc_used_pitches_section or "",
            }
            for key, value in replacements.items():
                prompt = prompt.replace(key, value)

        # Fill in roast-specific template variables if applicable
        if game_state.roast_phase:
            replacements = {
                "{celebrity_name}": game_state.roast_celebrity_name or "Unknown Celebrity",
                "{celebrity_associations}": game_state.roast_celebrity_associations or "famous person",
                "{roast_round}": str(game_state.roast_round_number or 1),
            }
            for key, value in replacements.items():
                prompt = prompt.replace(key, value)

        # Add optional turn context
        if game_state.turn_context:
            prompt = prompt + "\n" + game_state.turn_context

        return prompt

    def get_all_active_games(self) -> Dict[str, AgentGameState]:
        """
        Get all active game states.

        Returns:
            Dictionary mapping agent_name -> AgentGameState
        """
        return self.active_games.copy()

    def force_exit_all(self, agent_manager) -> int:
        """
        Force all agents out of game mode (emergency cleanup).

        Args:
            agent_manager: AgentManager instance to access agents

        Returns:
            Number of agents exited
        """
        count = 0
        agent_names = list(self.active_games.keys())

        for agent_name in agent_names:
            agent = agent_manager.get_agent(agent_name)
            if agent:
                self.exit_game_mode(agent)
                count += 1

        logger.warning(f"[GameContext] Force exited {count} agents from game mode")
        return count


# Global instance
game_context_manager = GameContextManager()


class GameContext:
    """
    Convenience wrapper for game mode entry/exit.

    Used by individual game classes to manage player/spectator state.
    This is the primary way games should handle entry and exit, providing:
    1. Pre-game conversation context capture
    2. Game-specific post-game transition messages
    3. Automatic settings restoration via game_context_manager
    """

    def __init__(self, agents, game_name: str, player_names: list):
        """
        Initialize game context wrapper.

        Args:
            agents: List of Agent objects (players + spectators)
            game_name: Name of the game
            player_names: List of player names (not spectators)
        """
        self.agents = agents
        self.game_name = game_name
        self.player_names = player_names
        self.pre_game_context = {}  # agent_name -> last few messages before game

        # Outcome tracking - set these before calling exit()
        self.winner_name: Optional[str] = None
        self.outcome_description: Optional[str] = None  # Custom outcome description

    async def enter(self):
        """
        Enter game mode - capture pre-game conversation context.

        Note: game_orchestrator already calls game_context_manager.enter_game_mode()
        for players, so we don't duplicate that here.
        """
        # Capture pre-game conversation context for each agent
        for agent in self.agents:
            # Get last 5 non-game messages before game started
            recent_messages = []
            with agent.lock:
                for msg in reversed(agent.conversation_history[-20:]):  # Check last 20 messages
                    author = msg.get('author', '')
                    content = msg.get('content', '')

                    # Skip GameMaster, system messages, and [SYSTEM] notifications
                    if 'GameMaster' in author or '(system)' in author or 'System' in author or author == '[SYSTEM]':
                        continue

                    recent_messages.append(f"{author}: {content[:100]}")  # Truncate long messages

                    if len(recent_messages) >= 3:  # Capture last 3 messages
                        break

            # Store in reverse order (chronological)
            self.pre_game_context[agent.name] = list(reversed(recent_messages))

        logger.info(f"[GameContext] Captured pre-game context for {len(self.agents)} agents")

    def set_outcome(self, winner_name: Optional[str] = None, description: Optional[str] = None):
        """
        Set the game outcome before calling exit().

        Args:
            winner_name: Name of the winner (or None for tie/no winner)
            description: Custom outcome description (overrides auto-generated)
        """
        self.winner_name = winner_name
        self.outcome_description = description

    def _build_outcome_summary(self, agent_name: str) -> str:
        """Build outcome summary personalized for each agent."""
        if self.outcome_description:
            return self.outcome_description

        if self.winner_name:
            if agent_name == self.winner_name:
                return f"**You won!** Congratulations!"
            elif self.winner_name in self.player_names:
                return f"**{self.winner_name} won!**"
            else:
                return f"**Winner: {self.winner_name}**"
        else:
            return "The game ended in a tie!" if len(self.player_names) > 1 else "The game has concluded."

    async def exit(self):
        """
        Exit game mode: restore settings and inject game-specific transition messages.

        This provides agents with a checkpoint to help them:
        1. Understand the game outcome
        2. Recall what they were discussing before the game
        3. Return to normal conversation smoothly

        Call set_outcome() before this to include outcome information.
        """
        for agent in self.agents:
            # First, restore settings via game_context_manager (if agent was in game mode)
            if game_context_manager.is_in_game(agent.name):
                game_context_manager.exit_game_mode(agent, inject_transition=False)

            # Build pre-game context string
            pre_game_str = ""
            if agent.name in self.pre_game_context and self.pre_game_context[agent.name]:
                context_messages = self.pre_game_context[agent.name]
                if context_messages:
                    pre_game_str = "\n".join(f"  - {msg}" for msg in context_messages)

            # Build outcome summary (personalized for this agent)
            outcome_summary = self._build_outcome_summary(agent.name)

            # Get game-specific transition message
            transition_message = get_post_game_prompt(
                game_name=self.game_name,
                outcome_summary=outcome_summary,
                pre_game_context=pre_game_str
            )

            # Inject transition message
            agent.add_message_to_history(
                author="[SYSTEM]",
                content=transition_message,
                message_id=None,
                replied_to_agent=None,
                user_id=None
            )

        logger.info(f"[GameContext] Exited game mode for {len(self.agents)} agents with game-specific transitions")
