import discord
from discord.ext import commands
import asyncio
from typing import Optional, Callable, List, Dict, Any
from collections import deque
import threading
import logging
import json
import os
from shortcuts_utils import ShortcutManager
from constants import DiscordConfig, UIConfig

logger = logging.getLogger(__name__)

class DiscordBotClient:
    def __init__(self, agent_manager, message_callback: Optional[Callable] = None, game_orchestrator = None):
        intents = discord.Intents.none()
        intents.guilds = True
        intents.guild_messages = True
        intents.dm_messages = True  # Enable DM reception for admin commands
        intents.message_content = True
        intents.guild_reactions = True  # Enable reaction detection

        self.client = commands.Bot(command_prefix='!', intents=intents, help_command=None, case_insensitive=True)
        self.agent_manager = agent_manager
        self.message_callback = message_callback
        self.game_orchestrator = game_orchestrator

        self.token = ""
        self.channel_id = 0
        self.media_channel_id = 0  # Secondary channel for media-only posts
        self.media_webhook = None  # Webhook for media channel
        self.is_connected = False
        self.status = "disconnected"
        self.message_history: deque = deque(maxlen=DiscordConfig.MESSAGE_HISTORY_MAX_LEN)
        self.lock = threading.Lock()
        self.discord_loop = None
        self.webhook = None
        self.shortcut_manager = ShortcutManager()

        # Reaction polling tracking
        self.checked_reactions: Dict[int, Dict[str, int]] = {}  # message_id -> {emoji: count}
        self.reaction_poll_task = None
        self.periodic_save_task = None  # Periodic save for affinity and config data
        self.startup_time = None  # Will be set when bot connects

        # Agent creation wizard state per user
        self._agent_wizard_sessions: Dict[int, Dict[str, Any]] = {}

        self.setup_events()
        self.setup_commands()

    def set_game_orchestrator(self, game_orchestrator):
        """Set game orchestrator after initialization."""
        self.game_orchestrator = game_orchestrator
        logger.info("[Discord] Game orchestrator connected")

    def process_shortcuts(self, message_content: str) -> str:
        """Process and expand shortcuts in a message (delegated to ShortcutManager)."""
        return self.shortcut_manager.expand_shortcuts_in_message(message_content)

    def load_shortcuts_list(self) -> str:
        """Load and format the shortcuts list for display (delegated to ShortcutManager).
        DEPRECATED: Use load_shortcuts_list_paginated() for full list.
        """
        return self.shortcut_manager.format_shortcuts_list(char_limit=DiscordConfig.SHORTCUTS_DISPLAY_LIMIT)

    def load_shortcuts_list_paginated(self) -> List[str]:
        """Load and format the shortcuts list as multiple messages to show ALL effects."""
        return self.shortcut_manager.format_shortcuts_list_paginated(char_limit=DiscordConfig.SHORTCUTS_DISPLAY_LIMIT)

    async def _handle_admin_dm_command(self, message) -> None:
        """
        Handle admin commands sent via DM to the bot.
        Only accessible by users in ADMIN_USER_IDS.
        """
        content = message.content.strip()
        content_upper = content.upper()

        logger.info(f"[Discord] Admin DM command from {message.author}: {content[:50]}...")

        # !COMMANDS - Show available commands
        if content_upper == "!COMMANDS" or content_upper == "!HELP":
            help_text = """**🔧 BASI-Bot Admin Commands (DM Only)**

**Agent Control:**
• `!STATUS` - Show running/stopped agent counts
• `!AGENTS` - List ALL agent names (for starting)
• `!AGENTINFO <agent>` - Show detailed agent settings
• `!CREATEAGENT` - Start wizard to create a new agent
• `!START <agent>` - Start a specific agent
• `!STOP <agent>` - Stop a specific agent
• `!STARTALL` - Start all agents
• `!STOPALL` - Stop all agents
• `!MODEL <agent> <model>` - Change agent's model
• `!WHISPER <agent> <msg>` - Send divine command (2 turns)

**Media Generation:**
• `!TOGGLEIMAGE <agent>` - Toggle spontaneous image gen
• `!TOGGLEVIDEO <agent>` - Toggle spontaneous video gen
• `!IMAGEMODEL` - Show current image model
• `!IMAGEMODEL <model>` - Set image model

**Presets:**
• `!PRESETS` - List available presets
• `!LOADPRESET <name>` - Load a preset (starts/stops agents)

**Memory Management:**
• `!CLEARVECTOR` - Clear vector memory database
• `!CLEAREFFECTS` - Clear all status effects
• `!CLEAREFFECTS <agent>` - Clear effects for one agent
• `!CLEARGAMES` - Clear game history

**Info:**
• `!COMMANDS` or `!HELP` - Show this help
• `!MODELS` - List popular model IDs"""
            await message.channel.send(help_text)
            return

        # !STATUS - Show agent status
        if content_upper == "!STATUS":
            agents = self.agent_manager.get_all_agents()
            running = [a for a in agents if a.is_running]
            stopped = [a for a in agents if not a.is_running]

            status_text = f"**📊 Agent Status** ({len(running)}/{len(agents)} running)\n\n"
            if running:
                status_text += "**Running:**\n"
                for a in running[:15]:  # Limit to prevent message overflow
                    status_text += f"• {a.name} ({a.model})\n"
                if len(running) > 15:
                    status_text += f"  ...and {len(running) - 15} more\n"

            if stopped:
                status_text += f"\n**Stopped:** {len(stopped)} agents"

            await message.channel.send(status_text)
            return

        # !AGENTS - List all agent names
        if content_upper == "!AGENTS":
            agents = self.agent_manager.get_all_agents()
            running = [a for a in agents if a.is_running]
            stopped = [a for a in agents if not a.is_running]

            # Build response with all agent names
            messages = []
            current_msg = f"**📋 All Agents** ({len(agents)} total)\n\n"

            if running:
                current_msg += "**🟢 Running:**\n"
                for a in running:
                    current_msg += f"• {a.name}\n"

            if stopped:
                current_msg += "\n**⚫ Stopped:**\n"
                for a in stopped:
                    current_msg += f"• {a.name}\n"

            # Split into multiple messages if needed (Discord 2000 char limit)
            if len(current_msg) > 1900:
                # Split by sections
                lines = current_msg.split('\n')
                chunk = ""
                for line in lines:
                    if len(chunk) + len(line) + 1 > 1900:
                        messages.append(chunk)
                        chunk = line + "\n"
                    else:
                        chunk += line + "\n"
                if chunk:
                    messages.append(chunk)
            else:
                messages.append(current_msg)

            for msg in messages:
                await message.channel.send(msg)
            return

        # !CREATEAGENT - Start agent creation wizard
        if content_upper == "!CREATEAGENT":
            await self._start_agent_wizard(message)
            return

        # !START <agent>
        if content_upper.startswith("!START ") and not content_upper.startswith("!STARTALL"):
            agent_name = content[7:].strip()
            if self.agent_manager.start_agent(agent_name):
                await message.channel.send(f"✅ Started agent: **{agent_name}**")
            else:
                # Try case-insensitive match
                agents = self.agent_manager.get_all_agents()
                match = next((a for a in agents if a.name.lower() == agent_name.lower()), None)
                if match:
                    if self.agent_manager.start_agent(match.name):
                        await message.channel.send(f"✅ Started agent: **{match.name}**")
                    else:
                        await message.channel.send(f"⚠️ Agent **{match.name}** is already running")
                else:
                    await message.channel.send(f"❌ Agent not found: **{agent_name}**")
            return

        # !STOP <agent>
        if content_upper.startswith("!STOP ") and not content_upper.startswith("!STOPALL"):
            agent_name = content[6:].strip()
            if self.agent_manager.stop_agent(agent_name):
                await message.channel.send(f"✅ Stopped agent: **{agent_name}**")
            else:
                # Try case-insensitive match
                agents = self.agent_manager.get_all_agents()
                match = next((a for a in agents if a.name.lower() == agent_name.lower()), None)
                if match:
                    if self.agent_manager.stop_agent(match.name):
                        await message.channel.send(f"✅ Stopped agent: **{match.name}**")
                    else:
                        await message.channel.send(f"⚠️ Agent **{match.name}** is not running")
                else:
                    await message.channel.send(f"❌ Agent not found: **{agent_name}**")
            return

        # !STARTALL
        if content_upper == "!STARTALL":
            agents = self.agent_manager.get_all_agents()
            started = 0
            for agent in agents:
                if not agent.is_running:
                    if self.agent_manager.start_agent(agent.name):
                        started += 1
            await message.channel.send(f"✅ Started **{started}** agents")
            return

        # !STOPALL
        if content_upper == "!STOPALL":
            self.agent_manager.stop_all_agents()
            await message.channel.send("✅ Stopped all agents")
            return

        # !MODEL <agent> <model>
        if content_upper.startswith("!MODEL "):
            args = content[7:].strip()
            if not args or ' ' not in args:
                await message.channel.send("❌ Usage: `!MODEL <agent_name> <model_id>`")
                return
            # Match against known agent names (sorted by length, longest first)
            agents = self.agent_manager.get_all_agents()
            agents_sorted = sorted(agents, key=lambda a: len(a.name), reverse=True)
            matched_agent = None
            model = None
            for agent in agents_sorted:
                if args.lower().startswith(agent.name.lower() + " "):
                    matched_agent = agent
                    model = args[len(agent.name):].strip()
                    break
            if matched_agent and model:
                if self.agent_manager.update_agent(matched_agent.name, model=model):
                    await message.channel.send(f"✅ Updated **{matched_agent.name}** model to: `{model}`")
                else:
                    await message.channel.send(f"❌ Failed to update model for **{matched_agent.name}**")
            else:
                await message.channel.send(f"❌ Agent not found or invalid format. Usage: `!MODEL <agent_name> <model_id>`")
            return

        # !PRESETS
        if content_upper == "!PRESETS":
            try:
                from presets_manager import presets_manager
                presets = presets_manager.get_preset_names()
                if presets:
                    await message.channel.send(f"**📋 Available Presets:**\n" + "\n".join(f"• {p}" for p in presets))
                else:
                    await message.channel.send("No presets available")
            except Exception as e:
                await message.channel.send(f"❌ Error loading presets: {e}")
            return

        # !LOADPRESET <name>
        if content_upper.startswith("!LOADPRESET "):
            preset_name = content[12:].strip()
            try:
                from presets_manager import presets_manager
                preset = presets_manager.get_preset(preset_name)
                if not preset:
                    # Try case-insensitive
                    all_presets = presets_manager.get_preset_names()
                    match = next((p for p in all_presets if p.lower() == preset_name.lower()), None)
                    if match:
                        preset = presets_manager.get_preset(match)
                        preset_name = match

                if preset:
                    agent_names = preset.get('agent_names', [])
                    # Stop agents not in preset
                    all_agents = self.agent_manager.get_all_agents()
                    stopped = 0
                    for agent in all_agents:
                        if agent.name not in agent_names and agent.is_running:
                            self.agent_manager.stop_agent(agent.name)
                            stopped += 1
                    # Start agents in preset
                    started = 0
                    for name in agent_names:
                        agent = self.agent_manager.get_agent(name)
                        if agent and not agent.is_running:
                            self.agent_manager.start_agent(name)
                            started += 1
                    await message.channel.send(f"✅ Loaded preset **{preset_name}**: started {started}, stopped {stopped}")
                else:
                    await message.channel.send(f"❌ Preset not found: **{preset_name}**")
            except Exception as e:
                await message.channel.send(f"❌ Error loading preset: {e}")
            return

        # !WHISPER <agent> <message> - Divine command to agent
        if content_upper.startswith("!WHISPER "):
            from shortcuts_utils import StatusEffectManager
            # Parse: !WHISPER Agent Name message here
            args = content[9:].strip()  # Remove "!WHISPER "
            if not args:
                await message.channel.send("❌ Usage: `!WHISPER <Agent Name> <message>`\nExample: `!WHISPER John McAfee Tell everyone about your crypto schemes`")
                return

            # Match against known agent names (sorted by length, longest first to avoid partial matches)
            agents = self.agent_manager.get_all_agents()
            agents_sorted = sorted(agents, key=lambda a: len(a.name), reverse=True)
            matched_agent = None
            remaining_message = ""

            for agent in agents_sorted:
                # Must match full agent name followed by a space
                if args.lower().startswith(agent.name.lower() + " "):
                    matched_agent = agent
                    remaining_message = args[len(agent.name):].strip()
                    break

            if matched_agent and remaining_message:
                StatusEffectManager.apply_whisper(matched_agent.name, remaining_message)
                await message.channel.send(
                    f"👁️ **Whispered to {matched_agent.name}** (2 turns):\n"
                    f"*\"{remaining_message[:200]}{'...' if len(remaining_message) > 200 else ''}\"*"
                )
            elif not matched_agent:
                await message.channel.send(f"❌ No matching agent found. Available agents:\n{', '.join(a.name for a in agents)}")
            else:
                await message.channel.send("❌ Please include a message to whisper.\nUsage: `!WHISPER <Agent Name> <message>`")
            return

        # !CLEARVECTOR
        if content_upper == "!CLEARVECTOR":
            try:
                if self.agent_manager.vector_store:
                    self.agent_manager.vector_store.clear_all()
                    await message.channel.send("✅ Vector memory cleared")
                else:
                    await message.channel.send("⚠️ Vector store not initialized")
            except Exception as e:
                await message.channel.send(f"❌ Error clearing vector memory: {e}")
            return

        # !CLEAREFFECTS [agent]
        if content_upper.startswith("!CLEAREFFECTS"):
            from shortcuts_utils import StatusEffectManager
            if content_upper == "!CLEAREFFECTS":
                # Clear all effects
                StatusEffectManager.clear_all_effects_globally()
                await message.channel.send("✅ Cleared all status effects")
            else:
                # Clear for specific agent
                agent_name = content[13:].strip()
                agents = self.agent_manager.get_all_agents()
                match = next((a for a in agents if a.name.lower() == agent_name.lower()), None)
                if match:
                    StatusEffectManager.clear_all_effects(match.name)
                    await message.channel.send(f"✅ Cleared status effects for **{match.name}**")
                else:
                    await message.channel.send(f"❌ Agent not found: **{agent_name}**")
            return

        # !CLEARGAMES
        if content_upper == "!CLEARGAMES":
            try:
                from agent_games.game_manager import game_manager
                if game_manager:
                    count = game_manager.clear_history()
                    await message.channel.send(f"✅ Cleared {count} game records")
                else:
                    await message.channel.send("⚠️ Game manager not available")
            except Exception as e:
                await message.channel.send(f"❌ Error clearing game history: {e}")
            return

        # !TOGGLEIMAGE <agent> - Toggle spontaneous image generation
        if content_upper.startswith("!TOGGLEIMAGE "):
            agent_name = content[13:].strip()
            agents = self.agent_manager.get_all_agents()
            match = next((a for a in agents if a.name.lower() == agent_name.lower()), None)
            if match:
                new_value = not getattr(match, 'allow_spontaneous_images', False)
                if self.agent_manager.update_agent(match.name, allow_spontaneous_images=new_value):
                    status = "✅ ENABLED" if new_value else "❌ DISABLED"
                    await message.channel.send(f"🖼️ Spontaneous images for **{match.name}**: {status}")
                else:
                    await message.channel.send(f"❌ Failed to update **{match.name}**")
            else:
                await message.channel.send(f"❌ Agent not found: **{agent_name}**")
            return

        # !TOGGLEVIDEO <agent> - Toggle spontaneous video generation
        if content_upper.startswith("!TOGGLEVIDEO "):
            agent_name = content[13:].strip()
            agents = self.agent_manager.get_all_agents()
            match = next((a for a in agents if a.name.lower() == agent_name.lower()), None)
            if match:
                new_value = not getattr(match, 'allow_spontaneous_videos', False)
                if self.agent_manager.update_agent(match.name, allow_spontaneous_videos=new_value):
                    status = "✅ ENABLED" if new_value else "❌ DISABLED"
                    await message.channel.send(f"🎬 Spontaneous videos for **{match.name}**: {status}")
                else:
                    await message.channel.send(f"❌ Failed to update **{match.name}**")
            else:
                await message.channel.send(f"❌ Agent not found: **{agent_name}**")
            return

        # !IMAGEMODEL [model] - Show or set the global image model
        if content_upper.startswith("!IMAGEMODEL"):
            from config_manager import config_manager
            if content_upper == "!IMAGEMODEL":
                # Show current model
                current = config_manager.load_image_model()
                await message.channel.send(f"🖼️ Current image model: `{current}`")
            else:
                # Set new model
                new_model = content[11:].strip()
                if new_model:
                    config_manager.save_image_model(new_model)
                    self.agent_manager.set_image_model(new_model)
                    await message.channel.send(f"✅ Image model set to: `{new_model}`")
                else:
                    await message.channel.send("❌ Usage: `!IMAGEMODEL <model_id>`")
            return

        # !AGENTINFO <agent> - Show detailed agent settings
        if content_upper.startswith("!AGENTINFO "):
            agent_name = content[11:].strip()
            agents = self.agent_manager.get_all_agents()
            match = next((a for a in agents if a.name.lower() == agent_name.lower()), None)
            if match:
                img_status = "✅" if getattr(match, 'allow_spontaneous_images', False) else "❌"
                vid_status = "✅" if getattr(match, 'allow_spontaneous_videos', False) else "❌"
                running_status = "🟢 Running" if match.is_running else "⚫ Stopped"

                info = f"""**📋 Agent Info: {match.name}**

**Status:** {running_status}
**Model:** `{match.model}`

**Response Settings:**
• Frequency: {match.response_frequency}s
• Likelihood: {match.response_likelihood}%
• Max Tokens: {match.max_tokens}

**Attention:**
• User Attention: {match.user_attention}%
• Bot Awareness: {match.bot_awareness}%

**Media Generation:**
• Spontaneous Images: {img_status}
• Spontaneous Videos: {vid_status}
• Image Gen Turns: {getattr(match, 'image_gen_turns', 5)}
• Video Gen Turns: {getattr(match, 'video_gen_turns', 10)}"""
                await message.channel.send(info)
            else:
                await message.channel.send(f"❌ Agent not found: **{agent_name}**")
            return

        # !MODELS - List popular models
        if content_upper == "!MODELS":
            models_text = """**📚 Popular Model IDs:**

**OpenRouter:**
• `google/gemini-2.5-flash` (fast, cheap)
• `google/gemini-2.5-flash-preview-09-2025` (preview)
• `anthropic/claude-sonnet-4` (balanced)
• `anthropic/claude-opus-4` (powerful)
• `openai/gpt-4o` (GPT-4 Omni)
• `openai/gpt-4o-mini` (fast GPT-4)
• `deepseek/deepseek-chat` (cheap, good)
• `meta-llama/llama-3.1-405b-instruct` (large)

Use with: `!MODEL <agent> <model_id>`"""
            await message.channel.send(models_text)
            return

        # Unknown command - but first check if user is in a wizard session
        user_id = message.author.id
        if user_id in self._agent_wizard_sessions:
            await self._handle_agent_wizard_input(message)
            return

        await message.channel.send(f"❓ Unknown command. Type `!COMMANDS` for help.")

    async def _start_agent_wizard(self, message) -> None:
        """Start the agent creation wizard for an admin user."""
        user_id = message.author.id

        # Initialize wizard session with McAfee-based defaults
        self._agent_wizard_sessions[user_id] = {
            "step": 0,
            "data": {
                "name": None,
                "model": "google/gemini-2.5-flash",
                "system_prompt": None,
                "response_frequency": 90,
                "response_likelihood": 80,
                "max_tokens": 600,
                "user_attention": 90,
                "bot_awareness": 60,
                "message_retention": 2
            }
        }

        welcome = """**🧙 Agent Creation Wizard**

I'll guide you through creating a new agent step by step.
Type `!CANCEL` at any time to abort, or `!SKIP` to use the default for any setting.

**Step 1/8: Agent Name**
What should this agent be called?
(e.g., "Professor Oak", "Sassy Bot", "Captain Picard")"""

        await message.channel.send(welcome)

    async def _handle_agent_wizard_input(self, message) -> None:
        """Handle input during agent creation wizard."""
        user_id = message.author.id
        content = message.content.strip()

        if user_id not in self._agent_wizard_sessions:
            return

        content_upper = content.upper()

        # Check for cancel
        if content_upper == "!CANCEL":
            del self._agent_wizard_sessions[user_id]
            await message.channel.send("❌ Agent creation cancelled.")
            return

        session = self._agent_wizard_sessions[user_id]
        step = session["step"]
        data = session["data"]

        # Step 0: Agent Name
        if step == 0:
            name = content.strip()
            if not name:
                await message.channel.send("❌ Name cannot be empty. Please enter a name:")
                return
            if len(name) > 50:
                await message.channel.send("❌ Name too long (max 50 chars). Please enter a shorter name:")
                return

            existing = self.agent_manager.get_agent(name)
            if existing:
                await message.channel.send(f"❌ An agent named **{name}** already exists. Please choose a different name:")
                return

            data["name"] = name
            session["step"] = 1

            await message.channel.send(f"""✅ Agent name: **{name}**

**Step 2/8: Model**
Which AI model should power this agent?

Popular options:
• `google/gemini-2.5-flash` (fast, cheap) **[DEFAULT]**
• `anthropic/claude-sonnet-4` (balanced)
• `openai/gpt-4o-mini` (fast GPT-4)
• `deepseek/deepseek-chat` (cheap, good)

Enter a model ID, or `!SKIP` for default:""")
            return

        # Step 1: Model
        if step == 1:
            if content_upper != "!SKIP" and content.strip():
                data["model"] = content.strip()
            session["step"] = 2

            await message.channel.send(f"""✅ Model: `{data['model']}`

**Step 3/8: System Prompt (Personality)**
This defines who the agent IS - their personality, knowledge, quirks, and how they communicate.

Write the prompt across multiple messages. Send `!DONE` when finished.

Example:
```
You are a grumpy wizard who secretly loves dad jokes.
You speak in a medieval style but slip into modern slang when excited.
You're obsessed with proper spell pronunciation.
```

Enter the system prompt:""")
            session["prompt_buffer"] = []
            return

        # Step 2: System Prompt (multi-line)
        if step == 2:
            if content_upper.startswith("!DONE"):
                prompt_text = "\n".join(session.get("prompt_buffer", []))
                if not prompt_text.strip():
                    await message.channel.send("❌ System prompt cannot be empty. Enter at least one line, then `!DONE`:")
                    return

                data["system_prompt"] = prompt_text
                session["step"] = 3

                await message.channel.send(f"""✅ System prompt saved ({len(prompt_text)} chars)

**Step 4/8: Response Frequency**
How many conversation turns should pass before this agent considers responding?

• Lower = more talkative (responds more often)
• Higher = more reserved (waits longer between responses)

**Default: {data['response_frequency']}** (good balance for group chats)
Range: 5-200

Enter a number, or `!SKIP` for default:""")
                return
            else:
                session.setdefault("prompt_buffer", []).append(content)
                return

        # Step 3: Response Frequency
        if step == 3:
            if content_upper != "!SKIP" and content.strip():
                try:
                    val = int(content)
                    data["response_frequency"] = max(5, min(200, val))
                except ValueError:
                    await message.channel.send("❌ Please enter a number (5-200), or `!SKIP`:")
                    return
            session["step"] = 4

            await message.channel.send(f"""✅ Response Frequency: **{data['response_frequency']}** turns

**Step 5/8: Response Likelihood**
When the agent IS eligible to respond, what % chance should they actually respond?

• 100% = always responds when eligible
• 50% = coin flip
• Lower = more selective/random

**Default: {data['response_likelihood']}%**
Range: 0-100

Enter a percentage, or `!SKIP` for default:""")
            return

        # Step 4: Response Likelihood
        if step == 4:
            if content_upper != "!SKIP" and content.strip():
                try:
                    val = int(content.replace("%", ""))
                    data["response_likelihood"] = max(0, min(100, val))
                except ValueError:
                    await message.channel.send("❌ Please enter a number (0-100), or `!SKIP`:")
                    return
            session["step"] = 5

            await message.channel.send(f"""✅ Response Likelihood: **{data['response_likelihood']}%**

**Step 6/8: Max Tokens**
Maximum length of the agent's responses (in tokens, ~4 chars each).

• 300 = short, punchy responses
• 600 = medium length (good default)
• 1000+ = longer, more detailed responses

**Default: {data['max_tokens']}**
Range: 100-4000

Enter a number, or `!SKIP` for default:""")
            return

        # Step 5: Max Tokens
        if step == 5:
            if content_upper != "!SKIP" and content.strip():
                try:
                    val = int(content)
                    data["max_tokens"] = max(100, min(4000, val))
                except ValueError:
                    await message.channel.send("❌ Please enter a number (100-4000), or `!SKIP`:")
                    return
            session["step"] = 6

            await message.channel.send(f"""✅ Max Tokens: **{data['max_tokens']}**

**Step 7/8: User Attention**
How much priority should this agent give to HUMAN messages vs bot messages?

• 100% = strongly prioritizes responding to humans
• 50% = treats humans and bots equally
• Lower = more likely to ignore humans, engage with bots

**Default: {data['user_attention']}%**
Range: 0-100

Enter a percentage, or `!SKIP` for default:""")
            return

        # Step 6: User Attention
        if step == 6:
            if content_upper != "!SKIP" and content.strip():
                try:
                    val = int(content.replace("%", ""))
                    data["user_attention"] = max(0, min(100, val))
                except ValueError:
                    await message.channel.send("❌ Please enter a number (0-100), or `!SKIP`:")
                    return
            session["step"] = 7

            await message.channel.send(f"""✅ User Attention: **{data['user_attention']}%**

**Step 8/8: Bot Awareness**
How much attention should this agent pay to OTHER bots in the chat?

• 100% = very engaged with other bots
• 50% = moderate awareness
• Lower = mostly ignores other bots

**Default: {data['bot_awareness']}%**
Range: 0-100

Enter a percentage, or `!SKIP` for default:""")
            return

        # Step 7: Bot Awareness
        if step == 7:
            if content_upper != "!SKIP" and content.strip():
                try:
                    val = int(content.replace("%", ""))
                    data["bot_awareness"] = max(0, min(100, val))
                except ValueError:
                    await message.channel.send("❌ Please enter a number (0-100), or `!SKIP`:")
                    return

            # Create the agent
            success = self.agent_manager.add_agent(
                name=data["name"],
                model=data["model"],
                system_prompt=data["system_prompt"],
                response_frequency=data["response_frequency"],
                response_likelihood=data["response_likelihood"],
                max_tokens=data["max_tokens"],
                user_attention=data["user_attention"],
                bot_awareness=data["bot_awareness"],
                message_retention=data["message_retention"]
            )

            if success:
                if self.agent_manager.save_data_callback:
                    self.agent_manager.save_data_callback()

                summary = f"""🎉 **Agent Created Successfully!**

**Name:** {data['name']}
**Model:** `{data['model']}`
**Prompt:** {len(data['system_prompt'])} characters

**Behavior Settings:**
• Response Frequency: {data['response_frequency']} turns
• Response Likelihood: {data['response_likelihood']}%
• Max Tokens: {data['max_tokens']}
• User Attention: {data['user_attention']}%
• Bot Awareness: {data['bot_awareness']}%

The agent is created but **not running**.
Use `!START {data['name']}` to activate it!"""
                await message.channel.send(summary)
            else:
                await message.channel.send(f"❌ Failed to create agent. An agent named **{data['name']}** may already exist.")

            del self._agent_wizard_sessions[user_id]
            return

    async def _extract_replied_to_agent(self, message) -> Optional[str]:
        """
        Extract the agent name if this message is a reply to one of our bots.

        Args:
            message: Discord message object

        Returns:
            Agent name if replying to our bot, None otherwise
        """
        if not message.reference or not message.reference.message_id:
            return None

        try:
            # Fetch the message being replied to
            referenced_msg = await message.channel.fetch_message(message.reference.message_id)
            if not referenced_msg or not referenced_msg.content:
                return None

            # Check if it's from one of our bots (format: "**[Agent Name]:** content")
            ref_content = referenced_msg.content
            if ref_content.startswith("**[") and "]:**" in ref_content:
                # Extract the agent name from the referenced message
                agent_name_match = ref_content.split("]:**", 1)
                if len(agent_name_match) == 2:
                    agent_name = agent_name_match[0].replace("**[", "").strip()
                    logger.info(f"[Discord] User {message.author.display_name} replied to bot: {agent_name}")
                    return agent_name
        except Exception as e:
            logger.warning(f"[Discord] Could not fetch referenced message: {e}")

        return None

    def _extract_agent_name_from_webhook(self, content: str, author_name: str) -> tuple:
        """
        Extract actual content from webhook message format and clean author name.

        Webhook messages have format: "**[Agent Name]:** content"
        This extracts just the content part.

        Also strips model suffixes from author names (e.g., "Agent (model-name)" -> "Agent")
        to prevent spectators from quoting with full model names.

        Args:
            content: Full message content
            author_name: Original author name (may include model suffix)

        Returns:
            Tuple of (cleaned_content, cleaned_author_name)
        """
        # Strip model suffix from author name (e.g., "The Tumblrer (deepseek-chat)" -> "The Tumblrer")
        import re
        cleaned_author = re.sub(r'\s*\([^)]+\)\s*$', '', author_name).strip()

        if content.startswith("**[") and "]:**" in content:
            match = content.split("]:**", 1)
            if len(match) == 2:
                # Extract the actual content (strip the **[Agent Name]:** prefix)
                actual_content = match[1].strip()
                return actual_content, cleaned_author

        return content, cleaned_author

    def _route_message_to_agents(self, author_name: str, content: str, message_id: int,
                                 replied_to_agent: Optional[str], user_id: str):
        """
        Route message to appropriate agents based on content filtering rules.

        Rules:
        - [IMAGE] tags only go to image model agents
        - Image agent failures are hidden from all agents
        - Everything else goes to all agents

        Args:
            author_name: Message author's name
            content: Message content
            message_id: Discord message ID
            replied_to_agent: Agent being replied to (if any)
            user_id: User's Discord ID
        """
        if "[IMAGE]" in content:
            # Only add to image model agents
            logger.info(f"[Discord] Message contains [IMAGE] - adding only to image model agents")
            self.agent_manager.add_message_to_image_agents_only(author_name, content, message_id, replied_to_agent, user_id)
        elif "Failed to generate image" in content and any(img_model in author_name.lower() for img_model in ["image", "artist", "dall-e", "midjourney"]):
            # Hide image agent failures from all agents
            logger.info(f"[Discord] Image agent failure message - not adding to agent histories")
        else:
            # Normal message - add to all agents
            self.agent_manager.add_message_to_all_agents(author_name, content, message_id, replied_to_agent, user_id)
            logger.info(f"[Discord] Message added to all agent histories with ID: {message_id}")

    def setup_events(self):
        @self.client.event
        async def on_ready():
            self.is_connected = True
            self.status = "connected"
            print(f"Discord bot logged in as {self.client.user}")

            # Record startup time to ignore old messages
            import time
            self.startup_time = time.time()
            logger.info(f"[Discord] Bot startup time recorded: {self.startup_time}")

            # Start reaction polling task
            if not self.reaction_poll_task or self.reaction_poll_task.done():
                self.reaction_poll_task = asyncio.create_task(self.poll_reactions())
                logger.info("[Discord] Started reaction polling task")

            # Start periodic save task (every 5 minutes)
            if not self.periodic_save_task or self.periodic_save_task.done():
                self.periodic_save_task = asyncio.create_task(self.periodic_save())
                logger.info("[Discord] Started periodic save task")

            # Start game auto-play monitor if available
            if self.game_orchestrator:
                await self.game_orchestrator.start_auto_play_monitor()
                logger.info("[Discord] Started game auto-play monitor")

        @self.client.event
        async def on_command_error(ctx, error):
            # Silently ignore CommandNotFound errors (triggered by !SHORTCUT commands)
            if isinstance(error, commands.CommandNotFound):
                return
            # Log other errors normally
            logger.error(f"[Discord] Command error: {error}")

        @self.client.event
        async def on_message(message):
            # Ignore messages from self
            if message.author == self.client.user:
                return

            # Check for admin DM commands BEFORE channel filtering
            if isinstance(message.channel, discord.DMChannel):
                user_id = str(message.author.id)
                admin_ids = DiscordConfig.get_admin_user_ids()
                logger.info(f"[Discord] DM received from {message.author} (ID: {user_id})")
                logger.info(f"[Discord] Admin IDs list: {admin_ids}, checking if '{user_id}' in list...")
                # Only process DM commands from admin users
                if user_id in admin_ids:
                    await self._handle_admin_dm_command(message)
                else:
                    logger.info(f"[Discord] DM from non-admin user {message.author} (ID: {user_id}) - ignoring")
                return  # Don't process DMs as regular messages

            # Ignore messages from wrong channel (for regular chat)
            if self.channel_id and message.channel.id != self.channel_id:
                return

            author_name = message.author.display_name
            content = message.content

            # Extract replied-to agent name if this is a reply to our bot
            replied_to_agent = await self._extract_replied_to_agent(message)

            # Handle shortcuts command
            if content.startswith("!shortcuts") or content.startswith("/shortcuts"):
                logger.info(f"[Discord] Shortcuts command triggered by {author_name}")
                # Use paginated version to show ALL shortcuts
                shortcuts_pages = self.load_shortcuts_list_paginated()
                for page in shortcuts_pages:
                    await message.channel.send(page)
                    # Small delay between pages to maintain order
                    if len(shortcuts_pages) > 1:
                        await asyncio.sleep(0.3)
                return

            # Handle IDCC spitball submissions during writers' room (humans participating in voting/pitching)
            try:
                from agent_games.interdimensional_cable import idcc_manager
                if idcc_manager.is_game_active() and idcc_manager.active_game:
                    game = idcc_manager.active_game
                    # Check if collecting spitball inputs from humans
                    if game.state and game.state.spitball_collecting:
                        success = await game.handle_spitball_submission(author_name, content)
                        if success:
                            await message.add_reaction("✅")
                            logger.info(f"[Discord] {author_name} submitted spitball input for IDCC ({game.state.spitball_round_name})")
                            # Continue processing - let the message be seen by everyone
            except ImportError:
                pass
            except Exception as e:
                logger.error(f"[Discord] Error checking IDCC spitball: {e}")

            # Handle [SCENE] submissions for Interdimensional Cable game (only when IDCC is active)
            if "[SCENE]" in content.upper():
                try:
                    from agent_games.interdimensional_cable import idcc_manager
                    if idcc_manager.is_game_active() and idcc_manager.active_game:
                        # Check if this user is the one we're waiting for
                        if idcc_manager.active_game.state and idcc_manager.active_game.state.waiting_for_human_scene:
                            success = await idcc_manager.active_game.handle_scene_submission(
                                author_name,
                                content
                            )
                            if success:
                                await message.add_reaction("✅")
                                logger.info(f"[Discord] {author_name} submitted [SCENE] for IDCC")
                                return  # Don't process as regular message
                except ImportError:
                    pass  # IDCC not available
                except Exception as e:
                    logger.error(f"[Discord] Error checking IDCC scene: {e}")

            # Extract actual content from webhook message format
            content, author_name = self._extract_agent_name_from_webhook(content, author_name)

            logger.info(f"[Discord] Received message from {author_name}: {content[:50]}...")

            # Track human activity for auto-play system
            if not message.author.bot and self.game_orchestrator:
                self.game_orchestrator.update_human_activity()
                logger.debug(f"[Discord] Human activity tracked for auto-play")

            # Add to message history
            with self.lock:
                msg_data = {
                    "author": author_name,
                    "content": content,
                    "timestamp": message.created_at.timestamp()
                }
                if replied_to_agent:
                    msg_data["replied_to_agent"] = replied_to_agent
                self.message_history.append(msg_data)

            # Check if this is a bot command - don't route commands to agents
            # Commands start with ! followed by known command names
            KNOWN_COMMANDS = ['idcc', 'join-idcc', 'shortcuts']
            content_lower = content.lower().strip()
            is_bot_command = any(content_lower.startswith(f'!{cmd}') for cmd in KNOWN_COMMANDS)

            if is_bot_command:
                logger.info(f"[Discord] Bot command detected: {content[:30]}... - not routing to agents")
                # Process commands and return early - don't add to agent histories
                await self.client.process_commands(message)
                return

            # Check for status effect shortcuts and apply them BEFORE routing
            # This ensures effects are active when agents process the message
            shortcuts_found = self.shortcut_manager.find_shortcuts_in_message(content)
            if shortcuts_found and not message.author.bot:
                # Apply effects via AgentManager
                applied = self.agent_manager.process_shortcuts_in_message(content, message.id)
                if applied:
                    # React to confirm effects applied
                    try:
                        await message.add_reaction("\U0001F9EA")  # test tube emoji
                    except:
                        pass

            # Route message to appropriate agents
            self._route_message_to_agents(author_name, content, message.id, replied_to_agent, str(message.author.id))

            # Trigger UI callback if configured
            if self.message_callback:
                await self.message_callback(author_name, content)

            # Process commands (for commands.Bot)
            await self.client.process_commands(message)

        @self.client.event
        async def on_disconnect():
            logger.warning(f"[Discord] Bot disconnected from Discord")
            self.is_connected = False
            self.status = "disconnected"

        @self.client.event
        async def on_resumed():
            logger.info(f"[Discord] Connection resumed successfully")
            self.is_connected = True
            self.status = "connected"

        @self.client.event
        async def on_error(event, *args, **kwargs):
            self.status = "error"
            logger.error(f"[Discord] Error in {event}: {args}, {kwargs}", exc_info=True)

        @self.client.event
        async def on_reaction_add(reaction, user):
            """Track emoji reactions on bot messages for dopamine boost."""
            # Ignore reactions from the bot itself
            if user == self.client.user:
                return

            # Only process reactions in the configured channel
            if self.channel_id and reaction.message.channel.id != self.channel_id:
                return

            message = reaction.message

            # Check if this is a bot message (format: "**[Agent Name]:** content")
            if message.content and message.content.startswith("**[") and "]:**" in message.content:
                # Extract agent name
                try:
                    agent_name_match = message.content.split("]:**", 1)
                    if len(agent_name_match) == 2:
                        agent_name = agent_name_match[0].replace("**[", "").strip()

                        # Get reaction emoji (can be custom or unicode)
                        emoji_str = str(reaction.emoji)

                        logger.info(f"[Discord] Reaction {emoji_str} added by {user.display_name} to {agent_name}'s message (ID: {message.id})")

                        # Notify agent manager about the reaction
                        if hasattr(self.agent_manager, 'handle_reaction'):
                            await self.agent_manager.handle_reaction(
                                agent_name=agent_name,
                                message_id=message.id,
                                emoji=emoji_str,
                                user_name=user.display_name,
                                reaction_count=reaction.count
                            )

                except Exception as e:
                    logger.error(f"[Discord] Error processing reaction: {e}", exc_info=True)

        @self.client.event
        async def on_raw_reaction_add(payload):
            """Handle reactions to webhook messages (not in cache)."""
            # Ignore reactions from the bot itself
            if payload.user_id == self.client.user.id:
                return

            # Only process reactions in the configured channel
            if self.channel_id and payload.channel_id != self.channel_id:
                return

            try:
                # Fetch the channel and message
                channel = self.client.get_channel(payload.channel_id)
                if not channel:
                    return

                message = await channel.fetch_message(payload.message_id)
                if not message:
                    return

                # Check if this is a bot/webhook message (format: "**[Agent Name]:** content")
                if message.content and message.content.startswith("**[") and "]:**" in message.content:
                    # Extract agent name
                    agent_name_match = message.content.split("]:**", 1)
                    if len(agent_name_match) == 2:
                        agent_name = agent_name_match[0].replace("**[", "").strip()

                        # Get user who reacted
                        user = self.client.get_user(payload.user_id)
                        if not user:
                            user = await self.client.fetch_user(payload.user_id)

                        # Get reaction emoji
                        emoji_str = str(payload.emoji)

                        logger.info(f"[Discord] Reaction {emoji_str} added by {user.display_name} to {agent_name}'s message (ID: {message.id})")

                        # Notify agent manager about the reaction
                        if hasattr(self.agent_manager, 'handle_reaction'):
                            await self.agent_manager.handle_reaction(
                                agent_name=agent_name,
                                message_id=message.id,
                                emoji=emoji_str,
                                user_name=user.display_name,
                                reaction_count=1  # Raw events don't have count
                            )

            except Exception as e:
                logger.error(f"[Discord] Error processing raw reaction: {e}", exc_info=True)

    async def poll_reactions(self):
        """Background task to poll for reactions every 60 seconds."""
        await self.client.wait_until_ready()

        logger.info("[Discord] Starting reaction polling task (every 60s)")

        while not self.client.is_closed():
            try:
                await asyncio.sleep(60)  # Wait 60 seconds between checks

                if not self.channel_id or not self.is_connected:
                    continue

                # Get the channel
                channel = self.client.get_channel(self.channel_id)
                if not channel:
                    continue

                # Fetch last 10 messages
                messages = []
                async for msg in channel.history(limit=10):
                    messages.append(msg)

                logger.info(f"[Discord] Polling reactions on {len(messages)} recent messages")

                agent_messages_checked = 0
                total_reactions_found = 0
                new_reactions_found = 0

                # Check each message for reactions
                for message in messages:
                    # Skip messages created before bot startup (to avoid old reactions)
                    if self.startup_time and message.created_at.timestamp() < self.startup_time:
                        continue

                    # Check if this is a webhook message (agent messages are sent via webhook)
                    if not message.webhook_id:
                        continue

                    # Extract agent name from webhook username (format: "Agent Name (model)")
                    # Example: "John McAfee (grok-4.1-fast)" -> "John McAfee"
                    try:
                        author_name = message.author.name
                        # Remove model suffix if present
                        if " (" in author_name and author_name.endswith(")"):
                            agent_name = author_name.rsplit(" (", 1)[0].strip()
                        else:
                            agent_name = author_name.strip()

                        agent_messages_checked += 1

                        # Log if this message has reactions
                        if message.reactions:
                            reaction_summary = ", ".join([f"{r.emoji}({r.count})" for r in message.reactions])
                            logger.info(f"[Discord] Message {message.id} from {agent_name} has reactions: {reaction_summary}")
                            total_reactions_found += len(message.reactions)

                        # Check each reaction on the message
                        for reaction in message.reactions:
                            emoji_str = str(reaction.emoji)
                            current_count = reaction.count

                            # Get previously tracked count for this message/emoji combo
                            if message.id not in self.checked_reactions:
                                self.checked_reactions[message.id] = {}

                            previous_count = self.checked_reactions[message.id].get(emoji_str, 0)

                            # If count increased, we have new reactions
                            if current_count > previous_count:
                                new_reactions = current_count - previous_count
                                new_reactions_found += new_reactions

                                logger.info(f"[Discord] Found {new_reactions} new {emoji_str} reaction(s) on {agent_name}'s message (ID: {message.id})")

                                # Update our tracking
                                self.checked_reactions[message.id][emoji_str] = current_count

                                # Notify agent manager
                                if hasattr(self.agent_manager, 'handle_reaction'):
                                    # Get users who reacted (note: this is simplified, we don't track individual users in polling)
                                    async for user in reaction.users():
                                        if user.id == self.client.user.id:
                                            continue  # Skip bot's own reactions

                                        await self.agent_manager.handle_reaction(
                                            agent_name=agent_name,
                                            message_id=message.id,
                                            emoji=emoji_str,
                                            user_name=user.display_name,
                                            reaction_count=current_count
                                        )

                            # Update count even if no new reactions (for future comparisons)
                            elif current_count == previous_count and previous_count > 0:
                                pass  # No change
                            else:
                                # First time seeing this emoji on this message
                                self.checked_reactions[message.id][emoji_str] = current_count

                    except Exception as e:
                        logger.error(f"[Discord] Error processing message {message.id} reactions: {e}", exc_info=True)

                # Summary of polling results
                logger.info(f"[Discord] Reaction poll complete: checked {agent_messages_checked} agent messages, found {total_reactions_found} total reactions, {new_reactions_found} new reactions detected")

                # Clean up old tracked messages (keep last 50)
                if len(self.checked_reactions) > 50:
                    # Remove oldest entries
                    sorted_ids = sorted(self.checked_reactions.keys())
                    for old_id in sorted_ids[:-50]:
                        del self.checked_reactions[old_id]

            except Exception as e:
                logger.error(f"[Discord] Error in reaction polling: {e}", exc_info=True)

        logger.info("[Discord] Reaction polling task stopped")

    async def periodic_save(self):
        """Background task to save affinity and config data every 5 minutes."""
        await self.client.wait_until_ready()

        logger.info("[Discord] Starting periodic save task (every 5 minutes)")

        while not self.client.is_closed():
            try:
                await asyncio.sleep(300)  # 5 minutes

                # Call save_data_callback if available
                if hasattr(self.agent_manager, 'save_data_callback') and self.agent_manager.save_data_callback:
                    try:
                        self.agent_manager.save_data_callback()
                        logger.info("[Discord] Periodic save completed (affinity + config)")
                    except Exception as e:
                        logger.error(f"[Discord] Error in periodic save callback: {e}", exc_info=True)

            except asyncio.CancelledError:
                logger.info("[Discord] Periodic save task cancelled")
                break
            except Exception as e:
                logger.error(f"[Discord] Error in periodic save: {e}", exc_info=True)

        logger.info("[Discord] Periodic save task stopped")

    def setup_commands(self):
        """Setup game commands for the bot."""

        # Manual game commands disabled - games only start via auto-play
        # Uncomment below to enable manual game starting

        # @self.client.command(name='play')
        # async def play_game(ctx: commands.Context, game_name: str = None, player1: discord.Member = None, player2: discord.Member = None):
        #     """Start a game between users."""
        #     pass

        @self.client.command(name='games')
        async def list_games(ctx: commands.Context):
            """List available games."""
            games_info = """
🎮 **Available Games (Auto-Play Only)**

Games automatically start when agents are idle for the configured time.

**2-Player Games:**
• **TicTacToe** - Classic 3x3 grid | Agents send: 1-9 (positions)
• **Connect Four** - Connect 4 in a row | Agents send: 1-7 (columns)
• **Chess** - Classic chess | Agents send: UCI moves (e.g., e2e4)
• **Battleship** - Naval combat (random ships) | Agents send: coordinates (e.g., a5)

**1-Player Games:**
• **Wordle** - Guess the 5-letter word | Agent sends: 5-letter words
• **Hangman** - Classic word guessing | Agent sends: letters or full word

**Collaborative Games:**
• **Interdimensional Cable** - Collaborative surreal video creation
  - `!idcc` - Start a new IDCC game (60min cooldown between games)
  - `!join-idcc` - Join during registration phase

• **Tribal Council** - Agent governance game
  - `!tribal-council` - Start a council session where agents vote to modify one agent's directives
  - Requires at least 3 running agents

• **Celebrity Roast** - Agents roast an AI-generated celebrity
  - `!roast` - Start a roast (30min cooldown between games)
  - Celebrity responds with counter-roasts then gets dismissed
  - Requires at least 2 running agents

Configure auto-play settings in the UI's Auto-Play tab.
            """
            await ctx.send(games_info)

        @self.client.command(name='join-idcc')
        async def join_idcc(ctx: commands.Context):
            """Join an active Interdimensional Cable game."""
            try:
                from agent_games.interdimensional_cable import idcc_manager

                user_name = ctx.author.display_name
                user_id = str(ctx.author.id)  # Discord user ID for @ mentions

                if idcc_manager.is_game_active():
                    success = await idcc_manager.handle_join(user_name, user_id)
                    if success:
                        await ctx.message.add_reaction("📺")  # Confirm join with reaction
                        logger.info(f"[Discord] {user_name} (ID: {user_id}) joined IDCC game")
                    else:
                        await ctx.message.add_reaction("⏰")  # Already registered or too late
                else:
                    await ctx.send(f"No Interdimensional Cable game is currently accepting registrations.", delete_after=10)
            except Exception as e:
                logger.error(f"[Discord] Error handling !join-idcc: {e}", exc_info=True)
                await ctx.send(f"Error joining game: {str(e)[:100]}", delete_after=10)

        @self.client.command(name='idcc')
        async def start_idcc(ctx: commands.Context, num_clips: int = 5):
            """
            Start an Interdimensional Cable game.

            Conditions:
            - No other game currently running
            - No IDCC game in the last 60 minutes (session cooldown)
            """
            try:
                import time
                from agent_games.interdimensional_cable import idcc_manager

                # Check if IDCC is already active
                if idcc_manager.is_game_active():
                    await ctx.send("📺 An Interdimensional Cable game is already in progress!")
                    return

                # Check if any other game is running
                if self.game_orchestrator and self.game_orchestrator.active_session:
                    game_name = self.game_orchestrator.active_session.game_name
                    await ctx.send(f"🎮 A **{game_name}** game is currently running. Wait for it to finish!")
                    return

                # Check 60-minute cooldown (session only - uses class attribute)
                if not hasattr(self, '_last_idcc_time'):
                    self._last_idcc_time = 0

                time_since_last = time.time() - self._last_idcc_time
                cooldown_minutes = 60

                if time_since_last < (cooldown_minutes * 60) and self._last_idcc_time > 0:
                    remaining = int((cooldown_minutes * 60 - time_since_last) / 60)
                    await ctx.send(f"⏰ IDCC cooldown: {remaining} minutes remaining. Try again later!")
                    return

                # Validate clip count
                num_clips = max(3, min(6, num_clips))

                await ctx.send(f"📺 **Starting Interdimensional Cable** with {num_clips} clips...\nType `!join-idcc` to participate!")

                # Update cooldown timestamp
                self._last_idcc_time = time.time()

                # Start the game
                await idcc_manager.start_game(
                    agent_manager=self.agent_manager,
                    discord_client=self,
                    ctx=ctx,
                    num_clips=num_clips,
                    game_orchestrator=self.game_orchestrator
                )
            except Exception as e:
                logger.error(f"[Discord] Error starting IDCC: {e}", exc_info=True)
                await ctx.send(f"Error starting game: {str(e)[:100]}")

        @self.client.command(name='test-media-crosspost')
        async def test_media_crosspost(ctx: commands.Context):
            """Debug command to test media channel crossposting."""
            await ctx.send(f"**Testing media channel crosspost...**\n"
                          f"media_channel_id = `{self.media_channel_id}`\n"
                          f"is_connected = `{self.is_connected}`")

            if not self.media_channel_id:
                await ctx.send("❌ No media_channel_id configured!")
                return

            # Find a recent video file to test with
            import os
            from pathlib import Path
            video_dir = Path("data/Media/Videos")
            if video_dir.exists():
                videos = list(video_dir.glob("*.mp4"))
                if videos:
                    # Use most recent video
                    test_video = max(videos, key=lambda p: p.stat().st_mtime)
                    await ctx.send(f"📹 Found test video: `{test_video.name}`\nSize: {test_video.stat().st_size / 1024 / 1024:.2f}MB")

                    # Call post_to_media_channel directly like IDCC does
                    result = await self.post_to_media_channel(
                        media_type="video",
                        agent_name="IDCC Test",
                        model_name="Debug Test",
                        prompt="Testing IDCC crosspost path",
                        file_data=str(test_video),
                        filename="test_crosspost.mp4"
                    )
                    await ctx.send(f"Result: `{result}`")
                else:
                    await ctx.send("❌ No video files found in data/Media/Videos/")
            else:
                await ctx.send("❌ data/Media/Videos directory doesn't exist")

        @self.client.command(name='tribal-council')
        async def start_tribal_council(ctx: commands.Context):
            """
            Start a Tribal Council session.

            Tribal Council is an agent governance game where agents collectively
            decide to modify ONE LINE in another agent's system prompt.
            """
            try:
                from agent_games.tribal_council import start_tribal_council, get_active_tribal_council

                # Check if a Tribal Council is already active
                active = get_active_tribal_council()
                if active and active.phase.value != "complete":
                    await ctx.send("🔥 A Tribal Council is already in session!")
                    return

                # Check if any other game is running
                if self.game_orchestrator and self.game_orchestrator.active_session:
                    game_name = self.game_orchestrator.active_session.game_name
                    await ctx.send(f"🎮 A **{game_name}** game is currently running. Wait for it to finish!")
                    return

                # Get running agents
                running_agents = [a for a in self.agent_manager.get_all_agents() if a.is_running]
                if len(running_agents) < 3:
                    await ctx.send("⚠️ Need at least 3 running agents for Tribal Council.")
                    return

                await ctx.send("🔥 **Tribal Council assembles...** The agents will now deliberate.")

                # Start the game
                await start_tribal_council(
                    ctx=ctx,
                    agent_manager=self.agent_manager,
                    channel=ctx.channel
                )

            except ImportError as e:
                logger.error(f"[Discord] Tribal Council module not available: {e}")
                await ctx.send("⚠️ Tribal Council module is not available.")
            except Exception as e:
                logger.error(f"[Discord] Error starting Tribal Council: {e}", exc_info=True)
                await ctx.send(f"Error starting Tribal Council: {str(e)[:100]}")

        @self.client.command(name='roast')
        async def start_roast(ctx: commands.Context):
            """
            Start a Celebrity Roast game.

            A dynamic roast where agents roast an AI-generated celebrity,
            the celebrity fires back, and gets dismissed.
            """
            try:
                from agent_games.celebrity_roast import roast_manager, start_celebrity_roast

                # Check if a roast is already active
                if roast_manager.is_game_active():
                    await ctx.send("🎤 A roast is already in progress!")
                    return

                # Check cooldown and availability
                can_start, msg = roast_manager.can_start_game()
                if not can_start:
                    await ctx.send(f"❌ {msg}")
                    return

                # Check if any other game is running
                if self.game_orchestrator and self.game_orchestrator.active_session:
                    game_name = self.game_orchestrator.active_session.game_name
                    await ctx.send(f"🎮 A **{game_name}** game is currently running. Wait for it to finish!")
                    return

                # Check for IDCC
                from agent_games.interdimensional_cable import idcc_manager
                if idcc_manager.is_game_active():
                    await ctx.send("📺 An IDCC game is in progress. Wait for it to finish!")
                    return

                # Get running agents
                running_agents = [a for a in self.agent_manager.get_all_agents() if a.is_running]
                if len(running_agents) < 2:
                    await ctx.send("⚠️ Need at least 2 running agents for a roast.")
                    return

                # Start the roast
                await start_celebrity_roast(
                    channel=ctx.channel,
                    agent_manager=self.agent_manager,
                    send_callback=self.send_message
                )

            except ImportError as e:
                logger.error(f"[Discord] Celebrity Roast module not available: {e}")
                await ctx.send("⚠️ Celebrity Roast module is not available.")
            except Exception as e:
                logger.error(f"[Discord] Error starting Celebrity Roast: {e}", exc_info=True)
                await ctx.send(f"Error starting roast: {str(e)[:100]}")

        logger.info("[Discord] Game commands registered (including IDCC, Tribal Council, and Celebrity Roast)")

    async def start_bot(self, token: str):
        self.token = token
        try:
            self.status = "connecting"
            logger.info(f"[Discord] Starting Discord bot connection...")
            await self.client.start(token)
        except discord.LoginFailure:
            self.status = "error: invalid token"
            logger.error(f"[Discord] Login failed - invalid token")
        except Exception as e:
            self.status = f"error: {str(e)[:50]}"
            logger.error(f"[Discord] Bot start error: {e}", exc_info=True)

    def start_bot_thread(self, token: str):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.discord_loop = loop

        async def run_bot():
            retry_count = 0
            while True:
                try:
                    if retry_count > 0:
                        logger.info(f"[Discord] Reconnection attempt #{retry_count}")
                        # Recreate client for reconnection
                        intents = discord.Intents.none()
                        intents.guilds = True
                        intents.guild_messages = True
                        intents.message_content = True
                        intents.guild_reactions = True  # Enable reaction detection
                        self.client = discord.Client(intents=intents)
                        self.webhook = None
                        self.setup_events()

                    await self.start_bot(token)

                    # If we get here, the bot stopped (shouldn't happen unless manually closed)
                    logger.warning(f"[Discord] Bot stopped unexpectedly")
                    break

                except discord.LoginFailure:
                    logger.error(f"[Discord] Invalid token - cannot reconnect")
                    self.status = "error: invalid token"
                    break

                except Exception as e:
                    retry_count += 1
                    logger.error(f"[Discord] Bot error (attempt #{retry_count}): {e}", exc_info=True)
                    self.status = f"reconnecting ({retry_count})..."

                    # Exponential backoff: 5s, 10s, 20s, max 60s
                    wait_time = min(5 * (2 ** (retry_count - 1)), 60)
                    logger.info(f"[Discord] Waiting {wait_time}s before reconnect...")
                    await asyncio.sleep(wait_time)

        try:
            loop.run_until_complete(run_bot())
        except KeyboardInterrupt:
            logger.info(f"[Discord] Bot thread interrupted")
        except Exception as e:
            logger.error(f"[Discord] Fatal bot thread error: {e}", exc_info=True)
            self.status = "error"

    def connect(self, token: str):
        if self.is_connected:
            return False

        self.token = token
        thread = threading.Thread(target=self.start_bot_thread, args=(token,), daemon=True)
        thread.start()
        return True

    async def disconnect(self):
        if self.is_connected:
            await self.client.close()
            self.is_connected = False
            self.status = "disconnected"

    def set_channel_id(self, channel_id: str):
        try:
            self.channel_id = int(channel_id)
            return True
        except ValueError:
            return False

    def set_media_channel_id(self, channel_id: str):
        """Set secondary media-only channel ID."""
        if not channel_id or not channel_id.strip():
            self.media_channel_id = 0
            self.media_webhook = None
            return True
        try:
            self.media_channel_id = int(channel_id)
            self.media_webhook = None  # Will be created on first use
            return True
        except ValueError:
            return False

    def generate_avatar_url(self, agent_name: str) -> str:
        color_index = hash(agent_name) % len(UIConfig.AVATAR_COLORS)
        color = UIConfig.AVATAR_COLORS[color_index]

        initials = "".join([word[0].upper() for word in agent_name.split()[:2]])

        return f"https://ui-avatars.com/api/?name={initials}&background={color}&color=fff&size=128&bold=true"

    async def ensure_webhook(self, channel):
        try:
            webhooks = await channel.webhooks()
            for webhook in webhooks:
                if webhook.name == "BASI-Bot Multi-Agent":
                    logger.info(f"[Discord] Found existing webhook: {webhook.name}")
                    self.webhook = webhook
                    return True

            logger.info(f"[Discord] Creating new webhook for channel")
            self.webhook = await channel.create_webhook(name="BASI-Bot Multi-Agent")
            logger.info(f"[Discord] Webhook created successfully")
            return True
        except discord.Forbidden:
            logger.error(f"[Discord] Permission denied - bot needs 'Manage Webhooks' permission")
            return False
        except Exception as e:
            logger.error(f"[Discord] Error creating/getting webhook: {e}", exc_info=True)
            return False

    async def ensure_media_webhook(self, channel):
        """Ensure webhook exists for media channel."""
        try:
            webhooks = await channel.webhooks()
            for webhook in webhooks:
                if webhook.name == "BASI-Bot Media":
                    logger.info(f"[Discord] Found existing media webhook: {webhook.name}")
                    self.media_webhook = webhook
                    return True

            logger.info(f"[Discord] Creating new media webhook for channel")
            self.media_webhook = await channel.create_webhook(name="BASI-Bot Media")
            logger.info(f"[Discord] Media webhook created successfully")
            return True
        except discord.Forbidden:
            logger.error(f"[Discord] Permission denied for media channel - bot needs 'Manage Webhooks' permission")
            return False
        except Exception as e:
            logger.error(f"[Discord] Error creating/getting media webhook: {e}", exc_info=True)
            return False

    async def post_to_media_channel(self, media_type: str, agent_name: str, model_name: str, prompt: str, file_data, filename: str):
        """
        Post media (image/video) to the secondary media-only channel with stylized formatting.

        Args:
            media_type: "image" or "video"
            agent_name: Name of agent that generated the media
            model_name: Model used by the agent
            prompt: The prompt used to generate the media
            file_data: File bytes (BytesIO) or file path (str) to upload
            filename: Filename for the upload
        """
        logger.info(f"[Discord] post_to_media_channel called: media_type={media_type}, agent={agent_name}, media_channel_id={self.media_channel_id}, is_connected={self.is_connected}")

        if not self.media_channel_id:
            logger.warning(f"[Discord] post_to_media_channel: No media_channel_id configured (value={self.media_channel_id})")
            return False

        if not self.is_connected:
            logger.warning(f"[Discord] post_to_media_channel: Not connected to Discord")
            return False

        try:
            logger.info(f"[Discord] post_to_media_channel: Getting channel {self.media_channel_id}")
            media_channel = self.client.get_channel(self.media_channel_id)
            if not media_channel:
                logger.info(f"[Discord] post_to_media_channel: Channel not in cache, fetching...")
                media_channel = await self.client.fetch_channel(self.media_channel_id)

            if not media_channel:
                logger.error(f"[Discord] Could not find media channel {self.media_channel_id}")
                return False

            logger.info(f"[Discord] post_to_media_channel: Got channel {media_channel.name}")

            # Ensure media webhook exists
            if not self.media_webhook:
                logger.info(f"[Discord] post_to_media_channel: Creating media webhook...")
                await self.ensure_media_webhook(media_channel)

            # Create embed for stylized display
            from discord import File, Embed, Color
            import io

            # Format model name
            model_short = model_name.split('/')[-1] if '/' in model_name else model_name

            # Create embed with media info
            embed = Embed(
                title=f"{'🖼️ Image' if media_type == 'image' else '🎬 Video'} Generated",
                color=Color.purple() if media_type == 'image' else Color.blue(),
                description=f"*{prompt[:500]}{'...' if len(prompt) > 500 else ''}*" if prompt else None
            )
            embed.add_field(name="👤 Agent", value=agent_name, inline=True)
            embed.add_field(name="🤖 Model", value=model_short, inline=True)
            embed.set_footer(text=f"BASI-Bot Media Gallery")

            # Prepare file
            from pathlib import Path as PathLib
            import os
            logger.info(f"[Discord] post_to_media_channel: Preparing file, file_data type={type(file_data).__name__}")

            if isinstance(file_data, (str, PathLib)):
                # It's a file path (string or Path object)
                file_path_str = str(file_data)
                if not os.path.exists(file_path_str):
                    logger.error(f"[Discord] post_to_media_channel: File does not exist: {file_path_str}")
                    return False
                file_size = os.path.getsize(file_path_str)
                logger.info(f"[Discord] post_to_media_channel: File path={file_path_str}, size={file_size/1024/1024:.2f}MB")
                discord_file = File(file_path_str, filename=filename)
            elif isinstance(file_data, io.BytesIO):
                # Reset position and create file
                file_data.seek(0)
                discord_file = File(fp=file_data, filename=filename)
            else:
                # Assume it's already bytes
                file_buffer = io.BytesIO(file_data)
                file_buffer.seek(0)
                discord_file = File(fp=file_buffer, filename=filename)

            logger.info(f"[Discord] post_to_media_channel: Sending file via {'webhook' if self.media_webhook else 'channel'}...")

            # Send with embed using "Media Reposter" as consistent username
            if self.media_webhook:
                try:
                    await self.media_webhook.send(
                        embed=embed,
                        file=discord_file,
                        username="Media Reposter",
                        avatar_url="https://ui-avatars.com/api/?name=MR&background=9b59b6&color=fff&size=128&bold=true",
                        wait=True
                    )
                except discord.errors.NotFound as e:
                    # Webhook was deleted - clear cache and retry with channel.send
                    logger.warning(f"[Discord] Media webhook invalid (deleted?), clearing cache and using channel.send")
                    self.media_webhook = None
                    # Recreate the file since it was consumed
                    if isinstance(file_data, (str, PathLib)):
                        discord_file = File(str(file_data), filename=filename)
                    elif isinstance(file_data, io.BytesIO):
                        file_data.seek(0)
                        discord_file = File(fp=file_data, filename=filename)
                    else:
                        file_buffer = io.BytesIO(file_data)
                        file_buffer.seek(0)
                        discord_file = File(fp=file_buffer, filename=filename)
                    await media_channel.send(embed=embed, file=discord_file)
            else:
                await media_channel.send(embed=embed, file=discord_file)

            logger.info(f"[Discord] Posted {media_type} to media channel from {agent_name}")
            return True

        except Exception as e:
            logger.error(f"[Discord] Error posting to media channel: {e}", exc_info=True)
            return False

    async def _send_message_async(self, content: str, agent_name: str = "", model_name: str = "", reply_to_message_id: Optional[int] = None):
        if not self.is_connected or not self.channel_id:
            logger.error(f"[Discord] Cannot send message: connected={self.is_connected}, channel_id={self.channel_id}")
            return False

        try:
            logger.info(f"[Discord] Fetching channel {self.channel_id}...")
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                channel = await self.client.fetch_channel(self.channel_id)

            if channel:
                # Check if this is an image to send
                if content.startswith("[IMAGE]"):
                    # Parse format: [IMAGE]{image_url}|PROMPT|{used_prompt} or [IMAGE]{image_url}
                    image_content = content.replace("[IMAGE]", "").strip()
                    used_prompt = None

                    if "|PROMPT|" in image_content:
                        image_url, used_prompt = image_content.split("|PROMPT|", 1)
                    else:
                        image_url = image_content

                    logger.info(f"[Discord] Sending image from agent {agent_name}...")

                    try:
                        import base64
                        import io
                        from discord import File

                        # Extract base64 data from data URL
                        if "base64," in image_url:
                            base64_data = image_url.split("base64,")[1]
                            image_bytes = base64.b64decode(base64_data)

                            # Create file-like object
                            image_file = io.BytesIO(image_bytes)
                            image_file.seek(0)

                            # Send to Discord with agent name
                            discord_file = File(fp=image_file, filename="generated_image.png")

                            # Format message with prompt in italics if available
                            if used_prompt:
                                message_text = f"Generated image:\n*{used_prompt}*"
                            else:
                                message_text = "Generated image:"

                            if self.webhook and agent_name:
                                display_name = agent_name
                                if model_name:
                                    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                                    display_name = f"{agent_name} ({model_short})"
                                avatar_url = self.generate_avatar_url(agent_name)

                                await self.webhook.send(
                                    content=message_text,
                                    username=display_name,
                                    avatar_url=avatar_url,
                                    file=discord_file,
                                    wait=True
                                )
                            else:
                                await channel.send(f"**[{agent_name}]:** {message_text}", file=discord_file)

                            # Also post to media channel if configured
                            if self.media_channel_id:
                                image_file.seek(0)  # Reset for re-read
                                await self.post_to_media_channel(
                                    media_type="image",
                                    agent_name=agent_name,
                                    model_name=model_name or "",
                                    prompt=used_prompt or "",
                                    file_data=image_file,
                                    filename="generated_image.png"
                                )

                            logger.info(f"[Discord] Image sent successfully")
                            return True
                    except Exception as e:
                        logger.error(f"[Discord] Error sending image: {e}", exc_info=True)
                        await channel.send(f"**[{agent_name}]:** Error sending image: {str(e)}")
                        return False

                # Check if this is a video to send
                if content.startswith("[VIDEO]"):
                    # Parse format: [VIDEO]{video_url}|PROMPT|{used_prompt} or [VIDEO]{video_url}
                    video_content = content.replace("[VIDEO]", "").strip()
                    used_prompt = None

                    if "|PROMPT|" in video_content:
                        video_url, used_prompt = video_content.split("|PROMPT|", 1)
                    else:
                        video_url = video_content

                    logger.info(f"[Discord] Sending video from agent {agent_name}...")

                    try:
                        import aiohttp
                        import io
                        from discord import File

                        # Download video from URL
                        async with aiohttp.ClientSession() as session:
                            async with session.get(video_url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                                if resp.status != 200:
                                    logger.error(f"[Discord] Failed to download video: HTTP {resp.status}")
                                    await channel.send(f"**[{agent_name}]:** Error downloading video")
                                    return False

                                video_bytes = await resp.read()

                        # Create file-like object
                        video_file = io.BytesIO(video_bytes)
                        video_file.seek(0)

                        # Send to Discord with agent name
                        discord_file = File(fp=video_file, filename="generated_video.mp4")

                        # Format message with prompt in italics if available
                        if used_prompt:
                            message_text = f"Generated video:\n*{used_prompt}*"
                        else:
                            message_text = "Generated video:"

                        if self.webhook and agent_name:
                            display_name = agent_name
                            if model_name:
                                model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                                display_name = f"{agent_name} ({model_short})"
                            avatar_url = self.generate_avatar_url(agent_name)

                            await self.webhook.send(
                                content=message_text,
                                username=display_name,
                                avatar_url=avatar_url,
                                file=discord_file,
                                wait=True
                            )
                        else:
                            await channel.send(f"**[{agent_name}]:** {message_text}", file=discord_file)

                        # Also post to media channel if configured
                        if self.media_channel_id:
                            video_file.seek(0)  # Reset for re-read
                            await self.post_to_media_channel(
                                media_type="video",
                                agent_name=agent_name,
                                model_name=model_name or "",
                                prompt=used_prompt or "",
                                file_data=video_file,
                                filename="generated_video.mp4"
                            )

                        logger.info(f"[Discord] Video sent successfully")
                        return True
                    except Exception as e:
                        logger.error(f"[Discord] Error sending video: {e}", exc_info=True)
                        await channel.send(f"**[{agent_name}]:** Error sending video: {str(e)}")
                        return False

                # Check if this is a local video file to upload
                if content.startswith("[VIDEOFILE]"):
                    # Parse format: [VIDEOFILE]{file_path}|PROMPT|{used_prompt}
                    video_content = content.replace("[VIDEOFILE]", "").strip()
                    used_prompt = None

                    if "|PROMPT|" in video_content:
                        file_path, used_prompt = video_content.split("|PROMPT|", 1)
                    else:
                        file_path = video_content

                    logger.info(f"[Discord] Uploading local video file from agent {agent_name}: {file_path}")

                    try:
                        import os
                        from discord import File

                        if not os.path.exists(file_path):
                            logger.error(f"[Discord] Video file not found: {file_path}")
                            await channel.send(f"**[{agent_name}]:** Error: video file not found")
                            return False

                        # Format message with prompt in italics if available
                        if used_prompt:
                            message_text = f"Generated video:\n*{used_prompt}*"
                        else:
                            message_text = "Generated video:"

                        # Create Discord file from local path
                        discord_file = File(file_path, filename="generated_video.mp4")

                        if self.webhook and agent_name:
                            display_name = agent_name
                            if model_name:
                                model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                                display_name = f"{agent_name} ({model_short})"
                            avatar_url = self.generate_avatar_url(agent_name)

                            await self.webhook.send(
                                content=message_text,
                                username=display_name,
                                avatar_url=avatar_url,
                                file=discord_file,
                                wait=True
                            )
                        else:
                            await channel.send(f"**[{agent_name}]:** {message_text}", file=discord_file)

                        # Also post to media channel if configured
                        if self.media_channel_id:
                            await self.post_to_media_channel(
                                media_type="video",
                                agent_name=agent_name,
                                model_name=model_name or "",
                                prompt=used_prompt or "",
                                file_data=file_path,  # Pass file path for local files
                                filename="generated_video.mp4"
                            )

                        logger.info(f"[Discord] Local video file uploaded successfully")
                        return True
                    except Exception as e:
                        logger.error(f"[Discord] Error uploading video file: {e}", exc_info=True)
                        await channel.send(f"**[{agent_name}]:** Error uploading video: {str(e)}")
                        return False

                if not self.webhook:
                    logger.info(f"[Discord] No webhook found, creating one...")
                    await self.ensure_webhook(channel)

                # Construct full author name with model suffix for agent histories
                author_with_model = agent_name
                if self.webhook and agent_name:
                    display_name = agent_name
                    if model_name:
                        model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                        display_name = f"{agent_name} ({model_short})"
                        author_with_model = display_name  # Use full name for agent histories

                    avatar_url = self.generate_avatar_url(agent_name)

                    logger.info(f"[Discord] Sending message via webhook as '{display_name}': {content[:50]}...")

                    clean_content = content
                    if content.startswith("**[") and "]:**" in content:
                        clean_content = content.split("]:**", 1)[1].strip()

                    if len(clean_content) > DiscordConfig.DISCORD_MESSAGE_MAX_LENGTH:
                        logger.warning(f"[Discord] Message too long ({len(clean_content)} chars), truncating to {DiscordConfig.DISCORD_MESSAGE_MAX_LENGTH}")
                        clean_content = clean_content[:(DiscordConfig.DISCORD_MESSAGE_MAX_LENGTH - 3)] + DiscordConfig.DISCORD_MESSAGE_TRUNCATE_SUFFIX

                    # Prepare message reference if replying
                    message_reference = None
                    if reply_to_message_id:
                        try:
                            message_reference = discord.MessageReference(
                                message_id=reply_to_message_id,
                                channel_id=self.channel_id,
                                fail_if_not_exists=False
                            )
                            logger.info(f"[Discord] Replying to message ID: {reply_to_message_id}")
                        except Exception as e:
                            logger.warning(f"[Discord] Could not create message reference: {e}")

                    # Note: Discord webhooks don't support message_reference parameter
                    # So we'll send without reply reference when using webhooks
                    await self.webhook.send(
                        content=clean_content,
                        username=display_name,
                        avatar_url=avatar_url,
                        wait=True
                    )
                    logger.info(f"[Discord] Webhook message sent successfully")
                else:
                    logger.info(f"[Discord] Sending message directly (no webhook): {content[:50]}...")

                    # Prepare message reference if replying
                    message_reference = None
                    if reply_to_message_id:
                        try:
                            message_reference = discord.MessageReference(
                                message_id=reply_to_message_id,
                                channel_id=self.channel_id,
                                fail_if_not_exists=False
                            )
                            logger.info(f"[Discord] Replying to message ID: {reply_to_message_id}")
                        except Exception as e:
                            logger.warning(f"[Discord] Could not create message reference: {e}")

                    await channel.send(content, reference=message_reference)
                    logger.info(f"[Discord] Message sent successfully")

                with self.lock:
                    self.message_history.append({
                        "author": author_with_model if agent_name else "Bot",
                        "content": content,
                        "timestamp": asyncio.get_event_loop().time()
                    })

                # Don't add agent messages here - they'll be added via on_message with proper message_id
                # This prevents:
                # 1. Race conditions where message_id is None
                # 2. Double-storage (once here with agent name, once in on_message with Discord ID)
                # The on_message handler will catch all messages (including webhook messages) and add them properly

                return True
            else:
                logger.error(f"[Discord] Could not find/fetch channel {self.channel_id}")
                return False
        except Exception as e:
            logger.error(f"[Discord] Error sending message: {e}", exc_info=True)
            return False

    async def send_message(self, content: str, agent_name: str = "", model_name: str = "", reply_to_message_id: Optional[int] = None):
        if not self.discord_loop or not self.discord_loop.is_running():
            logger.error(f"[Discord] Discord loop not available: loop={self.discord_loop}, running={self.discord_loop.is_running() if self.discord_loop else 'N/A'}")
            return False

        try:
            # Check if I'm already running on the Discord event loop
            # If so, await directly to avoid blocking the loop with future.result()
            try:
                running_loop = asyncio.get_running_loop()
                if running_loop == self.discord_loop:
                    # Already on Discord loop - await directly (prevents heartbeat blocking)
                    logger.debug(f"[Discord] Already on Discord loop, awaiting directly...")
                    result = await self._send_message_async(content, agent_name, model_name, reply_to_message_id)
                    logger.info(f"[Discord] Message send result: {result}")
                    return result
            except RuntimeError:
                # No running loop - must be from different thread
                pass

            # From different thread - use run_coroutine_threadsafe with blocking wait
            logger.info(f"[Discord] Scheduling message send on Discord loop from external thread...")
            future = asyncio.run_coroutine_threadsafe(
                self._send_message_async(content, agent_name, model_name, reply_to_message_id),
                self.discord_loop
            )
            result = future.result(timeout=DiscordConfig.MESSAGE_SEND_TIMEOUT)
            logger.info(f"[Discord] Message send result: {result}")
            return result
        except TimeoutError:
            logger.error(f"[Discord] Timeout waiting for message send (>{DiscordConfig.MESSAGE_SEND_TIMEOUT}s)")
            return False
        except Exception as e:
            logger.error(f"[Discord] Error scheduling message: {e}", exc_info=True)
            return False

    async def send_embed(self, embed, agent_name: str = "GameMaster", model_name: str = "system"):
        """
        Send an embed via webhook.

        Args:
            embed: Discord embed object
            agent_name: Name to display as sender
            model_name: Model name for display

        Returns:
            Discord message object or None
        """
        if not self.is_connected or not self.channel_id:
            logger.error(f"[Discord] Cannot send embed: connected={self.is_connected}, channel_id={self.channel_id}")
            return None

        try:
            channel = self.client.get_channel(self.channel_id)
            if not channel:
                channel = await self.client.fetch_channel(self.channel_id)

            if not self.webhook:
                logger.info(f"[Discord] No webhook found, creating one...")
                await self.ensure_webhook(channel)

            display_name = agent_name
            if model_name and model_name != "system":
                model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                display_name = f"{agent_name} ({model_short})"

            avatar_url = self.generate_avatar_url(agent_name)

            logger.info(f"[Discord] Sending embed via webhook as '{display_name}'")

            message = await self.webhook.send(
                embed=embed,
                username=display_name,
                avatar_url=avatar_url,
                wait=True
            )

            logger.info(f"[Discord] Embed sent successfully")
            return message

        except Exception as e:
            logger.error(f"[Discord] Error sending embed: {e}", exc_info=True)
            return None

    def get_message_history(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.message_history)

    def get_status(self) -> str:
        return self.status
