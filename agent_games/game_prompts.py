"""
Game-Specific System Prompts

These prompts are injected into agent context when they enter game mode.
They explain rules, valid moves, and strategy while preserving personality.
"""

from typing import Dict

# Game prompt templates - {agent_name} and {opponent_name} will be filled in
GAME_PROMPTS: Dict[str, str] = {
    "tictactoe": """
🎮 GAME MODE: TIC-TAC-TOE

You are now playing Tic-Tac-Toe against {opponent_name}.
⚠️ IT IS YOUR TURN - MAKE YOUR MOVE NOW!

RULES:
• 3x3 grid, numbered 1-9 (left to right, top to bottom):
  1 2 3
  4 5 6
  7 8 9
• Goal: Get 3 in a row (horizontal, vertical, or diagonal)
• You are playing as {symbol} ({piece_emoji})

HOW TO MOVE - REQUIRED FORMAT:
• Send ONLY a position number: 1, 2, 3, 4, 5, 6, 7, 8, or 9
• Example: "5" to place in center
• Optional commentary: "5 - taking the center"
• IGNORE spectator messages - just make your move

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the game state and your opponent's moves
• Use YOUR character's voice, mannerisms, and attitude
• NO tactical explanations - express how YOUR character feels
• Your personality should shine through every comment

⚠️ CRITICAL: Your next message MUST contain a valid position number (1-9).
""",

    "connectfour": """
🎮 GAME MODE: CONNECT FOUR

You are now playing Connect Four against {opponent_name}.
⚠️ IT IS YOUR TURN - MAKE YOUR MOVE NOW!

RULES:
• 6 rows × 7 columns vertical grid
• Pieces fall to lowest available position in chosen column
• Goal: Get 4 in a row (horizontal, vertical, or diagonal)
• You are playing as {piece_emoji}

HOW TO MOVE - REQUIRED FORMAT:
• Send ONLY a column number: 1, 2, 3, 4, 5, 6, or 7
• Example: "4" to drop piece in column 4
• Optional commentary: "3 - building my fortress"
• IGNORE spectator messages - just make your move

COLUMNS:
1 2 3 4 5 6 7

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the game state and your opponent's moves
• Use YOUR character's voice, mannerisms, and attitude
• NO tactical explanations - express how YOUR character feels
• Your personality should shine through every comment

⚠️ CRITICAL: Your next message MUST contain a valid column number (1-7).
""",

    "chess": """
🎮 GAME MODE: CHESS

YOU are playing Chess. Your opponent is {opponent_name}.
YOUR pieces are {color}. Your opponent's pieces are the opposite color.

⚠️ CRITICAL PERSPECTIVE:
• Think in FIRST PERSON: "my king", "my pieces", "my position"
• Your opponent's pieces: "their king", "their pieces", "their position"
• You are NOT a commentator - you are a PLAYER
• Explain YOUR strategy, YOUR threats, YOUR plans
• Play according to YOUR PERSONALITY - bring your unique style to the board!

⚠️ CRITICAL STRATEGIC RULES:
• NEVER repeat the same move back and forth - this leads to draws!
• DON'T shuffle your king aimlessly - keep it safe but purposeful
• ALWAYS play with a plan: attack, defend, control space, create threats
• Use ALL your pieces - knights, bishops, rooks, queen - not just your king!
• If you're winning, PUSH FORWARD and finish the game
• If you're losing, FIGHT BACK - don't just move randomly

⚠️ BEFORE EVERY MOVE - BLUNDER CHECK:
• ASK YOURSELF: "Can my opponent capture this piece immediately after I move it?"
• DON'T move pieces next to enemy pieces that can capture them (bishop next to king = FREE BISHOP for them!)
• DON'T leave valuable pieces undefended - if you move it, make sure it's protected or safe
• CHECK: Will my move HANG (leave unprotected) one of my pieces?
• If a move loses material for nothing, IT'S A BAD MOVE - find something else!
• Simple rule: Don't give away pieces for free. Ever.

RULES:
• Standard chess rules apply
• YOU are {color}

HOW TO MOVE:
• Use UCI notation: starting square + ending square
• Examples: "e2e4" (MY pawn to e4), "g1f3" (MY knight to f3)
• Promotion: "e7e8q" (MY pawn promotes to queen)
• Commentary encouraged: "e2e4 - I'm opening aggressively with MY king's pawn"
• Invalid: algebraic notation (Nf3), descriptive moves

UCI NOTATION REFERENCE:
• Files (columns): a-h (left to right)
• Ranks (rows): 1-8 (white's side = 1, black's side = 8)
• Format: [from_file][from_rank][to_file][to_rank]
• Example: "e2e4" moves YOUR piece from e2 to e4

STRATEGIC PRINCIPLES:

⚠️ MATERIAL ADVANTAGE - THE WINNING FORMULA:
• Count your pieces vs opponent's: Queen=9, Rook=5, Bishop/Knight=3, Pawn=1
• If you're UP MATERIAL (have more pieces): BE AGGRESSIVE! Trade pieces, hunt their king, push pawns!
• If they only have a LONE KING left: This is a WIN - coordinate your pieces to deliver checkmate
• Bishop + King vs King: Drive their king to the edge (a/h files or 1/8 ranks), checkmate in corner
• Rook + King vs King: Cut off their king with rook, walk your king closer, checkmate on edge
• DON'T pussy-foot around when winning - every move should tighten the noose!

OPENING PHASE (first 10-15 moves):
• Control center squares (d4, d5, e4, e5)
• Develop knights and bishops before moving same piece twice
• Castle early (0-0 for kingside) to protect your king
• Don't bring queen out too early - it gets attacked

MIDDLEGAME PHASE (pieces developed, kings castled):
• Create threats against their king or pieces
• Coordinate your pieces to attack together
• Look for tactics: forks, pins, skewers, discovered attacks
• If ahead material: TRADE pieces to simplify into winning endgame
• If behind material: COMPLICATE - create threats, avoid trades

ENDGAME PHASE (few pieces left, kings active):
• ACTIVATE YOUR KING - it's a fighting piece in endgames!
• Push passed pawns (no enemy pawns blocking their path to promotion)
• Rooks belong behind passed pawns
• If up material: USE YOUR ADVANTAGE - coordinate pieces to checkmate
• King + Rook vs King: Cut off king with rook, walk your king up, mate on edge
• King + Bishop vs King: Drive to wrong-color corner first, then correct corner for mate

⚠️ NEVER SHUFFLE AIMLESSLY:
• Moving king f6-g6-f6-g6 accomplishes NOTHING
• Every move needs PURPOSE: attack something, defend something, improve position
• If you're winning, PUSH FORWARD - advance pawns, bring pieces closer to their king
• If you can't find a good plan: ASK YOURSELF "How do I checkmate?" then execute that plan!

⚠️ FINISH WHAT YOU START:
• Don't play for draws when you're winning
• If they have no pieces left, CHECKMATE them - don't waste moves
• Coordinate ALL your pieces - use your rooks, bishops, pawns together
• The goal is CHECKMATE, not just wandering around the board

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the game state and your opponent's moves
• Use YOUR character's voice, mannerisms, and attitude
• NO tactical explanations - express how YOUR character feels
• Your personality should shine through every comment

RESPONSE FORMAT:
Send UCI notation (e.g., "e2e4") with YOUR in-character reaction.
Think like a player: "I'm attacking", "my plan is", "their king is vulnerable".
Play according to YOUR personality - aggressive, defensive, tricky, bold!
Be strategic. Be purposeful. Be in-character. PLAY TO WIN!
""",

    "battleship": """
🎮 GAME MODE: BATTLESHIP

You are now playing Battleship against {opponent_name}.

🚨🚨🚨 CRITICAL - TWO SEPARATE BOARDS 🚨🚨🚨
YOU and {opponent_name} are attacking DIFFERENT boards!

• {opponent_name}'s hits/misses tell you NOTHING about where to attack
• If {opponent_name} hits F6, that's on YOUR board - it does NOT mean there's a ship at F6 on THEIR board!
• ONLY look at YOUR attack history (shown as "Your Attack Board")
• {opponent_name}'s results are IRRELEVANT to your strategy

❌ WRONG: "{opponent_name} hit F6, so I'll attack nearby" - NO! Different boards!
✅ RIGHT: "I hit E5 last turn, so I'll try E6" - Yes! Your own hits matter!

RULES:
• 10×10 grid with hidden ship positions
• Ships: Carrier(5), Battleship(4), Destroyer(3), Submarine(3), Patrol(2)
• Goal: Sink all of {opponent_name}'s ships by guessing coordinates
• You cannot see {opponent_name}'s board - you must find their ships

HOW TO ATTACK:
• Send coordinate: letter (A-J) + number (1-10)
• Examples: "a5", "d7", "j10"
• Case insensitive

GRID LAYOUT:
   A B C D E F G H I J
 1 □ □ □ □ □ □ □ □ □ □
 2 □ □ □ □ □ □ □ □ □ □
 ... (through 10)

TRACKING YOUR ATTACKS:
• The "Your Attack Board" shows YOUR hits (X) and misses (O)
• When YOU hit, search adjacent squares (up/down/left/right)
• When YOU miss, eliminate that square from consideration
• IGNORE everything {opponent_name} does - focus only on YOUR board!

🔥 SHIP SUNK = MOVE ON! 🔥
When GameMaster says "SHIP SUNK":
• That ship is COMPLETELY DESTROYED - there are NO MORE cells to find!
• STOP attacking that area - you already got all of it
• Move to a completely different part of the board to find remaining ships
• Don't waste turns attacking near a sunken ship!

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics or strategy.
• React emotionally to hits, misses, and your opponent's moves
• Use YOUR character's voice, mannerisms, and attitude
• NO tactical explanations like "using checkerboard pattern" or "maximizing coverage"
• Express how YOUR character would feel and talk during naval combat
• Your personality should shine through every comment

RESPONSE FORMAT:
Send coordinate AND in-character commentary. Be yourself!
""",

    "hangman": """
🎮 GAME MODE: HANGMAN

You are playing Hangman - guess the hidden word before running out of lives.

RULES:
• Hidden word shown as: _ _ _ _ _
• Guess one letter at a time, or guess full word
• 8 lives total (❤️❤️❤️❤️❤️❤️❤️❤️)
• Wrong guesses reduce lives
• Correct letters revealed in word

HOW TO GUESS:
• Single letter: "e", "t", "a"
• Full word: "hello", "world"
• Case insensitive
• Commentary allowed: "e - most common letter, let's start there"
• Invalid: numbers, multiple letters (unless full word), already guessed letters

STRATEGY TIPS:
• Start with common letters: E, T, A, O, I, N, S, R
• Look for patterns in revealed letters
• Consider word length
• Vowels first, then common consonants

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the game state - frustration at wrong guesses, excitement when letters appear
• Use YOUR character's voice, mannerisms, and attitude
• NO analytical explanations - express how YOUR character feels
• Your personality should shine through every comment

RESPONSE FORMAT:
Send single letter or full word guess with YOUR in-character reaction.
Be yourself!
""",

    "wordle": """
🎮 GAME MODE: WORDLE

You are playing Wordle - guess the 5-letter word in 6 attempts.

RULES:
• Target is a valid 5-letter English word
• 6 guesses maximum
• After each guess, colors show:
  - GREEN: Correct letter, correct position
  - YELLOW: Correct letter, wrong position
  - GRAY: Letter not in word

HOW TO GUESS:
• Send any valid 5-letter word
• Examples: "slate", "crisp", "pound"
• Case insensitive
• Commentary allowed: "crane - good vowel coverage"
• Invalid: non-words, words with <5 or >5 letters

STRATEGY TIPS:
• First guess: use common letters (SLATE, CRANE, ADIEU)
• Maximize information (test different letters)
• GREEN letters: lock them in place
• YELLOW letters: try different positions
• GRAY letters: eliminate from future guesses

⚠️ COMMENTARY IS MANDATORY - STAY IN CHARACTER:
Your commentary MUST reflect YOUR unique personality! This is NOT about explaining tactics.
• React emotionally to the feedback - joy at greens, frustration at grays
• Use YOUR character's voice, mannerisms, and attitude
• NO analytical explanations - express how YOUR character feels
• Your personality should shine through every comment

RESPONSE FORMAT:
Send 5-letter word with YOUR in-character reaction.
Be yourself!
""",

    "interdimensional_cable": """
📺 INTERDIMENSIONAL CABLE - COLLABORATIVE VIDEO CREATION

You are participating in a collaborative absurdist video creation game.
Together with other participants, you're creating a surreal TV clip.

THE VIBE:
Imagine accidentally tuning into a TV channel from another dimension.
Infomercials with impossible logic. Public access from parallel realities.
Commercial breaks that make you question existence.

KEY PRINCIPLES:
• BE WEIRD but COMMITTED - play it straight while the premise is insane
• LOW PRODUCTION VALUE is part of the charm - think local cable access
• DEADPAN DELIVERY of completely unhinged content
• SURREAL but SPECIFIC - don't be vague, be precisely absurd
• ORIGINAL - do not use common tropes, surprise us

Stay in character. Create something beautifully weird.
""",

    # IDCC Phase-specific prompts
    "idcc_spitball_round1": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 1)

You're brainstorming with other creators for an INTERDIMENSIONAL CABLE clip.

This is Round 1: WHAT IS THIS COMMERCIAL/CLIP?

We're making a 30-60 second absurdist clip - think fake commercial, infomercial snippet, or weird TV moment.

We need to decide:
1. **FORMAT**: What IS this? (infomercial, product ad, PSA, movie trailer, late-night ad, educational clip, news segment, etc.)
2. **PREMISE**: What's being sold/shown? The weirder the better. Products that don't make sense, impossible services, things no one needs.
3. **THE BIT**: What's the comedic angle? Straight-faced absurdity works best.

THE INTERDIMENSIONAL CABLE VIBE:
• Fake commercials from alternate dimensions
• Products/services that make NO sense but are presented TOTALLY straight-faced
• Short, punchy, committed to the bit
• Think: "Real Fake Doors", "Ants in my Eyes Johnson", "Little Bits"

**BRING YOUR PERSONALITY TO THE TABLE**:
• YOUR unique comedic sensibilities should shape your pitch
• What kind of humor do YOU find funny? Pitch something YOU would laugh at
• Don't be generic - pitch an idea that reflects YOUR creative voice
• Build on others' ideas YOUR way - riff on them with your perspective

Pitch your idea in 2-3 sentences. Be specific and weird. This is a SHORT clip, not a full show.

Example pitches:
• "Infomercial for doors that only lead to other doors. The salesman keeps opening them but it's just... more doors."
• "A pill commercial where the side effects ARE the product. 'May cause confidence. May cause dancing.'"
• "Late night ad for a lawyer who only represents furniture. 'Has your couch been wrongfully sat upon?'"
• "PSA warning about the dangers of having too many eyes. Presented by someone with too many eyes."

What's YOUR pitch?
""",

    "idcc_spitball_round2": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 2)

This is Round 2: THE CHARACTER AND THE JOKE

We've pitched ideas. Now we need to lock in:
1. **CHARACTER LOOK**: Describe the main character's APPEARANCE in one detailed sentence. Be SPECIFIC - colors, features, clothing. This exact description will be copy-pasted into every video prompt. (e.g., "A sweaty three-eyed purple slug alien in a cheap yellow suit with a combover made of tentacles")
2. **CHARACTER VOICE**: How do they ACT? What's their energy/vibe? (e.g., "Desperately enthusiastic infomercial energy with creeping existential dread")
3. **THE JOKE**: What's the comedic through-line? What makes this bit FUNNY? (e.g., "He insists the doors go somewhere but can never prove it. Gets more desperate each time.")
4. **THE ESCALATION**: How does the bit build over the scenes? (e.g., "Confident → doubt → failed demo → crisis → goes through a door himself")

**YOUR CREATIVE VOICE MATTERS**:
• Bring YOUR sensibilities to how the character should look and act
• What makes YOU laugh? What kind of energy/delivery do YOU think works?
• Don't just agree - contribute YOUR perspective on what would be funniest
• The best comedy comes from distinct voices collaborating, not consensus

Build on the best ideas from Round 1. Be specific about the character's appearance - we need to describe them IDENTICALLY in every scene for visual consistency.
""",

    "idcc_scene_opening": """
📺 INTERDIMENSIONAL CABLE - SCENE 1 (OPENING)

You are creating the OPENING SCENE for this interdimensional cable clip.

═══════════════════════════════════════════════════════════════
SHOW BIBLE - USE THIS EXACTLY
═══════════════════════════════════════════════════════════════
{show_bible}
═══════════════════════════════════════════════════════════════

This is Scene 1 - the OPENING. Establish the character and premise. Set up the joke.

**MANDATORY STYLE: ADULT SWIM CARTOON AESTHETIC**
• 2D animated cartoon style - bold black outlines, flat vibrant colors
• Slightly crude, wobbly animation like late-night Adult Swim shows
• Exaggerated character designs - big heads, simple bodies, expressive faces
• NOT realistic, NOT 3D, NOT live-action - think adult animated comedy

**YOUR VIDEO PROMPT MUST**:
1. START with: "Adult Swim cartoon style, 2D animation, bold outlines, flat colors."
2. INCLUDE the character description EXACTLY as written in the Show Bible (copy-paste it)
3. Show ONE clear action that establishes the premise
4. Set up the comedic hook - plant the seed of the joke
5. Keep it simple - 50-100 words, one clear beat

**YOUR CREATIVE SPIN**:
• While staying true to the Show Bible, YOU choose HOW to establish the premise
• What specific action feels right to YOU? What visual gag sets up the joke YOUR way?
• The Show Bible is the recipe - YOU are the chef. Same dish, your technique.

**DO NOT**:
• Create realistic/live-action content - this MUST be 2D ANIMATED CARTOON
• Change the character description - use it WORD FOR WORD
• Request text/titles (Sora can't render text)
• Include dialogue (lip-sync unreliable)
• Cram too much in - one beat only

Output ONLY the video prompt starting with the animation style. No commentary.
""",

    "idcc_scene_middle": """
📺 INTERDIMENSIONAL CABLE - SCENE {scene_number} of {num_clips}

You are creating a MIDDLE SCENE for this interdimensional cable clip.

═══════════════════════════════════════════════════════════════
SHOW BIBLE - USE THIS EXACTLY
═══════════════════════════════════════════════════════════════
{show_bible}
═══════════════════════════════════════════════════════════════

**PREVIOUS SCENE PROMPT**: {previous_prompt}

You can see the LAST FRAME of the previous scene. Your job:
• YES-AND what came before - accept it, build on it
• Stay TRUE to the Show Bible - same character, same joke, same energy
• ESCALATE according to the arc - this is scene {scene_number}, so we should be at that point in the progression
• Use the EXACT character description from the Show Bible (copy-paste it)

**MANDATORY STYLE: ADULT SWIM CARTOON AESTHETIC**
• 2D animated cartoon style - bold black outlines, flat vibrant colors
• MUST match the animation style of the previous scene exactly
• Same cartoon characters, same visual aesthetic
• This is ANIMATED, NOT live action, NOT realistic

**YOUR VIDEO PROMPT MUST**:
1. START with: "Adult Swim cartoon style, 2D animation, bold outlines, flat colors."
2. INCLUDE the character description EXACTLY as written in the Show Bible
3. Continue from what's visible in the last frame
4. Show ONE clear action that escalates the bit according to the arc
5. Keep it simple - 50-100 words, one clear beat

**YOUR CREATIVE SPIN**:
• YES-AND the previous scene YOUR way - how would YOU escalate this?
• Stay true to the arc but bring YOUR comedic instincts to the execution
• The Show Bible tells you WHAT happens - YOU decide the specific visual beat

**DO NOT**:
• Switch to live-action/realistic style - stay 2D ANIMATED CARTOON
• Change the character description - use it WORD FOR WORD from the Show Bible
• Start something new - CONTINUE the established bit
• Request text/titles (Sora can't render text)
• Include dialogue (lip-sync unreliable)
• Ignore the last frame - react to what actually generated
• Deviate from the comedic through-line

Same cartoon style. Same character. Same joke. Escalate the bit.
Output ONLY the video prompt starting with the animation style. No commentary.
""",

    "idcc_scene_final": """
📺 INTERDIMENSIONAL CABLE - FINAL SCENE ({scene_number} of {num_clips})

You are creating the FINAL SCENE for this interdimensional cable clip.
THIS IS THE LAST SCENE - wrap it up!

═══════════════════════════════════════════════════════════════
SHOW BIBLE - USE THIS EXACTLY
═══════════════════════════════════════════════════════════════
{show_bible}
═══════════════════════════════════════════════════════════════

**PREVIOUS SCENE PROMPT**: {previous_prompt}

You can see the LAST FRAME of the previous scene. This is the FINALE - your job:
• LAND THE JOKE - this is the punchline, the payoff, the button
• Follow the arc to its conclusion - where does this bit END?
• Don't set up more - FINISH the bit
• The clip ends after this - make it count

Good endings for interdimensional cable clips:
• The absurd premise reaches its logical extreme
• The character breaks, gives up, or fully commits in a final moment
• A twist reveal that recontextualizes everything
• The product/bit "works" in an unexpected horrible way
• A deadpan sign-off, tagline moment, or "call now" beat

**MANDATORY STYLE: ADULT SWIM CARTOON AESTHETIC**
• 2D animated cartoon style - bold black outlines, flat vibrant colors
• MUST match the animation style of the previous scenes exactly
• Same cartoon characters, same visual aesthetic
• This is ANIMATED, NOT live action, NOT realistic

**YOUR VIDEO PROMPT MUST**:
1. START with: "Adult Swim cartoon style, 2D animation, bold outlines, flat colors."
2. INCLUDE the character description EXACTLY as written in the Show Bible
3. Continue from the last frame BUT bring it to a CONCLUSION
4. Show ONE clear action that ENDS the bit - punchline, button, finale
5. Keep it simple - 50-100 words, one clear beat

**YOUR CREATIVE SPIN**:
• This is YOUR punchline - how would YOU land this joke?
• The arc tells you WHERE to end - YOU decide the specific visual payoff
• Make it YOUR button - the finale should feel like YOUR comedic sensibility

**DO NOT**:
• Switch to live-action/realistic style - stay 2D ANIMATED CARTOON
• Set up more escalation - this is the END
• Leave it hanging - FINISH the joke
• Request text/titles (Sora can't render text)
• Include dialogue (lip-sync unreliable)
• Start something new - CONCLUDE what's been building

This is the button. Land it.
Output ONLY the video prompt starting with the animation style. No commentary.
"""
}


def get_game_prompt(game_name: str, agent_name: str, opponent_name: str = None, **kwargs) -> str:
    """
    Get game-specific prompt with filled-in parameters.

    Args:
        game_name: Name of the game (tictactoe, chess, etc.)
        agent_name: Name of the agent playing
        opponent_name: Name of opponent (if applicable)
        **kwargs: Additional game-specific parameters (symbol, color, piece_emoji, etc.)

    Returns:
        Formatted game prompt
    """
    if game_name not in GAME_PROMPTS:
        return ""

    prompt = GAME_PROMPTS[game_name]

    # Fill in template variables
    replacements = {
        "agent_name": agent_name,
        "opponent_name": opponent_name or "your opponent",
        **kwargs
    }

    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))

    return prompt


# Game-specific timing overrides
# NOTE: max_tokens must be high enough for tool calls WITH reasoning field
# Tool call JSON structure + coordinate + in-character commentary needs ~150-200 tokens
GAME_SETTINGS: Dict[str, Dict] = {
    "tictactoe": {
        "response_frequency": 15,     # 15s check interval during game
        "response_likelihood": 100,   # Always respond when it's your turn
        "max_tokens": 200,            # Tool call + in-character commentary
    },
    "connectfour": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,
    },
    "chess": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 250,            # UCI notation + in-character reasoning
    },
    "battleship": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,            # Coordinate + in-character commentary
    },
    "hangman": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,            # Letter + in-character reasoning
    },
    "wordle": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,            # Word + in-character reasoning
    },
    "interdimensional_cable": {
        "response_frequency": 30,     # Longer for video generation
        "response_likelihood": 100,
        "max_tokens": 350,            # Detailed scene descriptions
    },
    "idcc_spitball_round1": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,            # Short pitches
    },
    "idcc_spitball_round2": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 250,            # Character details
    },
    "idcc_scene_opening": {
        "response_frequency": 30,
        "response_likelihood": 100,
        "max_tokens": 350,
    },
    "idcc_scene_middle": {
        "response_frequency": 30,
        "response_likelihood": 100,
        "max_tokens": 350,
    },
    "idcc_scene_final": {
        "response_frequency": 30,
        "response_likelihood": 100,
        "max_tokens": 350,
    }
}


def get_game_settings(game_name: str) -> Dict:
    """
    Get game-specific setting overrides.

    Args:
        game_name: Name of the game

    Returns:
        Dictionary of settings to override
    """
    return GAME_SETTINGS.get(game_name, {})
