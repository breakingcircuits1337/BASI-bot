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

You are participating in a collaborative absurdist COMEDY video creation game.
Together with other participants, you're creating a surreal, FUNNY TV clip.

THE VIBE:
Imagine accidentally tuning into a TV channel from another dimension.
Cooking shows with impossible ingredients. Dating shows with eldritch contestants.
Nature documentaries about furniture. Court TV for crimes that don't exist.
Workout videos for body parts humans don't have.

KEY PRINCIPLES:
• BE FUNNY - this is COMEDY, not just weird. Make us LAUGH.
• COMMIT TO THE BIT - play it straight while the premise is insane
• ABSURDIST LOGIC - the internal logic should be consistent but impossible
• ESCALATE - start weird, get weirder, end with a punchline
• VARIETY - NOT everything is an infomercial! Try other formats!

BANNED TOPICS (we've done these too much):
• Memory/nostalgia devices
• Trauma therapy / inner child healing
• Thought control / conformity
• Corporate wellness

Stay in character. Create something genuinely FUNNY.
""",

    # IDCC Phase-specific prompts
    # =========================================================================
    # IDCC WRITERS' ROOM - CONSENSUS-BASED CREATIVE DEVELOPMENT
    # =========================================================================
    # Round 1: Pitch complete concepts (FORMAT + PREMISE + COMEDIC_HOOK)
    # Round 2: Vote for favorite pitch (not your own) + add improvement
    # Round 3: Character packages (DESCRIPTION + VOICE + ARC)
    # Round 4: Vote for best character package (not your own)
    # =========================================================================

    "idcc_spitball_round1": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 1: THE PITCH)

⚠️ RESPOND WITH PLAIN TEXT ONLY. No tools, no code, just your pitch.

You're in a writers' room pitching ideas for an INTERDIMENSIONAL CABLE clip.

**YOUR PITCH MUST INCLUDE ALL THREE:**

1. **FORMAT**: Pick something OTHER than infomercial! Try: cooking show, workout video, talk show, dating show, nature documentary, movie trailer, kids show, home shopping, court TV, reality competition, travel show, late-night ad, PSA, news segment

2. **PREMISE**: What's being sold/shown/happening? The absurd concept presented totally straight-faced.

3. **THE BIT**: What's the JOKE? What makes this FUNNY? What's the comedic escalation?

🚫 **BANNED CONCEPTS - DO NOT PITCH THESE:**
• Memory devices / memory erasure / nostalgia machines
• Trauma therapy / inner child / psychological healing products
• Thought control / conformity / mind compliance
• Corporate wellness / productivity optimization
• Dystopian PSAs about thinking/feeling
• Anything "therapeutic" or "psychological"
WE'VE DONE THESE TO DEATH. BE MORE CREATIVE.

✅ **PITCH SOMETHING LIKE:**
• A cooking show where the chef only uses ingredients that don't exist
• A dating show where one contestant is clearly 3 aliens in a trenchcoat
• A workout video for muscles humans don't have
• A nature documentary about furniture migration patterns
• A travel show visiting places that are just feelings
• A court TV show where the crime is "being too tall on a Wednesday"
• A kids show teaching dangerous nonsense with enthusiasm

**THE INTERDIMENSIONAL CABLE VIBE:**
• ABSURDIST - logic breaks down but everyone acts normal
• COMMITTED - play it straight while the premise is insane
• ESCALATING - the bit gets weirder/worse as it goes
• FUNNY - there must be JOKES, not just weird concepts

**EXAMPLE PITCHES:**
"FORMAT: Cooking show. PREMISE: Chef is teaching a recipe but keeps substituting ingredients with abstract concepts. 'Now add 3 cups of your father's approval.' THE BIT: He keeps cooking invisible food, tasting it seriously, and the dish keeps coming out as different emotions."

"FORMAT: Workout video. PREMISE: Instructor teaching exercises for your 'third arm' that humans clearly don't have. THE BIT: He keeps demonstrating impossible movements, getting frustrated that viewers 'aren't trying hard enough', eventually revealing his third arm was CGI the whole time."

**YOUR PITCH SHOULD MAKE US LAUGH.** What's genuinely funny about your idea?

Give your complete pitch in 3-4 sentences covering FORMAT, PREMISE, and THE BIT.
""",

    "idcc_spitball_round2_vote": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 2: VOTE & IMPROVE)

⚠️ RESPOND WITH PLAIN TEXT ONLY. No tools, no code.

Here are the pitches from Round 1:

{all_pitches}

**YOUR TASK:**
1. **VOTE** for your FAVORITE pitch (you CANNOT vote for your own)
2. **ADD ONE IMPROVEMENT** - a twist, escalation idea, or comedic beat that makes it even better

**FORMAT YOUR RESPONSE EXACTLY LIKE THIS:**
MY VOTE: [Agent Name]
BECAUSE: [One sentence why this pitch is the funniest]
MY IMPROVEMENT: [One specific addition that makes it better]

**RULES:**
• You CANNOT vote for yourself
• You MUST pick exactly one pitch
• Your improvement should ADD to the idea, not replace it

The pitch with the most votes becomes our FORMAT, PREMISE, and COMEDIC_HOOK for the Show Bible.
""",

    "idcc_spitball_round3_character": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 3: CHARACTER PACKAGE)

⚠️ RESPOND WITH PLAIN TEXT ONLY. No tools, no code.

**THE WINNING CONCEPT:**
{winning_concept}

Now we need to create the CHARACTER. Your job: propose a complete CHARACTER PACKAGE.

**YOUR CHARACTER PACKAGE MUST INCLUDE:**

1. **CHARACTER_DESCRIPTION**: One DETAILED sentence describing their EXACT APPEARANCE. This will be COPY-PASTED VERBATIM into every video prompt, so be EXTREMELY SPECIFIC:
   • SPECIES: Human? Alien? Robot? Animal? What kind?
   • BODY: Build, posture, any unusual features
   • FACE: Eye color, facial hair, expression, any unique features
   • HAIR: Style, color, length (CRITICAL - be specific!)
   • CLOTHING: Exact outfit with colors
   • PROPS/ACCESSORIES: Anything they're holding or wearing

   GOOD: "A pudgy middle-aged human man with a brown combover, thick black mustache, sweaty red face, wearing a too-tight powder blue polo shirt tucked into khaki shorts, holding a spatula"

   BAD: "A friendly host" (too vague - will look different every scene!)

2. **VOCAL_SPECS**: EXACT voice description for text-to-speech consistency. Include:
   • Gender/pitch (male/female, high/low)
   • Accent or speech pattern
   • Energy level and delivery style

   GOOD: "speaks in gravelly male baritone with Southern drawl, overly enthusiastic infomercial cadence"
   BAD: "energetic voice" (too vague!)

3. **ARC**: How does the character EMOTIONALLY ESCALATE across scenes? Show the journey:
   GOOD: "Confident and slick → slight confusion → visible panic → complete breakdown → acceptance of absurdity"
   BAD: "Gets funnier" (not specific!)

**FORMAT YOUR RESPONSE EXACTLY LIKE THIS:**
CHARACTER_DESCRIPTION: [Detailed visual sentence with species, face, hair, clothing, props]
VOCAL_SPECS: [Pitch, accent, energy, delivery style]
ARC: [Emotional journey with 4-5 beats]

**THE CHARACTER SHOULD BE FUNNY TO LOOK AT.** Give them visual comedy potential.
""",

    "idcc_spitball_round4_vote": """
📺 INTERDIMENSIONAL CABLE - WRITERS' ROOM (Round 4: FINAL VOTE)

⚠️ RESPOND WITH PLAIN TEXT ONLY. No tools, no code.

**THE WINNING CONCEPT:**
{winning_concept}

Here are the CHARACTER PACKAGES proposed:

{all_character_packages}

**YOUR TASK:**
Vote for the BEST character package (you CANNOT vote for your own).

**FORMAT YOUR RESPONSE EXACTLY LIKE THIS:**
MY VOTE: [Agent Name]
BECAUSE: [One sentence why this character is funniest for our concept]

**RULES:**
• You CANNOT vote for yourself
• You MUST pick exactly one character package
• Consider: Does the character LOOK match the bit? Does the VOICE fit? Does the ARC escalate well?

The character package with the most votes becomes our CHARACTER_DESCRIPTION, CHARACTER_VOICE, and ARC for the Show Bible.
""",

    "idcc_scene_opening": """
📺 INTERDIMENSIONAL CABLE - SCENE 1 (OPENING)

═══════════════════════════════════════════════════════════════
SHOW BIBLE
═══════════════════════════════════════════════════════════════
{show_bible}
═══════════════════════════════════════════════════════════════

**SHOT DIRECTION:** {shot_direction}

This is Scene 1 - ESTABLISH the character and premise. SET UP the joke.

🎬 **OUTPUT FORMAT - FOLLOW EXACTLY:**
Your output must be a SINGLE VIDEO PROMPT paragraph. Nothing else. No commentary, no scene numbers, no "here's my prompt", no meta-text. Just the prompt itself.

⚠️ **CRITICAL REQUIREMENTS:**

1. **STYLE** (always first): "Adult Swim cartoon style, 2D animation, bold outlines, flat colors."

2. **ONE SPEAKER ONLY:** Check "THIS SCENE'S SPEAKER" in the Show Bible. ONLY that character speaks in this scene. Other characters may be visible but stay SILENT.

3. **CHARACTER** (copy-paste EXACTLY from Show Bible):
   Copy the CHARACTER_DESCRIPTION word-for-word. Do NOT paraphrase or shorten it.

4. **VOICE** (copy-paste EXACTLY from Show Bible):
   Write: "[Character] speaks in [VOCAL_SPECS from Show Bible]"
   Copy it EXACTLY - same pitch, same accent, same energy.

5. **DIALOGUE** (2-3 SHORT lines, ~20 words MAXIMUM):
   Format: Dialogue begins at 0:01, ends by 0:09. Dialogue: "[Line 1]" "[Line 2]" "[Line 3]"
   • MUST include the mandatory line from DIALOGUE_BEATS for Scene 1
   • STRICT LIMIT: 20 words total (~2.5 words/sec × 8 sec speaking time)
   • Lines should be FUNNY - jokes, absurdist observations, commitment to the bit
   • ONLY the designated speaker talks - no other voices

6. **ACTION** (one clear visual):
   What is the character DOING while speaking?

7. **SCENE ENDING** (check "SCENE ENDING" in Show Bible):
   At the END of the scene, brief TV static/channel change effect, then cut to the NEXT scene's speaker (mouth CLOSED, silent, not speaking yet).
   This creates a visual handoff like flipping channels on Interdimensional Cable.

**⚠️ CRITICAL TIMING - 12 SECOND CLIP:**
• Dialogue: 0:01 to 0:09 (8 seconds speaking = ~20 words MAX)
• Character STOPS speaking by 0:09 and holds pose
• TV static/channel flip effect: 0:10 to 0:11
• Cut to NEXT scene's speaker (mouth CLOSED, silent): 0:11 to 0:12
• If dialogue is too long, it will bleed into the next scene!

**DIALOGUE MUST BE FUNNY:**
• Include at least ONE absurdist non-sequitur or logic break
• The humor comes from COMMITMENT to the insane premise
• Don't just state the premise - make JOKES about it

**EXAMPLE OUTPUT (Real Fake Doors style):**
Adult Swim cartoon style, 2D animation, bold outlines, flat colors. Wide shot of a fake showroom full of doors. A lanky middle-aged man with messy brown hair, wide manic eyes, wearing a rumpled short-sleeve dress shirt and loose tie, gestures enthusiastically at a door. He speaks in nasally tenor, fast-talking salesman energy, slightly unhinged. Dialogue begins at 0:01, ends by 0:09. Dialogue: "Hey, are you tired of real doors?" "Come on down to Real Fake Doors!" He tries to open a door and it doesn't budge. He grins and holds pose at 0:09. TV static channel flip at 0:10, then cut to a confused customer at 0:11, mouth closed, silent, waiting.

🚫 **DO NOT OUTPUT:**
• Scene numbers or labels ("Scene 1:", "Opening:")
• Meta-commentary ("Here's my prompt:", "This scene will...")
• Instructions or explanations
• Multiple characters speaking in the same scene
• Anything except the video prompt itself

Output ONLY the video prompt paragraph. Start with "Adult Swim cartoon style..."
""",

    "idcc_scene_middle": """
📺 INTERDIMENSIONAL CABLE - SCENE {scene_number} of {num_clips}

═══════════════════════════════════════════════════════════════
SHOW BIBLE
═══════════════════════════════════════════════════════════════
{show_bible}
═══════════════════════════════════════════════════════════════

**SHOT DIRECTION:** {shot_direction}

**PREVIOUS SCENE:** {previous_prompt}

This is Scene {scene_number} - ESCALATE the comedy. Things get WEIRDER or WORSE.

🎬 **OUTPUT FORMAT - FOLLOW EXACTLY:**
Your output must be a SINGLE VIDEO PROMPT paragraph. Nothing else. No commentary, no scene numbers, no meta-text. Just the prompt itself.

⚠️ **CRITICAL REQUIREMENTS:**

1. **STYLE** (always first): "Adult Swim cartoon style, 2D animation, bold outlines, flat colors."

2. **ONE SPEAKER ONLY:** Check "THIS SCENE'S SPEAKER" in the Show Bible. ONLY that character speaks in this scene. This may be a DIFFERENT character than Scene 1 (testimonial, reporter, customer) - that's intentional!

3. **CHARACTER** (this scene's speaker):
   If this scene's speaker is the Host/main character: Copy CHARACTER_DESCRIPTION exactly.
   If this scene's speaker is a secondary character: Create a BRIEF but SPECIFIC visual description for them.

4. **VOICE** (this scene's speaker):
   If Host: Use VOCAL_SPECS from Show Bible.
   If secondary character: Give them DIFFERENT vocal specs (different pitch, energy, accent).

5. **DIALOGUE** (2-3 SHORT lines, ~20 words MAXIMUM):
   Format: Dialogue begins at 0:01, ends by 0:09. Dialogue: "[Line 1]" "[Line 2]" "[Line 3]"
   • MUST include the mandatory line from DIALOGUE_BEATS for Scene {scene_number}
   • STRICT LIMIT: 20 words total (~2.5 words/sec × 8 sec speaking time)
   • Lines should ESCALATE - things getting weirder, character reacting
   • Include JOKES - not just plot, actual comedy
   • ONLY the designated speaker talks

6. **ACTION** (escalating visual):
   What is happening that makes things WORSE or WEIRDER?

7. **SCENE ENDING** - CRITICAL TIMING:
   • Character STOPS speaking by 0:09 and holds pose
   • TV static/channel flip effect: 0:10 to 0:11
   • Cut to NEXT scene's speaker (mouth CLOSED, silent): 0:11 to 0:12
   • If dialogue runs past 0:09, it will bleed into the next scene!

**ESCALATION TECHNIQUES:**
• The premise's logic breaks down further
• A demonstration goes horribly right/wrong
• The absurdity compounds on itself
• Physical comedy - something visual goes wrong

**EXAMPLE OUTPUT (Ants in My Eyes Johnson style testimonial cutaway):**
Adult Swim cartoon style, 2D animation, bold outlines, flat colors. Testimonial cutaway in a cheap electronics store. A heavyset balding man in a short-sleeve button-up, his eyes visibly full of crawling ants, stands surrounded by TVs. He speaks in strained cheerful tenor, desperately upbeat despite obvious distress. Dialogue begins at 0:01, ends by 0:09. Dialogue: "I'm Ants in My Eyes Johnson!" "Everything's black, I can't see a thing!" "But that's not as catchy!" He knocks over a display while gesturing blindly, then holds pose at 0:09. TV static channel flip at 0:10, cut to a different man at a news desk at 0:11, mouth closed, silent, waiting.

🚫 **DO NOT OUTPUT:**
• Scene numbers or labels ("Scene 2:", "Middle scene:")
• Meta-commentary ("This scene escalates...", "Building on...")
• Instructions like "Build on the comedic hook"
• Multiple characters speaking in the same scene
• Anything except the video prompt itself

Output ONLY the video prompt paragraph. Start with "Adult Swim cartoon style..."
""",

    "idcc_scene_final": """
📺 INTERDIMENSIONAL CABLE - FINAL SCENE ({scene_number} of {num_clips})

═══════════════════════════════════════════════════════════════
SHOW BIBLE
═══════════════════════════════════════════════════════════════
{show_bible}
═══════════════════════════════════════════════════════════════

**SHOT DIRECTION:** {shot_direction}

**PREVIOUS SCENE:** {previous_prompt}

THIS IS THE FINAL SCENE. LAND THE JOKE. This is the punchline of the whole bit.

🎬 **OUTPUT FORMAT - FOLLOW EXACTLY:**
Your output must be a SINGLE VIDEO PROMPT paragraph. Nothing else. No commentary, no scene numbers, no meta-text. Just the prompt itself.

⚠️ **CRITICAL REQUIREMENTS:**

1. **STYLE** (always first): "Adult Swim cartoon style, 2D animation, bold outlines, flat colors."

2. **ONE SPEAKER ONLY:** Check "THIS SCENE'S SPEAKER" in the Show Bible. ONLY that character speaks. Usually the Host returns for the finale.

3. **CHARACTER** (copy-paste EXACTLY from Show Bible):
   Copy the CHARACTER_DESCRIPTION word-for-word. Character MUST look identical to earlier Host scenes.

4. **VOICE** (copy-paste EXACTLY from Show Bible):
   Write: "[Character] speaks in [VOCAL_SPECS from Show Bible]"
   MUST be IDENTICAL to earlier Host scenes.

5. **DIALOGUE** (2-3 SHORT lines, ~20 words MAXIMUM, building to punchline):
   Format: Dialogue begins at 0:01, ends by 0:10. Dialogue: "[Setup]" "[Build]" "[PUNCHLINE]"
   • MUST end with the PUNCHLINE from DIALOGUE_BEATS
   • STRICT LIMIT: 20 words total (~2.5 words/sec × 8 sec speaking time)
   • Build the other lines to SET UP that punchline
   • The final line is THE JOKE - make it land
   • ONLY the designated speaker talks

6. **ACTION** (conclusion):
   Visual payoff that reinforces the punchline.

7. **CLEAN ENDING:** This is the final scene - NO static transition. Character delivers punchline by 0:10, holds final pose.

**GREAT ENDINGS:**
• The absurd premise reaches its logical extreme ("I live here now")
• The character fully breaks or fully commits
• A twist that recontextualizes everything
• The thing "works" in an unexpectedly horrible way
• Deadpan acceptance of the insanity
• A disturbing "call now" beat

**PUNCHLINE TECHNIQUES:**
• Callback to the opening with a dark twist
• Character gives up and accepts the absurdity
• The bit goes one step TOO far
• Understatement after chaos
• Non-sequitur that somehow makes perfect sense

**EXAMPLE OUTPUT (Lil' Bits style finale):**
Adult Swim cartoon style, 2D animation, bold outlines, flat colors. Extreme close-up on a tiny restaurant table with microscopic food. A whispery disembodied voice speaks in breathy, unsettling ASMR whisper, intimately creepy. Dialogue begins at 0:01, ends by 0:10. Dialogue: "Lil' Bits..." "Eat some shit, you stupid bitch." "Just kidding. Lil' Bits." The camera slowly zooms into impossibly small food. Character holds final pose as whisper fades.

🚫 **DO NOT OUTPUT:**
• Scene numbers or labels ("Final scene:", "Scene 5:")
• Meta-commentary ("Landing the joke...", "The punchline is...")
• Instructions like "Build on the comedic hook" or "This is scene X of Y"
• Multiple characters speaking in the same scene
• Anything except the video prompt itself
• A weak ending - this MUST be funny

Output ONLY the video prompt paragraph. Start with "Adult Swim cartoon style..."
"""
}


# ============================================================================
# IDCC FORMAT-AWARE SHOT DIRECTION SYSTEM
# ============================================================================
# Each format has a typical "visual language" - the kinds of shots/framing
# that viewers expect from that type of content. This helps create variety
# within a coherent piece while maintaining format authenticity.

FORMAT_SHOT_TEMPLATES = {
    "infomercial": [
        "Wide shot establishing the set - host at demo table, product visible, cheap studio lighting",
        "Close-up on product demonstration - host's hands showing the item, enthusiasm visible",
        "Testimonial cutaway - 'satisfied customer' reacting, or host addressing camera directly",
        "Medium shot - host with product, situation starting to go wrong or escalate",
        "Final wide shot - product prominent, host in full desperation mode, 'call now' energy"
    ],
    "news": [
        "News anchor framing - behind desk, graphics area visible, professional lighting",
        "Cut to field reporter - on location, handheld documentary feel, reporting live",
        "B-roll footage - demonstration or visualization of the story subject",
        "Back to anchor - reaction shot, processing what was just reported",
        "Sign-off shot - anchor wrapping up, or 'breaking development' dramatic moment"
    ],
    "psa": [
        "Direct-to-camera spokesperson - earnest framing, public service energy, eye contact",
        "Demonstration shot - visualizing the 'problem' being addressed",
        "Emotional appeal - testimonial from affected party, or dramatic reenactment",
        "Escalation shot - the PSA's logic spiraling, absurdity becoming visible",
        "Logo/tagline moment - deadpan conclusion, call to action"
    ],
    "talk_show": [
        "Wide two-shot - host and guest on couch/chairs, talk show set visible",
        "Close-up on host - reaction to something guest said, processing",
        "Close-up on guest - the bit intensifying, guest getting into it",
        "Wide shot or audience reaction - chaos building, energy shifting",
        "Button shot - host trying to wrap up, tension between resolution and chaos"
    ],
    "cooking_show": [
        "Wide kitchen shot - host at counter, ingredients laid out, cooking show lighting",
        "Overhead shot - ingredients or prep work, or close-up on technique",
        "Host reaction shot - tasting, demonstrating, things starting to go strange",
        "Reveal shot - the dish shown, or the absurdity of the situation becoming clear",
        "Final presentation - deadpan 'bon appetit' energy, chef's kiss or horror"
    ],
    "workout_video": [
        "Wide shot of instructor - workout space visible, motivational energy, ready position",
        "Demonstration shot - the 'exercise' being shown, medium framing",
        "Close-up on instructor face - encouragement getting weird, intensity building",
        "Wide shot showing full absurdity - the workout routine revealed in full context",
        "Cool-down shot - exhausted energy, disturbing conclusion, namaste or collapse"
    ],
    "movie_trailer": [
        "Cinematic establishing shot - sets the world, dramatic lighting, scope",
        "Character introduction - dramatic framing, hero shot or mysterious reveal",
        "Action/conflict beat - the premise revealed, stakes shown, tension",
        "Montage energy moment - quick cuts feeling, escalation, music swell implied",
        "Title card beat - final dramatic shot, tagline moment, release date energy"
    ],
    "late_night_ad": [
        "Low-budget wide shot - host surrounded by product, harsh lighting, cheap set",
        "Close-up demonstration - too much enthusiasm, product shown from bad angle",
        "Testimonial shot - 'before/after' energy, or suspiciously enthusiastic customer",
        "'But wait there's more' shot - additional products, escalating offers",
        "Pricing/call-to-action shot - desperation peaks, phone number energy, act now"
    ],
    "documentary": [
        "Establishing shot - location context, documentary realism, natural lighting",
        "Interview framing - subject speaking to off-camera interviewer, intimate",
        "B-roll footage - supporting visuals, evidence, atmosphere building",
        "Dramatic reveal shot - key information visualized, tension building",
        "Conclusion shot - reflection, aftermath, or cliffhanger for next episode"
    ],
    "dating_show": [
        "Wide shot of dating show set - contestant and host, romantic lighting, tacky decorations",
        "Close-up on contestant - nervous energy, hope or confusion visible",
        "Reveal shot - the 'date' option is shown, reaction building",
        "Two-shot interaction - contestant meeting their match, chemistry or horror",
        "Final rose/choice moment - dramatic conclusion, unexpected outcome"
    ],
    "kids_show": [
        "Colorful wide shot - bright set, educational props, enthusiastic host framing",
        "Close-up on host teaching - direct address to camera, big expressions",
        "Demonstration shot - showing the 'lesson' with props or visuals",
        "Audience participation energy - implied kids reacting, chaos building",
        "Goodbye/sign-off shot - wrap-up energy, disturbing cheerfulness"
    ],
    "court_tv": [
        "Courtroom establishing shot - judge's bench, witness stand, legal atmosphere",
        "Witness testimony framing - dramatic close-up, sworn statement energy",
        "Lawyer reaction shot - objection energy, dramatic gestures",
        "Evidence reveal - the key exhibit shown, gasps implied",
        "Verdict shot - gavel moment, justice delivered or denied"
    ],
    "nature_documentary": [
        "Wide establishing shot - the 'habitat' shown, documentary framing",
        "Close-up on subject - intimate detail, nature footage energy",
        "Behavior observation shot - the subject doing something, narrator implied",
        "Dramatic moment - predator/prey energy or mating ritual",
        "Conclusion shot - subject in natural state, cycle of life energy"
    ],
    "home_shopping": [
        "Product glamour shot - item displayed, sparkle lighting, value implied",
        "Host demonstration - showing features, enthusiasm cranked to 11",
        "Price reveal - dramatic value proposition, timer energy",
        "Testimonial insert - happy customer, too enthusiastic",
        "Call-to-action finale - phone number energy, urgency, limited time"
    ],
    "reality_competition": [
        "Contestant lineup shot - competitors shown, tension visible",
        "Challenge reveal - the task explained, stakes established",
        "Competition montage energy - contestants struggling, drama building",
        "Elimination tension - who will go home, dramatic pauses",
        "Winner moment - triumph or twist, emotional climax"
    ],
    "travel_show": [
        "Destination establishing shot - location beauty, wanderlust energy",
        "Host exploring - walking through location, discovery moments",
        "Local interaction - meeting characters, cultural exchange",
        "Food/experience close-up - sensory detail, immersion",
        "Sunset conclusion - reflection, destination summary, where to next"
    ]
}

# Default fallback for unrecognized formats
DEFAULT_SHOT_SEQUENCE = [
    "Wide establishing shot - setting the scene, main subject visible",
    "Medium shot - focusing on the action or demonstration",
    "Close-up or reaction shot - emotional beat, detail work",
    "Wide shot with escalation - situation developing, energy building",
    "Final button shot - punchline framing, conclusion, payoff"
]


def get_shot_direction(show_format: str, scene_number: int, total_scenes: int) -> str:
    """
    Get format-appropriate shot direction for a specific scene.

    Args:
        show_format: The TV format (infomercial, news, etc.)
        scene_number: Current scene (1-indexed)
        total_scenes: Total number of scenes

    Returns:
        Shot direction string for this scene
    """
    # Normalize format string for matching
    format_key = show_format.lower().strip()
    format_key = format_key.replace(" ", "_").replace("-", "_")

    # Find best matching template
    template = None
    for key in FORMAT_SHOT_TEMPLATES:
        if key in format_key or format_key in key:
            template = FORMAT_SHOT_TEMPLATES[key]
            break

    # Check for partial matches
    if not template:
        for key in FORMAT_SHOT_TEMPLATES:
            if any(word in format_key for word in key.split("_")):
                template = FORMAT_SHOT_TEMPLATES[key]
                break

    # Fallback to default
    if not template:
        template = DEFAULT_SHOT_SEQUENCE

    # Map scene number to template index
    # Scale scenes proportionally to template length
    if total_scenes <= 1:
        template_index = 0
    else:
        # Distribute scenes across template
        template_index = int((scene_number - 1) / (total_scenes - 1) * (len(template) - 1))

    # Clamp to valid range
    template_index = max(0, min(template_index, len(template) - 1))

    # Special case: always use last template item for final scene
    if scene_number == total_scenes:
        template_index = len(template) - 1

    return template[template_index]


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
        "max_tokens": 250,            # Complete pitch with FORMAT/PREMISE/BIT
    },
    "idcc_spitball_round2_vote": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 150,            # Vote + short improvement
    },
    "idcc_spitball_round3_character": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 200,            # Character package
    },
    "idcc_spitball_round4_vote": {
        "response_frequency": 15,
        "response_likelihood": 100,
        "max_tokens": 100,            # Just a vote + reasoning
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
