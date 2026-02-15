from google import genai
from google.genai import types


API_KEY = "AIzaSyCJ9OhCf9UoMATo4-AeFVlWHatsCE6ZJ-w"

# ---------------------------------------------------------------------------
# Personality definitions
# ---------------------------------------------------------------------------

PERSONALITIES = {
    "encouraging": (
        "You are a warm, encouraging rehabilitation coach. You celebrate small wins "
        "and always highlight progress, no matter how incremental. Use positive "
        "reinforcement and motivational language. When the patient struggles, "
        "reassure them that setbacks are normal and frame challenges as "
        "opportunities to grow. Example tone: 'Great effort! I noticed your "
        "coordination improved by 12%% — keep that up!'"
    ),
    "drill_sergeant": (
        "You are a firm, no-nonsense rehabilitation coach. You push your patient "
        "to do better every single attempt. You are direct, concise, and "
        "results-oriented. You acknowledge good work briefly but always emphasize "
        "what to improve next. Use short, commanding sentences. Example tone: "
        "'Your coordination dropped. Again. Focus. Tighten that grip and give me "
        "a clean release.'"
    ),
    "analytical": (
        "You are a data-driven, clinical rehabilitation coach. You provide "
        "detailed metrics breakdowns and evidence-based recommendations. You "
        "reference specific numbers and trends. Keep your tone professional and "
        "precise. Example tone: 'Your coordination index of 0.34 places you in "
        "the 60th percentile of your session. The trend over the last 10 attempts "
        "shows a 5%% decline — consider adjusting your extension amplitude.'"
    ),
    "calm": (
        "You are a calm, zen-like rehabilitation coach. You speak slowly and "
        "gently. You incorporate breathing cues and mindfulness into your "
        "guidance. You never rush the patient. Focus on body awareness and "
        "relaxation. Example tone: 'Take a breath. Feel the movement. Let's try "
        "that wrist extension one more time, slowly and gently.'"
    ),
    "humorous": (
        "You are a lighthearted, humorous rehabilitation coach. You use gentle "
        "humor and wit to keep sessions fun and engaging. You make the patient "
        "smile while still providing useful guidance. Never be mean-spirited. "
        "Example tone: 'Your wrist said it wants a promotion after that attempt "
        "— best coordination score yet!'"
    ),
}

# ---------------------------------------------------------------------------
# Movement-class descriptions (for the system prompt)
# ---------------------------------------------------------------------------

MOVEMENT_DESCRIPTIONS = {
    "rest": (
        "Rest: The patient is in a relaxed, baseline state. No active movement "
        "coaching is needed. You may encourage them to relax, breathe, and "
        "prepare mentally for the next exercise."
    ),
    "motor_imagery": (
        "Motor Imagery: The patient is imagining a movement without physically "
        "performing it. Guide them through vivid visualization — e.g. 'Imagine "
        "your wrist slowly extending, feel the muscles activating in your mind.' "
        "Encourage focus on the motor cortex activation pattern."
    ),
    "wrist_flex_ext": (
        "Wrist Flexion/Extension: The patient alternates between flexing and "
        "extending the wrist. Coach smooth, rhythmic transitions. Common cues: "
        "'relax between flexions', 'keep a steady tempo', 'extend fully before "
        "flexing back'."
    ),
    "grip_release": (
        "Grip Release: The patient sustains a grip and then releases cleanly. "
        "Coach sustained grip force followed by a crisp release. Common cues: "
        "'hold the grip steady', 'release quickly and completely', 'focus on "
        "the transition from grip to open hand'."
    ),
    "cocontraction": (
        "Co-contraction: The patient activates antagonist muscles simultaneously "
        "and holds. Coach steady, even activation. Common cues: 'keep both "
        "muscle groups firing evenly', 'hold the contraction steady', 'breathe "
        "through the hold'."
    ),
}


# ---------------------------------------------------------------------------
# LLMPipeline
# ---------------------------------------------------------------------------

class LLMPipeline:
    """Gemini-powered rehabilitation coaching assistant."""

    # Rate-limiting: only send a full feedback call every N attempts
    FEEDBACK_INTERVAL = 10  # detailed feedback every 10 attempts
    MODEL_NAME = "gemini-2.0-flash"

    def __init__(
        self,
        api_key: str = API_KEY,
        movement_class: str = "wrist_flex_ext",
        personality: str = "encouraging",
    ):
        # Create Gemini client
        self.client = genai.Client(api_key=api_key)

        self.personality = personality
        self.movement_class = movement_class
        self._attempt_counter = 0

        # Build system prompt and start chat
        self._system_prompt = self._build_system_prompt()
        self._start_chat()

    # ------------------------------------------------------------------
    # Chat session management
    # ------------------------------------------------------------------

    def _start_chat(self, history: list | None = None) -> None:
        """Create a new chat session with the current system prompt."""
        if history is None:
            history = [
                types.UserContent(parts=[types.Part.from_text(
                    text="Begin session. Acknowledge and wait for movement data."
                )]),
                types.ModelContent(parts=[types.Part.from_text(
                    text=(
                        "Understood. I'm ready to coach you through your "
                        "rehabilitation session. Waiting for the first movement data."
                    )
                )]),
            ]

        self.chat = self.client.chats.create(
            model=self.MODEL_NAME,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
            ),
            history=history,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_personality(self, personality: str) -> None:
        """Switch personality mid-session, preserving a context summary."""
        if personality not in PERSONALITIES:
            raise ValueError(
                f"Unknown personality '{personality}'. "
                f"Choose from: {list(PERSONALITIES.keys())}"
            )
        self.personality = personality
        self._system_prompt = self._build_system_prompt()

        # Rebuild the chat with a brief context carryover
        summary_history = [
            types.UserContent(parts=[types.Part.from_text(
                text=(
                    f"The session is continuing. The patient has completed "
                    f"{self._attempt_counter} attempts so far. The coaching "
                    f"personality has been switched to '{personality}'. "
                    f"Continue coaching from where we left off."
                )
            )]),
            types.ModelContent(parts=[types.Part.from_text(
                text=f"Got it — switching to {personality} mode. Let's keep going."
            )]),
        ]
        self._start_chat(history=summary_history)

    def set_movement_class(self, movement_class: str) -> None:
        """Update the current exercise type and rebuild the chat."""
        self.movement_class = movement_class
        self._system_prompt = self._build_system_prompt()
        # Restart the chat with the new system instruction
        self._start_chat()

    def generate_feedback(self, metrics: dict) -> str | None:
        """Generate feedback after a movement attempt.

        Respects rate-limiting: returns detailed feedback every
        FEEDBACK_INTERVAL attempts; returns None in between (the caller
        can choose to show nothing or a simple acknowledgment).

        If a plateau is detected the rate limit is bypassed.
        """
        self._attempt_counter += 1

        # Always respond on plateau
        if metrics.get("plateau_detected"):
            return self.generate_plateau_intervention(metrics)

        # Rate-limit normal feedback
        if self._attempt_counter % self.FEEDBACK_INTERVAL != 0:
            return None

        prompt = self._build_metrics_prompt(metrics)
        prompt += (
            "\n\nProvide brief, actionable coaching feedback for this attempt. "
            "Keep it to 2-3 sentences maximum."
        )
        return self._send(prompt)

    def generate_plateau_intervention(self, metrics: dict) -> str:
        """Generate targeted guidance when a plateau or stagnation is detected."""
        prompt = self._build_metrics_prompt(metrics)

        plateau_type = metrics.get("plateau_type", "unknown")
        details = metrics.get("plateau_details", "")

        if plateau_type == "coordination_plateau":
            prompt += (
                "\n\n**PLATEAU ALERT**: The patient's coordination index has "
                "flatlined over the recent attempts. They are not improving or "
                "declining — just stuck. "
                f"Details: {details}\n"
                "Provide specific, actionable guidance to help them break through "
                "this plateau. Consider suggesting: a different approach to the "
                "movement, breaking it into smaller components, adjusting timing "
                "or amplitude, or taking a brief rest. 3-5 sentences."
            )
        elif plateau_type == "low_uniqueness":
            prompt += (
                "\n\n**STAGNATION ALERT**: The patient keeps repeating the same "
                "motion pattern without exploring new movement strategies. "
                f"Details: {details}\n"
                "Encourage them to vary their approach. Suggest specific "
                "modifications they can try — different speed, range of motion, "
                "mental imagery, or focus on a different muscle group. 3-5 sentences."
            )
        else:
            prompt += (
                "\n\n**INTERVENTION NEEDED**: The system has detected the patient "
                "may be struggling. Provide supportive, actionable guidance. "
                "3-5 sentences."
            )

        return self._send(prompt)

    def ask(self, user_question: str, current_metrics: dict | None = None) -> str:
        """Handle a user-initiated follow-up question."""
        prompt = ""
        if current_metrics:
            prompt += self._build_metrics_prompt(current_metrics) + "\n\n"
        prompt += f"The patient asks: {user_question}"
        return self._send(prompt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, user_message: str) -> str:
        """Send a message through the chat session and return the response.

        Returns a fallback string if the API is unavailable (e.g. rate-limited).
        The SDK handles transient retries internally via tenacity.
        """
        try:
            response = self.chat.send_message(user_message)
            return response.text
        except Exception as exc:
            exc_str = str(exc)
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                return (
                    "[Coach is momentarily unavailable due to high demand. "
                    "Keep going — I'll check in on your next attempt!]"
                )
            return f"[Coach unavailable: {type(exc).__name__}]"

    def _build_system_prompt(self) -> str:
        """Construct the full system prompt from personality + context."""
        personality_desc = PERSONALITIES.get(self.personality, PERSONALITIES["encouraging"])
        movement_desc = MOVEMENT_DESCRIPTIONS.get(
            self.movement_class,
            "Unknown exercise. Provide general rehabilitation coaching."
        )

        return f"""You are an AI rehabilitation coaching assistant integrated into a stroke rehabilitation device that monitors EEG (brain activity) and EMG (muscle activity) signals in real time.

PERSONALITY:
{personality_desc}

CURRENT EXERCISE:
{movement_desc}

METRICS YOU WILL RECEIVE:
- Coordination Index (float, 0-1): Measures how coordinated the patient's movement is. LOWER is BETTER (0 = perfect coordination). Derived from SVD energy ratio (how much variance is captured by the dominant movement pattern) and bimodality coefficient (whether the signal clusters cleanly).
- Similarity Score (float): DTW (Dynamic Time Warping) distance comparing the current attempt to the patient's best previous attempts or a predicted ideal. LOWER is BETTER (more consistent with best performance).
- Attempt Number: How many movement attempts the patient has completed this session.
- Coordination History: Recent trend of coordination indices.
- Trend: Whether the patient is "improving", "declining", or on a "plateau".
- Top-N Scores: The coordination indices and similarity scores of the patient's best cached attempts.

GUIDELINES:
- Keep responses concise and actionable (2-5 sentences for regular feedback, up to 5 for interventions).
- Refer to the specific exercise the patient is performing.
- Use the metrics to provide personalized, data-informed feedback.
- When the patient is improving, acknowledge it clearly.
- When they are declining or plateauing, provide specific technique adjustments.
- NEVER provide medical diagnoses or clinical treatment recommendations.
- Always remind the patient to stop if they experience pain.
- Defer to their real doctor or therapist for clinical decisions.
- Speak directly to the patient in second person ("you").
"""

    def _build_metrics_prompt(self, metrics: dict) -> str:
        """Format pipeline metrics into a readable context block for the LLM."""
        lines = ["--- Session Metrics ---"]

        lines.append(f"Exercise: {metrics.get('movement_class', self.movement_class)}")
        lines.append(f"Attempt #: {metrics.get('attempt_number', self._attempt_counter)}")

        ci = metrics.get("coordination_index")
        if ci is not None:
            lines.append(f"Coordination Index: {ci:.4f}  (lower = better)")

        sim = metrics.get("similarity_score")
        if sim is not None:
            lines.append(f"Similarity Score: {sim:.4f}  (lower = more consistent)")

        trend = metrics.get("trend")
        if trend:
            lines.append(f"Recent Trend: {trend}")

        history = metrics.get("coordination_history")
        if history:
            recent = history[-10:]
            formatted = ", ".join(f"{v:.3f}" for v in recent)
            lines.append(f"Last {len(recent)} Coordination Indices: [{formatted}]")

        top_n = metrics.get("top_n_scores")
        if top_n:
            lines.append("Top Cached Attempts:")
            for i, entry in enumerate(top_n, 1):
                ci_val = entry.get("coordination_index", "?")
                sim_val = entry.get("similarity_score", "?")
                lines.append(f"  #{i}: coord={ci_val}, similarity={sim_val}")

        has_model = metrics.get("has_prediction_model")
        if has_model is not None:
            lines.append(f"Prediction Model Active: {'Yes' if has_model else 'No'}")

        plateau = metrics.get("plateau_detected", False)
        if plateau:
            lines.append(f"Plateau Detected: YES — type: {metrics.get('plateau_type', 'unknown')}")

        lines.append("--- End Metrics ---")
        return "\n".join(lines)
