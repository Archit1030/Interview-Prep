import os
import json
from google import genai
from google.genai import types
from engine.difficulty import get_difficulty_prompt
from engine.personas import get_persona_prompt

class AIEngine:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("CRITICAL: GOOGLE_API_KEY not found.")
        self.client = genai.Client(api_key=api_key)
        self.model_id = 'gemini-1.5-flash'
        self.history = []
        self.system_instruction = ""

    def reset_session(self, style="FAANG_Architect", difficulty="Intermediate", topic="System Design", resume_context=None):
        try:
            persona_prompt = get_persona_prompt(style)
            difficulty_prompt = get_difficulty_prompt(difficulty)

            self.system_instruction = (
                f"{persona_prompt}\n\n"
                f"{difficulty_prompt}\n\n"
                f"The specific interview topic is: {topic}.\n"
                "You are conducting a live video interview. "
                "Keep responses concise (1-3 sentences). Do not write long paragraphs."
            )
            if resume_context:
                self.system_instruction += f"\n\nRESUME CONTEXT: {resume_context}"

            self.history = []
            print(f"✅ AI Initialized: {style} | {difficulty} | {topic}")

            response = self.client.models.generate_content(
                model=self.model_id,
                contents=f"Start the interview. Ask the first question about {topic}.",
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction
                )
            )
            opening = response.text
            self.history.append({"role": "model", "parts": [{"text": opening}]})
            return opening

        except Exception as e:
            print(f"⚠️ AI Init Warning: {e}")
            self.history = []
            return "Hello. I'm ready to interview you. Shall we begin?"

    def get_response(self, user_text, metrics):
        prompt = (
            f"[Metrics] Eye Contact: {metrics.get('eye_contact_score', 0):.2f}, "
            f"Smiling: {metrics.get('is_smiling', False)}\n\n"
            f"Candidate Answer: \"{user_text}\"\n\n"
            "Respond to the answer. If eye contact is consistently low (<0.4), "
            "briefly mention it supportively once."
        )

        self.history.append({"role": "user", "parts": [{"text": prompt}]})

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=self.history,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction
            )
        )
        reply = response.text
        self.history.append({"role": "model", "parts": [{"text": reply}]})
        return reply

    def generate_feedback_report(self, transcript_text, behavioral_metrics=None):
        prompt = f"""
Analyze this interview transcript and return a JSON object.

TRANSCRIPT:
{transcript_text}

REQUIRED JSON FORMAT:
{{
    "radar_chart": {{
        "technical_accuracy": <0-100>,
        "communication_clarity": <0-100>,
        "confidence_level": <0-100>,
        "problem_solving": <0-100>,
        "cultural_fit": <0-100>
    }},
    "feedback": {{
        "strengths": ["point 1", "point 2"],
        "improvements": ["point 1", "point 2"],
        "hiring_verdict": "HIRE"
    }},
    "summary": "A 2-sentence summary."
}}
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Report Gen Error: {e}")
            return None
