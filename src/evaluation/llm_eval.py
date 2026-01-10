import os
import re
import json
import asyncio
from openai import AsyncOpenAI
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

class LlmEvaluator:
    def __init__(self, provider='openai'):
        self.provider = provider
        if self.provider == 'openai':
            self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-3.5-turbo"
        elif self.provider == 'google':
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            raise ValueError("Unsupported LLM provider. Choose 'openai' or 'google'.")

    def _get_prompt_template(self):
        return """
        The following exercise tests a language model's abilities and creativity.
        The model was given the beginning of a story and was required to complete it.
        The prompt ends and the model's completion begins after the "***" symbol.

        Here is the prompt and the model's completion:
        {story}

        Your task is to assess the model's completion based on the following criteria:
        - **Grammar**: Are there any grammatical errors in the completion?
        - **Consistency**: Does the completion logically and stylistically follow from the beginning prompt?
        - **Creativity**: How original or interesting is the model's addition to the story?
        - **Plot Sense**: Does the story have a coherent and sensible plot progression?

        Please provide a score from 1 to 10 for each criterion.
        Your response MUST be ONLY a JSON object with the keys 'grammar', 'creativity', 'consistency', and 'plot_sense'.
        Do not add any text before or after the JSON object.

        Example format:
        {
          "grammar": 8,
          "creativity": 7,
          "consistency": 9,
          "plot_sense": 8
        }
        """

    async def _evaluate_openai(self, story):
        prompt = self._get_prompt_template().format(story=story)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return None

    async def _evaluate_google(self, story):
        prompt = self._get_prompt_template().format(story=story)
        try:
            # Use synchronous call wrapped in asyncio for Google API
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.model.generate_content, prompt)
            # Clean up potential markdown formatting
            text_response = response.text.strip().replace("```json", "").replace("```", "").strip()
            return text_response
        except Exception as e:
            print(f"Error calling Google API: {e}")
            return None

    async def evaluate_story(self, story_text):
        if self.provider == 'openai':
            result_str = await self._evaluate_openai(story_text)
        else:
            result_str = await self._evaluate_google(story_text)
        
        if not result_str:
            return None

        try:
            # Parse the JSON string into a dictionary
            return json.loads(result_str)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from LLM response: {result_str}")
            return None

async def main():
    """Example usage of the LlmEvaluator class."""
    evaluator = LlmEvaluator(provider='openai') # or 'google'
    story = "Once upon a time, there was a little boat. It loved to sail on the big blue sea. *** One day, the boat saw a big whale. The whale was sad. 'Why are you sad?' asked the little boat. The whale said, 'I lost my family.' The little boat helped the whale find its family, and they were all happy again."
    
    scores = await evaluator.evaluate_story(story)
    if scores:
        print("Evaluation Scores:")
        print(json.dumps(scores, indent=2))

if __name__ == "__main__":
    asyncio.run(main())