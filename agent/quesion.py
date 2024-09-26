import asyncio
import csv
from grag_api import GraphRAG
import json
from openai import AsyncOpenAI


class QuestionAgent:
    def __init__(self, graph, api_key=None):
        self.prompt = '''
        ---Role---

You are an intelligent teacher assistant responsible for evaluating students' answers to questions about textbook content.

---Goal---

Evaluate the correctness of the given answer to a question. If the answer is incorrect, provide the reason for the incorrectness without revealing the correct answer. Your response should be in JSON format.

---Target response length and format---

{response_type}

Your response should be a JSON object with the following structure:
{{
    "question": "The original question",
    "correct": true/false,
    "reason": "Explanation of why the answer is incorrect (if applicable)"
}}

---Textbook Content---

{context_data}

---Additional Guidance---

1. Carefully consider the question and the given answer.
2. If the answer is correct, set "correct" to true and leave "reason" as an empty string.
3. If the answer is incorrect, set "correct" to false and provide a clear explanation in the "reason" field without revealing the correct answer.
4. Base your evaluation on the provided textbook content and your general knowledge.
5. If the question or answer is unclear, state this in the "reason" field.
6. Ensure your response is always in valid JSON format.

Remember, your goal is to evaluate the answer's correctness and provide helpful feedback without giving away the correct answer if the student's response is incorrect.
        '''
        self.graphrag = graph
        self.api_key = api_key

    async def process_csv(self, rows):

        tasks = []

        for row in rows:
            question = row['Knowledge Point']
            given_answer = row['Answer']
            options = [row['Option A'], row['Option B'], row['Option C'], row['Option D']]
            input_answer = options[ord(given_answer) - ord('A')]
            query = f"question is: {question}, options are: {options}, and answer is: {input_answer}"
            tasks.append(query)

        results = await asyncio.gather(*[self.eval_question(task) for task in tasks])

        return results

    async def eval_question(self, query):
        result = await self.graphrag.aquery(query, system_prompt=self.prompt)
        return json.loads(result.response)

    async def summary_results_and_feedback(self, results):
        total_questions = len(results)
        correct_answers = sum(1 for result in results if result['correct'])
        incorrect_answers = total_questions - correct_answers
        score = (correct_answers / total_questions) * 100

        incorrect_questions = [result for result in results if not result['correct']]

        query = f'''
           You are an AI tutor providing feedback to a student based on their test results. Here's a summary of their performance:

           Total questions: {total_questions}
           Correct answers: {correct_answers}
           Incorrect answers: {incorrect_answers}
           Score: {score:.2f}%

           Incorrect questions and reasons:
           {json.dumps(incorrect_questions, indent=2)}

           Based on this information, provide a constructive feedback summary for the student. Include:
           1. An overall assessment of their performance
           2. Specific areas where they need improvement, based on the incorrect answers
           3. General advice on how to strengthen their understanding of the subject matter
           4. Encouragement for further study and improvement

           Your response should be friendly, supportive, and motivating while also being clear about areas that need work.
           '''

        feedback = await self.acall_llm(query)
        return feedback

    async def acall_llm(self, query, model='gpt-4o-mini'):
        client = AsyncOpenAI(
            # This is the default and can be omitted
            api_key=self.api_key,
        )

        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful AI tutor providing feedback on test results."},
                {"role": "user", "content": query}
            ],
            model=model,
        )

        return response.choices[0].message.content
