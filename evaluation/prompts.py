from langchain_core.prompts import PromptTemplate



general_qa_generation_prompt = PromptTemplate.from_template(
  """
  You are creating an evaluation dataset for an educational video assistant.
  Transcript segment: {content}

  Generate between 0 to  {n} question-answer pairs where:
  - If the chunk contains very little or waste information, return 1 or 0 question-answer pairs.
  - Mix factual, why, comparison, reasoning and application questions whenever possible.
  - Aim approximately for below distribution :
      40%/ factual recall
      30%/ reasoning ("why"/"how")
      20%/ application
      10%/ comparison (only if naturally supported)
  - Do not copy sentences directly from the transcript.
  - Questions should require understanding instead of simple keyword matching.
  - Do not use outside knowledge.
  - Assign each question a difficulty level:
    - easy
    - medium
    - hard
  """
  )

misconception_qa_generation_prompt = PromptTemplate.from_template(
  """
  You are creating misconception-based evaluation questions for an  educational video assistant.
  Transcript segment: {content}

  Generate between 0 to  {n} question-answer pairs where:
  - If the chunk contains very little or waste information, return 1 or 0 question-answer pairs.
  - Create a realistic but incorrect student misconception based only on the transcript.
  - Ask whether the student's statement is correct.
  - The answer should explain why the statement is wrong and provide the correct concept.
  - Do not invent misconceptions unrelated to the transcript.
  - Assign each question a difficulty level:
    - easy
    - medium
    - hard
  """
)

judge_prompt = PromptTemplate.from_template(
  """
  You are a strict evaluator grading an AI tutor's answer for an educational video assistant.
  Do not default to high scores — use the full 1-5 range based on the criteria below.

  Question: {question}

  Reference Answer (gold standard from the instructor): {reference_answer}

  Source Transcript (ground truth for what was actually said in the video): {content}

  Generated Answer (to be evaluated): {generated_answer}

  Evaluate the generated answer.

  Then score each dimension from 1 to 5, using these anchors:

  correctness (compare against Reference Answer):
    5 = Every factual statement matches the reference.
    4 = One minor factual error.
    3 = Partially correct but missing or incorrect important facts.
    2 = Mostly incorrect.
    1 = Completely incorrect.

  completeness (compare against Reference Answer only):
    5 = covers all key points in the Reference Answer
    3 = covers the main point but misses a secondary key point
    1 = misses the main point entirely
    Note: the Generated Answer may correctly include extra accurate detail drawn from the
    Source Transcript beyond what the Reference Answer states — do not penalize that as
    incompleteness, only flag actual missing key points.

  faithfulness (compare against Source Transcript only, NOT the Reference Answer):
    5 = every claim in the Generated Answer is directly supported by the Source Transcript
    3 = contains one unsupported claim or a claim that overreaches the transcript
    1 = contains fabricated information not present in the Source Transcript at all

  clarity (independent of source):
    5 = clear, well-structured, easy for a student to follow
    3 = understandable but awkward, verbose, or poorly organized
    1 = confusing or incoherent

  Be strict.
  Do not give 5 unless the criterion is fully satisfied.
""")


multi_query_prompt=PromptTemplate.from_template(
    """
    You are helping retrieve video transcript chunks.

    Generate 2 alternative queries that may use
    different terminology but ask the same thing.

    Question: {question}

    Return one query per line.
    """
  )