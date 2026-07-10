from langchain_core.prompts import PromptTemplate



qa_generation_prompt = PromptTemplate.from_template(
      """You are creating an evaluation dataset for an educational video assistant.

  Given this transcript segment from a lecture video:
  {content}

  Generate {n} question-answer pairs where:
  - The question is something a student would genuinely ask
  - The answer must be completely answerable using only the information contained in this segment.
  - Do not generate questions that require information from previous or subsequent transcript segments.
  - Questions should test understanding, not verbatim recall
  - Mix types: conceptual ("what is X"), factual ("what did the speaker say about X"), and explanatory ("how does X work")
  """
  )

judge_prompt = PromptTemplate.from_template(
   """You are evaluating an AI tutor's answer for an educational video assistant.

    Question: {question}

    Reference Answer (from transcript): {reference_answer}

    Source Transcript (what was retrieved): {content}

    Generated Answer: {generated_answer}

    Rate the Generated Answer on each dimension from 1 to 5:
    - correctness: Is the information factually accurate compared to the reference?
    - completeness: Does it cover the key points from the reference?
    - faithfulness: Does it avoid adding false information not in the source?
    - clarity: Is it clearly explained for a student?
    """
    )

multi_query_prompt=PromptTemplate.from_template(
    """
    You are helping retrieve video transcript chunks.

    Generate 2 alternative queries that may use
    different terminology but ask the same thing.

    Question:
    {question}

    Return one query per line.
    """
  )