"""Answer quality evaluation."""

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel


class JudgeScore(BaseModel):
    correctness: int
    completeness: int
    faithfulness: int
    clarity: int
    reasoning: str


def judge_answer(question,reference_answer,generated_answer,content,llm):

    judge_prompt = PromptTemplate(
    template="""You are evaluating an AI tutor's answer for an educational video assistant.

    Question: {question}

    Reference Answer (from transcript): {reference_answer}

    Source Transcript (what was retrieved): {content}

    Generated Answer: {generated_answer}

    Rate the Generated Answer on each dimension from 1 to 5:
    - correctness: Is the information factually accurate compared to the reference?
    - completeness: Does it cover the key points from the reference?
    - faithfulness: Does it avoid adding false information not in the source?
    - clarity: Is it clearly explained for a student?
    """,
      input_variables=[
        "question",
        "reference_answer",
        "generated_answer",
        "content"]
    )

    judge_llm=llm.with_structured_output(JudgeScore)
    chain=judge_prompt|judge_llm
    response=chain.invoke({
       "question": question,
       "reference_answer": reference_answer,
       "generated_answer": generated_answer,
       "content": content
    })

    return response.model_dump()


def evaluate_answers(qa_pairs, chat_fn, llm):

    all_scores = []
    failures = []

    for qa in qa_pairs:
        try:
            result = chat_fn(qa["question"])
            generated = result.response

            scores = judge_answer(
                question=qa["question"],
                reference_answer=qa["reference_answer"],
                generated_answer=generated,
                content=qa["content"],
                llm=llm
            )

            scores["question"] = qa["question"]
            scores["source"] = result.source
            all_scores.append(scores)

        except Exception as e:
            failures.append({"question": qa["question"], "error": str(e)})
            print(f"Error evaluating question: {qa['question']}")
            print(e)

    # Aggregate
    dims = ["correctness", "completeness", "faithfulness", "clarity"]
    avg_scores = {d: sum(s[d] for s in all_scores) / len(all_scores) for d in dims}
    avg_scores["overall"] = sum(avg_scores.values()) / len(dims)

    print(f"\nAnswer Quality Results ({len(all_scores)} evaluated, {len(failures)} failed)")
    for dim, score in avg_scores.items():
        print(f"  {dim}: {score:.2f} / 5.0")

    return all_scores, avg_scores, failures
