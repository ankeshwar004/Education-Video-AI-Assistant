import os
import json
import math
import config
from pathlib import Path
from src.utils import load_json,save_json
from src.logger import get_logger

logger=get_logger(__name__)

retrieval_eval_dir=config.RETRIEVAL_EVAL_RESULTS_DIR
rerank_eval_dir=config.RERANK_EVAL_RESULTS_DIR
qa_pairs_dir=config.QA_PAIRS_DIR


qa_pairs_path=os.path.join(qa_pairs_dir, "all_qa_pairs")
if not os.path.exists(qa_pairs_path):
    logger.error(f"QA pairs directory {qa_pairs_path} does not exist.")

video_ids=[]
for result_dir in (qa_pairs_path, retrieval_eval_dir, rerank_eval_dir):
    if not os.path.exists(result_dir):
        continue
    for result_file in os.listdir(result_dir):
        if result_file.endswith(".json"):
            video_id=Path(result_file).stem
            if video_id not in video_ids:
                video_ids.append(video_id)

logger.info(f"Found {len(video_ids)} videos for evaluation")

qa_details={}
qa_counts={}  

for video_id in video_ids:
    qa_file_path=os.path.join(qa_pairs_path, f"{video_id}.json")
    if not os.path.exists(qa_file_path):
        logger.warning(f"QA pairs for video {video_id} not found at {qa_file_path}.")
        continue

    qa_pairs=load_json(qa_file_path)
    video_qa_details={"difficulty": {}, "question_type": {}, "chunk_id": {}}

    for qa in qa_pairs:
        video_qa_details["difficulty"][qa["difficulty"]]=video_qa_details["difficulty"].get(qa["difficulty"], 0) + 1
        video_qa_details["question_type"][qa["question_type"]]=video_qa_details["question_type"].get(qa["question_type"], 0) + 1
        video_qa_details["chunk_id"][qa["chunk_id"]]=video_qa_details["chunk_id"].get(qa["chunk_id"], 0) + 1

    qa_details[video_id]=video_qa_details
    qa_counts[video_id]=len(qa_pairs)


# Retrieval evaluation results
retrieval_eval_results={}
for video_id in video_ids:
    retrieval_eval_file=os.path.join(retrieval_eval_dir, f"{video_id}.json")
    if not os.path.exists(retrieval_eval_file):
        logger.warning(f"Retrieval evaluation results for {video_id} not found.")
        continue
    retrieval_eval_results[video_id]=load_json(retrieval_eval_file)


# Rerank evaluation results
rerank_eval_results={}
for video_id in video_ids:
    rerank_eval_file=os.path.join(rerank_eval_dir, f"{video_id}.json")
    if not os.path.exists(rerank_eval_file):
        logger.warning(f"Rerank evaluation results for {video_id} not found.")
        continue
    rerank_eval_results[video_id]=load_json(rerank_eval_file)


# ---- Aggregation ----

RETRIEVAL_METHODS=["bm25_only", "vector_only", "ensemble_only", "ensemble_rerank"]
K_VALUES=["1", "3", "5"]


def mean(values):
    return sum(values)/len(values) if values else None


def stdev(values):
    n=len(values)
    mean=sum(values)/n
    varience=sum((x-mean)**2 for x in values)/(n-1) if n>1 else 0
    return math.sqrt(varience)


def macro_average_retrieval(retrieval_eval_results):
    agg={}
    for method in RETRIEVAL_METHODS:
        hit_rates={k: [] for k in K_VALUES}
        mrrs=[]
        for video_id, result in retrieval_eval_results.items():
            if method not in result:
                continue
            for k in K_VALUES:
                hit_rates[k].append(result[method]["hit_rate"][k])
            mrrs.append(result[method]["mrr"])

        if not mrrs:
            logger.warning(f"No videos have results for retrieval method '{method}', skipping.")
            agg[method]=None
            continue

        agg[method]={
            "hit_rate": {
                k: {"mean": mean(v), "std": stdev(v)}
                for k, v in hit_rates.items()
            },
            "mrr": {"mean": mean(mrrs), "std": stdev(mrrs)},
            "n_videos": len(mrrs),
        }
    return agg


def micro_average_retrieval(retrieval_eval_results, qa_counts):
    agg={}
    for method in RETRIEVAL_METHODS:
        total_hits={k: 0 for k in K_VALUES}
        total_n=0
        total_reciprocal_rank=0.0
        for video_id, result in retrieval_eval_results.items():
            if method not in result:
                continue
            n=qa_counts.get(video_id)
            if not n:
                logger.warning(f"No QA count for {video_id}, skipping in micro-average.")
                continue
            for k in K_VALUES:
                total_hits[k] += round(result[method]["hit_rate"][k] * n)
            total_reciprocal_rank += result[method]["mrr"] * n
            total_n += n

        if total_n == 0:
            logger.warning(f"No pairs available for retrieval method '{method}' micro-average, skipping.")
            agg[method]=None
            continue

        agg[method]={
            "hit_rate": {k: total_hits[k] / total_n for k in K_VALUES},
            "mrr": total_reciprocal_rank / total_n,
            "n_pairs": total_n,
        }
    return agg


def macro_average_rerank(rerank_eval_results):

    if not rerank_eval_results:
        logger.warning("No rerank results available for macro-average.")
        return None

    mrr_before, mrr_after=[], []
    hit_before={k: [] for k in K_VALUES}
    hit_after={k: [] for k in K_VALUES}
    improved, worsened, unchanged, avg_movement=[], [], [], []

    for video_id, result in rerank_eval_results.items():
        mrr_before.append(result["mrr_before"])
        mrr_after.append(result["mrr_after"])
        
        for k in K_VALUES:
            hit_before[k].append(result["hit_rate_before"][k])
            hit_after[k].append(result["hit_rate_after"][k])
        rm=result["rank_movement"]
        improved.append(rm["improved"])
        worsened.append(rm["worsened"])
        unchanged.append(rm["unchanged"])
        avg_movement.append(rm["average_movement"])

    return {
        "mrr_before": {"mean": mean(mrr_before), "std": stdev(mrr_before)},
        "mrr_after": {"mean": mean(mrr_after), "std": stdev(mrr_after)},
        "hit_rate_before": {k: mean(v) for k, v in hit_before.items()},
        "hit_rate_after": {k: mean(v) for k, v in hit_after.items()},
        "rank_movement": {
            "improved_total": sum(improved),
            "worsened_total": sum(worsened),
            "unchanged_total": sum(unchanged),
            "average_movement_of_video_means": mean(avg_movement),
        },
        "n_videos": len(mrr_before),
    }


def micro_average_rerank(rerank_eval_results):

    total_n=0
    total_hits_before={k: 0 for k in K_VALUES}
    total_hits_after={k: 0 for k in K_VALUES}
    total_rr_before=0.0
    total_rr_after=0.0
    total_improved=total_worsened=total_unchanged=0
    total_movement=0.0

    total_n_valid=0

    for video_id, result in rerank_eval_results.items():
        rm=result["rank_movement"]
        n_valid=rm["improved"] + rm["worsened"] + rm["unchanged"]
        n=len(result.get("per_question") or []) or n_valid
        if n == 0:
            continue
        for k in K_VALUES:
            total_hits_before[k] += round(result["hit_rate_before"][k] * n)
            total_hits_after[k] += round(result["hit_rate_after"][k] * n)
        total_rr_before += result["mrr_before"] * n
        total_rr_after += result["mrr_after"] * n
        total_improved += rm["improved"]
        total_worsened += rm["worsened"]
        total_unchanged += rm["unchanged"]
        if rm["average_movement"] is not None:
            total_movement += rm["average_movement"] * n_valid
            total_n_valid += n_valid
        total_n += n


    if total_n == 0:
        logger.warning("No pairs available for rerank micro-average.")
        return None

    return {
        "mrr_before": total_rr_before / total_n,
        "mrr_after": total_rr_after / total_n,
        "hit_rate_before": {k: total_hits_before[k] / total_n for k in K_VALUES},
        "hit_rate_after": {k: total_hits_after[k] / total_n for k in K_VALUES},
        "rank_movement": {
            "improved": total_improved,
            "worsened": total_worsened,
            "unchanged": total_unchanged,
            "average_movement": total_movement / total_n_valid if total_n_valid else None,
        },
        "n_pairs": total_n,
    }



all_video_ids=set(qa_details) | set(retrieval_eval_results) | set(rerank_eval_results)

per_video_analysis={}
for video_id in sorted(all_video_ids):
    per_video_analysis[video_id]={
        "qa_count": qa_counts.get(video_id),
        "qa_distribution": qa_details.get(video_id),
        "retrieval": retrieval_eval_results.get(video_id),
        "rerank": rerank_eval_results.get(video_id),
    }



retrieval_summary={
    "macro": macro_average_retrieval(retrieval_eval_results),
    "micro": micro_average_retrieval(retrieval_eval_results, qa_counts),
    "n_videos_evaluated": len(retrieval_eval_results),
}

rerank_summary={
    "macro": macro_average_rerank(rerank_eval_results),
    "micro": micro_average_rerank(rerank_eval_results),
    "n_videos_evaluated": len(rerank_eval_results),
}

retrieval_out_path=os.path.join(config.EVALUATION_DIR, "retrieval_eval_summary.json")
rerank_out_path=os.path.join(config.EVALUATION_DIR, "rerank_eval_summary.json")
per_video_out_path=os.path.join(config.EVALUATION_DIR, "per_video_analysis.json")

save_json(retrieval_summary,retrieval_out_path)
save_json(rerank_summary,rerank_out_path)
save_json(per_video_analysis,per_video_out_path)

logger.info(f"Retrieval summary saved to {retrieval_out_path}")
logger.info(f"Rerank summary saved to {rerank_out_path}")
logger.info(f"Per-video analysis saved to {per_video_out_path}")