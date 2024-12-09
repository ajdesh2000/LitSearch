import json
from collections import defaultdict

from numpy import sqrt

def normalize_by_variance(rank_list):
        """Normalize scores in a rank list using variance."""
        scores = [item[1] for item in rank_list]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        std_dev = sqrt(variance) if variance > 0 else 1  # Avoid division by zero
        return [(item[0], (item[1] - mean_score) / std_dev) for item in rank_list]

def reciprocal_rank_fusion(*list_of_list_ranks_system, K=27):
    """
    Fuse rank from multiple IR systems using Reciprocal Rank Fusion.

    Args:
    * list_of_list_ranks_system: Ranked results from different IR system.
    K (int): A constant used in the RRF formula (default is 60).

    Returns:
    Tuple of list of sorted documents by score and sorted documents
    """
    # Dictionary to store RRF mapping
    rrf_map = defaultdict(float)

    # Calculate RRF score for each result in each list
    for j, rank_list in enumerate(list_of_list_ranks_system):
        if type(rank_list[0]) is tuple:
            min_score=min([i[1] for i in rank_list])
            max_score=max([i[1] for i in rank_list])
            size = max_score-min_score
            size_factor = (1/61-1/261)/size
            rank_list = [(i[0],((i[1]-min_score)*(size_factor))+1/261) for i in rank_list]
        for rank, item in enumerate(rank_list, 1):
            if type(item) is tuple:
                rrf_map[item[0]] += item[1]
                # rrf_map[item[0]] = max(rrf_map[item[0]],item[1])
            else:
                rrf_map[item] += 1 / (rank + K)

    # Sort items based on their RRF scores in descending order
    sorted_items = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)

    # Return tuple of list of sorted documents by score and sorted documents
    return sorted_items, [item for item, score in sorted_items]


def open_files(file):
    with open(file, "r") as f:
        lines = f.read().split("\n")
    return lines


def assert_len(list_files):
    for i in range(1, len(list_files)):
        assert len(list_files[i]) == len(list_files[0])
    return True


def rerankrrf(files_list):
    list_files = []
    for file in files_list:
        list_files.append(open_files(file))
    # check if corpus_id in top k
    if not assert_len(list_files):
        print("Error: Length of files are not equal")

    with open(
        "LitSearch.title_abstract.bm25.ggpt4o.grit.reranked_test.jsonl", "w"
    ) as f:
        pass

    with open(
        "LitSearch.title_abstract.bm25.ggpt4o.grit.reranked_test.jsonl", "a"
    ) as f:

        for i, line in enumerate(list_files[0]):
            rank_list = []
            if len(line) == 0:
                continue
            for j, file in enumerate(list_files):
                jd = json.loads(list_files[j][i])
                if "retrieved_scores" in jd: 
                    rank_list.append(list(zip(jd["retrieved"][:200],jd["retrieved_scores"][:200])))
                else:
                    rank_list.append(jd["retrieved"][:200])
            get_rank = reciprocal_rank_fusion(*rank_list)
            new_dict = {**jd}
            new_dict["retrieved"] = get_rank[1]

            f.write(json.dumps(new_dict) + "\n")


rerankrrf(
    [
        "/Users/ajinkya/Documents/CMU/fall24/anlp/LitSearch/results/reranking/LitSearch.title_abstract.grit.reranked_gptmini.jsonl",
        "/Users/ajinkya/Documents/CMU/fall24/anlp/LitSearch/LitSearch.conclusion.grit.jsonl",
        "/Users/ajinkya/Documents/CMU/fall24/anlp/LitSearch/LitSearch.introduction.grit.jsonl",        
        "grit_scores.jsonl",
    ]
)



with open("LitSearch.title_abstract.bm25.ggpt4o.grit.reranked_test.jsonl","r") as f:
    all_lines=f.read().split("\n")


all_scores={}
recalls=[]
for line in all_lines:
    if len(line)==0:
        continue
    jd=json.loads(line)
    if jd["query_set"]+"_"+str(jd["specificity"]) not in all_scores:
        all_scores[jd["query_set"]+"_"+str(jd["specificity"])]={"recall_5":[],"recall_20":[]}
    k=5
    all_scores[jd["query_set"]+"_"+str(jd["specificity"])]["recall_5"].append(len([ci for ci in jd["corpusids"] if ci in jd["retrieved"][:k]])/len(jd["corpusids"]))
    k=20
    all_scores[jd["query_set"]+"_"+str(jd["specificity"])]["recall_20"].append(len([ci for ci in jd["corpusids"] if ci in jd["retrieved"][:k]])/len(jd["corpusids"]))

# arr=all_scores["inline_acl_0"]["recall_5"]+all_scores["inline_nonacl_0"]["recall_5"]
# print("Broad inline citation recall 5",sum(arr)/len(arr))
arr=all_scores["inline_acl_0"]["recall_20"]+all_scores["inline_nonacl_0"]["recall_20"]
print("Broad inline citation recall 20: ",100*sum(arr)/len(arr))

arr=all_scores["inline_acl_1"]["recall_5"]+all_scores["inline_nonacl_1"]["recall_5"]
print("Specific inline citation recall 5: ",100*sum(arr)/len(arr))
arr=all_scores["inline_acl_1"]["recall_20"]+all_scores["inline_nonacl_1"]["recall_20"]
print("Specific inline citation recall 20: ",100*sum(arr)/len(arr))


# arr=all_scores["manual_acl_0"]["recall_5"]+all_scores["manual_iclr_0"]["recall_5"]
# print("Broad inline citation recall 5",sum(arr)/len(arr))
arr=all_scores["manual_acl_0"]["recall_20"]+all_scores["manual_iclr_0"]["recall_20"]
print("Broad manual citation recall 20: ",100*sum(arr)/len(arr))

arr=all_scores["manual_acl_1"]["recall_5"]+all_scores["manual_iclr_1"]["recall_5"]
print("Specific manual citation recall 5: ",100*sum(arr)/len(arr))
arr=all_scores["manual_acl_1"]["recall_20"]+all_scores["manual_iclr_1"]["recall_20"]
print("Specific manual citation recall 20: ",100*sum(arr)/len(arr))


arr=all_scores["inline_acl_0"]["recall_20"]+all_scores["inline_nonacl_0"]["recall_20"]+all_scores["manual_acl_0"]["recall_20"]+all_scores["manual_iclr_0"]["recall_20"]
print("Broad ALL citation recall 20: ",100*sum(arr)/len(arr))

arr=all_scores["inline_acl_1"]["recall_5"]+all_scores["inline_nonacl_1"]["recall_5"]+all_scores["manual_acl_1"]["recall_5"]+all_scores["manual_iclr_1"]["recall_5"]
print("Specific ALL citation recall 5: ",100*sum(arr)/len(arr))
