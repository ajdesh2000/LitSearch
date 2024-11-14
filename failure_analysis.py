import csv
import json
from datasets import load_dataset
import numpy as np
import torch

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk
import tqdm
from itertools import chain

from eval.retrieval.grit import GRIT

from eval.retrieval.grit import GRIT
from eval.retrieval.kv_store import TextType

from utils import utils
from eval.reranking.rerank import permutation_pipeline

# Download NLTK data files for tokenization (if not already downloaded)
nltk.download('punkt')

def calculate_bm25(query, documents):
    # Tokenize the documents and the query
    tokenized_corpus = [word_tokenize(document) for document in documents]
    tokenized_query = word_tokenize(query)

    # Initialize BM25 model
    bm25 = BM25Okapi(tokenized_corpus)

    # Get BM25 score
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Return the score for the first (and only) document
    return bm25_scores

gl=GRIT('i1', "Given a research query, retrieve the passage from the relevant research paper")
model=utils.get_gpt4_model("gpt-4o-mini", azure=True)
def calculate_grit(query, documents):
    # ks=gl._encode_batch(documents,TextType.KEY)
    ks=gl._model.encode(documents, batch_size=64, instruction=gl._get_instruction(TextType.KEY), show_progress_bar=True, convert_to_tensor=True)
    # qs=gl._encode_batch(query,TextType.QUERY, convert_to)
    qs=gl._model.encode([query], batch_size=64, instruction=gl._get_instruction(TextType.QUERY), show_progress_bar=True, convert_to_tensor=True)
    cs=torch.nn.functional.cosine_similarity(qs, ks)

    top_indices = cs.argsort().cpu().numpy()
    
    # Return the score for the first (and only) document
    return top_indices


def calculate_gpt(query, documents):
    # ks=gl._encode_batch(documents,TextType.KEY)
    res=permutation_pipeline(model, {"query":query,"documents":documents}, rank_start=0, rank_end=len("documents"), index_type="full_paper") 
    return [d["corpusid"] for d in res["documents"]]


corpus_clean_data = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")

id_dict = {r["corpusid"]:(r["title"],r["abstract"]) for r in corpus_clean_data}

id_dict2 = {r["corpusid"]:i for i,r in enumerate(corpus_clean_data)}

with open("results/reranking/LitSearch.title_abstract.grit.reranked.jsonl","r") as f:
    all_lines=f.read().split("\n")
import re
global_cache={}
def parse_paper(id):
    if id in global_cache:
        return global_cache[id]
    text=corpus_clean_data[id_dict2[id]]["full_paper"]
    # Define the main sections with common variations
    sections = {
        "Abstract": ["Abstract"],
        "Introduction": ["Introduction"],
        "Related Work": ["Related Work", "Literature Review"],
        "Background": ["Background", "Preliminaries"],
        "Problem Formulation": ["Problem Formulation", "Problem Statement"],
        "Methodology": ["Methodology", "Proposed Approach", "Method"],
        "Experiments": ["Experiments", "Experimental Setup"],
        "Results": ["Results", "Results and Analysis", "Evaluation"],
        "Ablation Studies": ["Ablation Studies", "Ablation Analysis"],
        "Discussion": ["Discussion", "Analysis"],
        "Conclusion": ["Conclusion"],
        "Future Work": ["Future Work", "Future Directions"],
        "Ethical Considerations": ["Ethical Considerations", "Broader Impact"],
        "References": ["References", "Bibliography"]
    }

    # Build a mapping from alternative section titles to standardized section names
    section_map = {alt.lower(): main for main, alts in sections.items() for alt in alts}

    # Build a flexible pattern for section titles
    section_pattern = '|'.join([
        rf"(?i)^\s*(\d+\.\s*)?({alt})\s*[:\-]?\s*$" for alt in section_map.keys()
    ])

    # Split text by the flexible section header pattern, keeping titles in the splits
    splits = re.split(f"({section_pattern})", text, flags=re.MULTILINE)

    # Debugging: Print the splits to see the structure
    # print(f"Splits: {splits}")

    # Initialize the dictionary to store parsed sections
    parsed_sections = {}
    current_section = None

    # Check if the split result is valid and process each split to assign text to the correct section
    if splits:
        for split in splits:
            # Ensure split is neither None nor empty before processing
            if split is not None and split.strip():
                split = split.strip()
                normalized_split = re.sub(r"^\d+\.\s*", "", split.lower())  # Remove leading numbering for matching

                if normalized_split in section_map:
                    # Set the current section to the standardized name
                    current_section = section_map[normalized_split]
                    parsed_sections[current_section] = ""  # Initialize the section in the dictionary
                    # print(f"Matched Section: {current_section}")
                elif current_section:
                    # Add content to the current section
                    parsed_sections[current_section] += split + " "
            else:
                # print(f"Skipping empty or None split: {split}")
                pass

    # Clean up extra whitespace from sections
    parsed_sections = {section: content.strip() for section, content in parsed_sections.items()}

    if "Introduction" not in parsed_sections:
        parsed_sections["Introduction"]=""
    if "Conclusion" not in parsed_sections:
        parsed_sections["Conclusion"]=""

    global_cache[id]=parsed_sections

    return parsed_sections

alljds=[]
for line in tqdm.tqdm(all_lines):
    if len(line)==0:
        continue
    if "Which paper found that mutual learning benefits multlingual models?" not in line:
        continue
    jd=json.loads(line)    
    n=100
    top_retrieved = jd["retrieved"][:n]
    # sorted_indices = np.argsort([-len(corpus_clean_data[id_dict2[r]]["citations"]) for r in top_retrieved])
    sections={}
    # chunks=[[corpus_clean_data[id_dict2[r]]["full_paper"][i:i+n] for i in range(0, len(corpus_clean_data[id_dict2[r]]["full_paper"]), n)] for r in top_retrieved]
    flattened_list = [corpus_clean_data[id_dict2[r]]["abstract"]+parse_paper(r)["Introduction"]+parse_paper(r)["Conclusion"] for r in jd['retrieved']]
    # flattened_list2 = [{"content":corpus_clean_data[id_dict2[r]]["abstract"]+parse_paper(r)["Introduction"]+parse_paper(r)["Conclusion"],"corpusid":r} for r in jd['retrieved']]
    sorted_indices=calculate_grit(jd["query"],flattened_list)
    
    # sorted_indices = np.argsort(bm25_scores)
    # sorted_indices=np.argsort(calculate_bm25(jd["query"],[corpus_clean_data[id_dict2[r]]["full_paper"] for r in top_retrieved]))

    # Sort jd["retrieved"] based on these indices
    jd["retrieved"] = [top_retrieved[i] for i in reversed(list(sorted_indices))]

    alljds.append(jd)


all_scores={}
recalls=[]
for jd in alljds:
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
