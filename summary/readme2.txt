read me

main code file name: clustering_summary.py
Inputs: clustering.py [dataset_file_name.txt] [num_of_words_in_summary]
Outputs: summary
Additional file: lexrankfinal.py , please keep both the python files in the same directory

In the main file, lines 472-483 are added. I have written the main summarisation code(graph generation, ranking, summary generation) in a separate file "lexrankfinal.py". No additional libraries are used except for TFIDF vectors.
def gen_lexrank_summary(orig_sents, max_words,docs_clean,upto) is the function that is called from the main file.


SUMMARISATION
def sim_adj_matrix(sents, min_sim=MIN_LEXPAGERANK_SIM):
def normalize_matrix(matrix):
def pagerank(matrix, d=0.85):
def has_converged(x, y, epsilon=EPSILON):
def gen_summary_from_rankings(score, clean_sents, tok_sents, orig_sents, max_words):
def gen_lexrank_summary(orig_sents, max_words,docs_clean,upto):