from nltk.corpus import stopwords 
import string
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from math import *
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer, WordPunctTokenizer
import sys
import string
from collections import OrderedDict

# import merge as mg
import re
import itertools
from functools import cmp_to_key

#global variables
cluster_list = OrderedDict()
global_predicted_labels = []

def cosine_similarity_metric(a1, a2):
	return cosine_similarity(np.asarray(a1).reshape(1,-1), a2.reshape(1,-1))[0][0]			#This is a 1x1 array, we get the float value from it through [0][0] indexing

def clean(doc):
	""" Removing Punctuation, Lemmatizing, Tokenization for text data/ doc. Arg: doc -> string"""
	exclude = set(string.punctuation) 
	lmtzr = WordNetLemmatizer()
	# tokens = WordPunctTokenizer().tokenize(doc)
	# clean = [token.lower() for token in tokens if token.lower()]
	tokenise = [i for i in doc.lower().split() ]

	punc_free = []

	for word in tokenise:
		#since fullstop and decimal are represented as samee '.'
		# prevent from confusing decimal of float as fullstop,
		if re.match("^\d+?\.\d+?$", word) is not None:
			punc_free.append(word)
		else:
			#removing punctuations
			tmp = ''.join(ch for ch in word if ch not in exclude)
			punc_free.append(tmp)

	# punc_free = ''.join(ch for ch in tokenize if ch not in exclude)
	final = [lmtzr.lemmatize(word, pos = 'v') for word in punc_free]
	final_doc = " ".join(final)
	# for s in final:
	# 	final_doc += s + " "
	return final_doc

def get_docs(fread):
	output_docs = [doc for doc in fread]
	return output_docs
def clean_data(fread):
	# fread = open('sandy_hook_TWEB_FACT_0.txt', 'r')
	# output_clean_docs = [clean(doc.split("\t")[3]) for doc in fread] 
	output_clean_docs = [clean(doc) for doc in fread] 
	# for doc in fread:
	# 	print(doc)
	# 	break
	# for i, doc in enumerate(output_clean_docs[0:100]):
	# 	print (i+1,doc)
	# sys.exit()
	# print(output_clean_docs[0:50])
	# output_clean_docs = [clean(doc).split() for doc in fread]  
	# sys.exit()
	return output_clean_docs

def gen_tfidf_matrix(cleaned_doc_list, sample_num):
	"""Generate and return tf-idf vector array from list of textual documents"""
	feature_names = []
	vectorizer = TfidfVectorizer(min_df=1, stop_words = 'english')
	Y = vectorizer.fit_transform(cleaned_doc_list[:sample_num])
	feature_names = vectorizer.get_feature_names()
	Y = Y.toarray()
	return Y, feature_names

def gen_tfidf_vector(cleaned_doc_list,curr_sample_num):
	feature_names = []
	vectorizer = TfidfVectorizer(min_df=1, stop_words = 'english')
	vectorizer.fit(cleaned_doc_list[:curr_sample_num])
	feature_names = vectorizer.get_feature_names()
	Y = vectorizer.transform([cleaned_doc_list[curr_sample_num]])
	Y = Y.toarray()
	return Y, feature_names

# def gen_tfidf_vector_fulldataset(cleaned_doc_list,curr_sample_num):
# 	#DIFFERENT FROM text_static
# 	feature_names = []
# 	vectorizer = TfidfVectorizer(min_df=1, stop_words = 'english')
# 	vectorizer.fit(cleaned_doc_list)
# 	feature_names = vectorizer.get_feature_names()
# 	Y = vectorizer.transform([cleaned_doc_list[curr_sample_num]])
# 	Y = Y.toarray()
# 	return Y

def get_focus_set(cluster, focus_num,docs_tfidf):
	dist = []
	pts = cluster.points_indices
	# print("len of docs")
	# print(len(docs_tfidf))
	# print("points: ")
	# print(pts)
	for pt in pts:
		pt_tfidf = docs_tfidf[pt]
		dist.append((pt,cluster.similarity(pt_tfidf)))
	# print("cosine dist before", dist)
	dist = sorted(dist, key=lambda x: x[1],reverse=True)
	# print("cosine dist before", dist)
	focus_set_indices = [x[0] for x in dist[:focus_num]]
	return focus_set_indices

class cluster_text:
	#MAIN CLUSTER CLASS
	# num_of_clusters = 0

	def __init__(self, num_points, centroid, data_point_indices_list,tfidf_feature_names):
		self.num_points=num_points
		self.points_indices = data_point_indices_list
		# cluster_text.num_of_clusters += 1
		self.tfidf_features = tfidf_feature_names
		self.sum_of_points = [x for x in centroid]
		self.centroid = [x for x in centroid]


	def similarity(self,data_point):
		# print("data_point point centroid: ", data_point)
		# print("data_point centroid type", type(data_point))
		# print("self centroid: ", self.centroid)
		# print("self centroid type", type(self.centroid))
		return cosine_similarity_metric(self.centroid, data_point)

	def add_point(self, data_point, data_point_idx):
		self.points_indices.append(data_point_idx)
		self.num_points += 1

		# print ("sumofpoints: ")
		# print( self.sum_of_points)
		# print ("len of sum of points", len(self.sum_of_points))
		# print ("point: ")
		# print(data_point)
		# print ("len point: ", len(data_point))
		self.sum_of_points = [sum(x) for x in zip(self.sum_of_points, data_point)]
		self.centroid = [x/float(self.num_points) for x in self.sum_of_points] 
	def update_tfidf(self,new_tfidf_features):
		set_old = set(self.tfidf_features)
		set_new = set(new_tfidf_features)

		new_features = set_new - set_old
		new_features = sorted(new_features)
		# print(new_features)
		for item in new_features:
			self.centroid.insert(new_tfidf_features.index(item),0)
			self.sum_of_points.insert(new_tfidf_features.index(item),0)
		self.tfidf_features = new_tfidf_features


def cosine_similarity_metric2(a1, a2):
	return cosine_similarity(np.asarray(a1).reshape(1,-1), np.asarray(a2).reshape(1,-1))[0][0]

def compare_cluster_tuples(t1, t2):
	""" Used for generating sorted combinations of cluster tuples in the merge algorithm"""
	# print("t1 centroid:",t1[0].centroid)
	# print("t1 centroid type", type(t1[0].centroid))
	sim_t1 = cosine_similarity_metric2(t1[0].centroid,t1[1].centroid)
	sim_t2 = cosine_similarity_metric2(t2[0].centroid,t2[1].centroid)
	# sys.exit()

	if sim_t1 > sim_t2:
		return -1
	elif sim_t1 < sim_t2:
		return 1
	else:
		return 0

# def com_farzi(t1,t2):
# 	sim_t1 = t1[1]-t1[0]
# 	sim_t2 = t2[1]-t2[0]

# 	if sim_t1 > sim_t2:
# 		return 1
# 	elif sim_t1 < sim_t2:
# 		return -1
# 	else:
# 		return 0

def aggregate(sibling_cluster_list):
	""" aggregating the clusters to merged and returns the merged cluster"""
	num_points = 0
	points_indices = []
	tfidf_features = []
	sum_of_points = []
	centroid = []
	for cidx, c in enumerate(sibling_cluster_list):

		if cidx==0:
			num_points = c.num_points
			points_indices = c.points_indices
			tfidf_features = c.tfidf_features
			sum_of_points = c.sum_of_points
			centroid = c.centroid
		else:
			num_points += c.num_points
			points_indices.extend(c.points_indices)
			sum_of_points = [sum(x) for x in zip(sum_of_points, c.sum_of_points)]
			centroid = [x/float(num_points) for x in sum_of_points] 

	parent_cluster = cluster_text(num_points, centroid, points_indices, tfidf_features)
	return parent_cluster

def get_cluster_key(cluster_list, cluster):
	for key in cluster_list.keys():
		if(cluster_list[key] == cluster):
			return key

def cluster_popular_words(cluster,k):
	"""return popular words in the cluster centroid"""
	popular_words_tuple_list = [(x,y) for (y,x) in sorted(zip(cluster.centroid,cluster.tfidf_features),reverse=True)]
	popular_word_list = [x for (x,y) in popular_words_tuple_list if y>0 ]
	# for t in range(0,len(cluster.centroid)):
	# 	print(cluster.tfidf_features[t],cluster.centroid[t])
	# for t in popular_words_tuple_list:
	# 	print(t[0],t[1])
	if(len(popular_word_list) <= k ):
		return popular_word_list
	else:
		return popular_word_list[:k]

def merge(cluster_list, num_clusters_to_remove):
	"""main merging algorithm when number of clusters become grater than threshold"""
	combi_list = list(itertools.combinations(cluster_list.values(),2))
	# combi_list.sort(cmp=com_farzi)
	# sorted_combi_list= combi_list
	sorted_combi_list = sorted(combi_list, key=cmp_to_key(compare_cluster_tuples))
	composite_clusters = OrderedDict()
	composite_parent = {}
	# print("Sorted Combi list:-")
	# print(sorted_combi_list)
	removed_so_far =0
	parent_key=0

	for t in sorted_combi_list:
		if removed_so_far >= num_clusters_to_remove:
			break

		if (t[0] not in composite_parent) and (t[1] not in composite_parent):
			temp = [t[0], t[1]]
			composite_clusters[parent_key] = temp
			composite_parent[t[0]] = parent_key
			composite_parent[t[1]] = parent_key
			parent_key+=1
			removed_so_far+=1

		elif (t[0] in composite_parent) and (t[1] not in composite_parent):
			parent = composite_parent[t[0]]
			composite_clusters[parent].append(t[1])
			composite_parent[t[1]] = parent
			removed_so_far+=1

		elif (t[0] not in composite_parent) and (t[1] in composite_parent):
			parent = composite_parent[t[1]]
			composite_clusters[parent].append(t[0])
			composite_parent[t[0]] = parent
			removed_so_far+=1

		elif composite_parent[t[0]]!= composite_parent[t[1]]:
			# both t1 and t2 have parents , but different membership
			parent_t0 = composite_parent[t[0]]
			parent_t1 = composite_parent[t[1]]
			# print("cluster par 1:" , len(composite_clusters[parent_t1]))
			# print("cluster par 1:" , len(composite_clusters[parent_t0]))
			siblings_t1 = composite_clusters[parent_t1]
			# print("siblings of t1 len", len(siblings_t1))
			composite_clusters.pop(parent_t1,None)
			composite_clusters[parent_t0].extend(siblings_t1)
			for elem in siblings_t1:
				composite_parent[elem]=parent_t0
			removed_so_far+=1
			# print("cluster par final:" , len(composite_clusters[parent_t0]))

	# print("Composite Clusters:-")
	# print(composite_clusters)

	# print("Following clusters have been merged:")
	for parent_idx in composite_clusters:
		# print("\n\n\n")
		sibling_clusters = composite_clusters[parent_idx]
		parent_cluster = aggregate(sibling_clusters)
		last_idx = list(cluster_list.keys())[-1]
		cidx = last_idx+1
		cluster_list[cidx] = parent_cluster
		for cluster in sibling_clusters:
			cluster_key = get_cluster_key(cluster_list,cluster)
			# print(cluster.centroid)
			# print(cluster.tfidf_features)
			# print("\n")
			# print("Child cluster popular words :", cluster_popular_words(cluster,len(cluster.tfidf_features)))
			cluster_list.pop(cluster_key,None)
		for doc_idx in parent_cluster.points_indices:
			global_predicted_labels[doc_idx] = cidx
		# print("Aggregate Cluster Now is:- ")
		# print(parent_cluster.centroid)
		# print(parent_cluster.tfidf_features)
		# print("Parent cluster popular words :", cluster_popular_words(parent_cluster,len(parent_cluster.tfidf_features)))


def clustering(start_idx, data_points, data_points_tfidf_features):
	# returns list of predicted labels for each datapoint and list of clusters 
	threshold = 0.7
	# local_predicted_label_list = []
	#document number to adding a point to a cluster: start_idx+idx
	for idx,point in enumerate(data_points):
		# if cluster_text.num_of_clusters == 0:
		if len(cluster_list) == 0:
			# print ("Initial Cluster")
			first_cluster = cluster_text(1,point, [start_idx+idx], data_points_tfidf_features)
			# cluster_list.append(first_cluster)					# Maintaining list of all clusters	 
			cluster_list[0] = first_cluster	
			# local_predicted_label_list.append(0)	
			global_predicted_labels.append(0)
			#TAKE CARE ABOUT POSITION WHERE TO ADD MERGE ST. 'idx' doesnot get skipped
		else:
			max_sim = -1		
			max_sim_cluster_idx = 0
			for cidx in cluster_list.keys():
				#better comparison is of len(cluster.centroid) < len(point)
				if(len(cluster_list[cidx].centroid)< len(point)):
					# print("updated")
					cluster_list[cidx].update_tfidf(data_points_tfidf_features)

			for (cidx, cluster) in cluster_list.items():
				# print("current feats in cluster: ", len(point),len(cluster.sum_of_points),len(cluster.centroid),len(cluster.tfidf_features))
				
				# 	print("point centroid: ", point)
				# 	print("point centroid type", type(point))
				sim = cluster.similarity(point)
				if(sim>max_sim):
					max_sim=sim
					max_sim_cluster_idx=cidx
			
			if(max_sim==-1):
				print("Error in finding maximum similarity of the current document/tweet with the existing clusters")
				print("Adding this document in Cluster 0")
			
			if max_sim>=threshold:
				cluster_list[max_sim_cluster_idx].add_point(point,start_idx+idx)
				# local_predicted_label_list.append(max_sim_cluster_idx)
				global_predicted_labels.append(max_sim_cluster_idx)

			else:
				# print ("Adding new text cluster")
				new_cluster = cluster_text(1,point,[start_idx+idx], data_points_tfidf_features)
				last_idx = list(cluster_list.keys())[-1]
				cidx = last_idx+1
				cluster_list[cidx] = new_cluster
				# cluster_list.append(new_cluster)
				# local_predicted_label_list.append(cidx)
				global_predicted_labels.append(cidx)
		
		#if NUMBER OF CLUSTERS BECOME GREATER THAN PERMISSIBLE LIMIT, MERGE HERE 
		if(len(cluster_list)>=max_clusters):
			num_of_clusters_to_remove = int(max_clusters*(1-merge_coeff))
			# print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
			# print("Merging CLUSTERS. Present cluster num: ", len(cluster_list))
			merge(cluster_list,num_of_clusters_to_remove)
			# print("Merge finish, Now clusterNum: ", len(cluster_list))
			# print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
	# return local_predicted_label_list

def ranked_cluster_list(cluster_list):
	cluster_dictionary_to_list = list(cluster_list.values())
	rcl = sorted(cluster_dictionary_to_list, key =lambda x: x.num_points, reverse = True)
	return rcl


#GLOBAL VARIABLES 2
max_clusters = 150
merge_coeff = 0.7
fread = open(sys.argv[1], 'r')
docs = get_docs(fread)
docs_clean = clean_data(docs)
sample_num = 100
#first cluster sample_num tweets and then cluster incoming tweets in incremental online fashion

#DIFFERENT FROM text_static
tfidf_matrix, tfidf_feature_names = gen_tfidf_matrix(docs_clean,sample_num)
# print ("Number of vocabulary words")
# print(len(tfidf_feature_names))
# print("Number of tfidf vectors")
# print(len(tfidf_matrix))

#DIFFERENT FROM text_static
# global_predicted_labels = clustering(tfidf_matrix, tfidf_feature_names)
starting_doc_idx =0
clustering(starting_doc_idx,tfidf_matrix, tfidf_feature_names)
# print("number of clusters")
# print(len(cluster_list))
# print("predicted labels")
# print("number of predicted labels: ", len(global_predicted_labels))


#incremental code / online updating begins here
upto = len(docs_clean)
# upto = 101
for i in range(sample_num,upto):
	# print("clustering document num: ", i)
	#generates feature vector of the current document based on the TF-IDF model of 0-(i-1) documents
	tfidf, tfidf_feature_names = gen_tfidf_vector(docs_clean,i)
	# print ("Number of features: ", len(tfidf_feature_names))
	# print("Number of tfidf vectors: ", len(tfidf))
	# predicted_label = clustering(tfidf, tfidf_feature_names)
	clustering(i,tfidf, tfidf_feature_names)
	# global_predicted_labels.append(predicted_label)
	# print("Document number %d belongs to cluster number %d: ",i, predicted_label)
	# print("Document number %d belongs to cluster number %d " %(i , global_predicted_labels[i]))

print("TOTAL CLUSTER COUNT: ", len(cluster_list))
# print("****************TEST BEGINS***************")
# for (key,cluster) in cluster_list.items():
# 	print("\n\n\n")
# 	print("info for cluster_num:" , key)
# 	# print("cluster sr num")
# 	print("cluster words", cluster_popular_words(cluster,len(cluster.tfidf_features)))
# 	for doc_idx in cluster.points_indices:
		
# 		print("doc: ", docs[doc_idx].split("\t")[3])

# 		print("Doc Index,Predicted Label,Current Cluster Number, Truth for this cluster is: ", doc_idx, global_predicted_labels[doc_idx], key, global_predicted_labels[doc_idx] == key )

# print("****************TEST ENDS***************")

# FOCUS SET
num_focus = 40
focus_set_dict = OrderedDict()
# print("docs clean len")
# print(len(docs_clean))
docs_tfidf, garbage = gen_tfidf_matrix(docs_clean, upto)
# print("docs tfidf len")
# print(len(docs_tfidf))
# print(docs_tfidf[0])
# print(docs_tfidf[1])
for c_key,cluster in cluster_list.items():
	fset_indices = get_focus_set(cluster, num_focus,docs_tfidf)
	focus_set_dict[c_key] = fset_indices


ranked_cluster_list = ranked_cluster_list(cluster_list)
# num_of_top_clusters = 10
num_of_top_clusters = int(sys.argv[2])
num_of_words_per_cluster =int(sys.argv[3])
# num_of_top_clusters = sys.argv[3]

# print("TOTAL CLUSTER COUNT: ", cluster_text.num_of_clusters)
print("Showing Top %d Clusters" %(num_of_top_clusters))
for idx, cluster in enumerate(ranked_cluster_list):
	if(idx>=(num_of_top_clusters)):
		break
	clust_key = get_cluster_key(cluster_list,cluster)
	pop_word_list = cluster_popular_words(cluster, num_of_words_per_cluster)
	# pop_word_list = cluster_popular_words(cluster, len(cluster.tfidf_features))
	#DIFFERENT FROM text_static
	print("Cluster with rank: ", idx)
	print("%d popular words" %(num_of_words_per_cluster))
	# print("%d popular words" %(len(cluster.tfidf_features)))
	print(pop_word_list)
	# print("Number of data points in cluster: ", cluster.num_points)
	# print("FOCUS SET")
	# for f_idx in focus_set_dict[clust_key]:
	# 	print(docs_clean[f_idx])

# summ = []
# for key, cluster in cluster_list.items():
# 	f_indices = focus_set_dict[key]
# 	for f_idx in f_indices:
# 		summ.append(docs_clean[f_idx])

# import lexrank as lx
# ans = lx.gen_lexrank_summary(summ, 250)
# print(ans)


