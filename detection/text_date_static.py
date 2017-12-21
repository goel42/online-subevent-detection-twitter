import pickle
import datetime
import operator
import datetime 
from similarity_metrics import *
from math import *
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import PorterStemmer, WordPunctTokenizer
import sys
import string
import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords 
import string
import numpy as np
from collections import OrderedDict
import itertools
from functools import cmp_to_key
import csv
import pickle
from datetime import datetime as dt


# A global list to store mapping of tf-idf values to feature names
feature_names = []
# A global list to store tf-idf values for tweets text
tfidf_titles = []
# A global dictionary storing list of tweet texts and datetime corresponding to serial nums
tweets_dict = {}

# def cosine_sim_metric(a1, a2):
# 	return cosine_similarity(np.asarray(a1).reshape(1,-1), a2.reshape(1,-1))[0][0]	

def corpus_list(tweets_dict, feature_index, sample_num):

	string_list = []
	for snum in range(0, sample_num):
		string_list.append(tweets_dict[snum][feature_index])
	return string_list

def clean_doc(doc):
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

	final = [lmtzr.lemmatize(word, pos = 'v') for word in punc_free]
	final_doc = " ".join(final)
	return final_doc

def clean_docs(feature_docs):

	idx = 0
	for doc in feature_docs:
		feature_docs[idx] = clean_doc(doc)
		idx += 1

def get_docs(fread):
	output_docs = [doc for doc in fread]
	return output_docs
# def clean_data(fread):
# 	# fread = open('sandy_hook_TWEB_FACT_0.txt', 'r')
# 	# output_clean_docs = [clean(doc.split("\t")[3]) for doc in fread] 
# 	output_clean_docs = [clean(doc) for doc in fread] 
# 	# for doc in fread:
# 	# 	print(doc)
# 	# 	break
# 	# for i, doc in enumerate(output_clean_docs[0:100]):
# 	# 	print (i+1,doc)
# 	# sys.exit()
# 	# print(output_clean_docs[0:50])
# 	# output_clean_docs = [clean(doc).split() for doc in fread]  
# 	# sys.exit()
# 	return output_clean_docs

def gen_tfidf_array(cleaned_doc_list):
	"""Generate and return tf-idf vector array from list of textual documents"""
	global feature_names
	vectorizer = TfidfVectorizer(min_df=1, stop_words = 'english')
	Y = vectorizer.fit_transform(cleaned_doc_list)
	feature_names = vectorizer.get_feature_names()
	Y = Y.toarray()
	return Y	

def get_days_list(data_dict, date_feature_index):
	base_day = datetime.datetime(1970, 1, 1)
	days_list = []
	for snum in range(0,sample_num):
		"""           #doubt					"""
		# days_list.append((data_dict[snum][date_feature_index] - base_day).days)
		# date = date.replace(tz_info=None)
		curr_date = data_dict[snum][date_feature_index] 
		curr_date = curr_date.replace(tzinfo=None)
		days_list.append((curr_date - base_day).days * 1440 + (curr_date- base_day).seconds )
	return days_list

class final_cluster:
	"""
	This is cluster class for the final clusters generated after binary weighted voting
	Data Members:
	data_points_idx : list of indices of data points in the cluster
	num_clusters : static -> Keeping count of the number of clusters formed
	"""
	num_clusters = 0
	def __init__(self, data_point_idx, data_point):
		self.data_points_idx = [data_point_idx]
		self.num_points = 1
		self.ft_idx = data_point_idx
		self.centroid = [x for x in data_point]
		self.sum_points = [x for x in data_point]
		final_cluster.num_clusters += 1

	def score(self, data_point, weight, pred_lists):
		score = 0.0
		count = 0
		for idx in self.data_points_idx:
			count += len(pred_lists)
			for feature_index in range(0, len(pred_lists)):  		
				if pred_lists[feature_index][data_point] == pred_lists[feature_index][idx]:
					score += weight[feature_index]
		return score/float(count)

	def add_point(self, data_point_idx, data_point):
		self.num_points += 1	
		self.sum_points = [sum(x) for x in zip(self.sum_points, data_point)]
		self.centroid = [x/float(self.num_points) for x in self.sum_points]
		self.data_points_idx.append(data_point_idx)
		if cosine_sim_metric(self.centroid, data_point) > cosine_sim_metric(self.centroid, tfidf_titles[self.ft_idx]):
			self.ft_idx = data_point_idx

class cluster:
	cluster_type_list = ['title', 'description', 'date', 'location']
	num_clusters = [0]*4
	# To be used only when cluster_type_num = 4 

	def __init__(self, cluster_type_num, centroid, data_point_idx):
		self.num_points = 1
		self.cluster_type_num = cluster_type_num
		cluster.num_clusters[cluster_type_num] += 1
		self.point_indices = [data_point_idx]
		if cluster_type_num == 0:
			self.sum_points = [x for x in centroid]
			self.centroid = [x for x in centroid]
		else:
			self.sum_points = centroid
			self.centroid = centroid
		# self.cluster_type = cluster_type_list[cluster_type_num]

	def similarity(self, data_point):
		if self.cluster_type_num in [0,1]:
			return cosine_sim_metric(self.centroid, data_point)
		elif self.cluster_type_num == 2:
			return date_similarity_metric(self.centroid, data_point)

	def add_point(self, data_point_metric, data_point_idx):
		if self.cluster_type_num in [0,1]:
			self.sum_points = [sum(x) for x in zip(self.sum_points, data_point_metric)]
			self.centroid = [x/float(self.num_points + 1) for x in self.sum_points] 
		elif self.cluster_type_num == 2:
			self.sum_points += data_point_metric
			self.centroid = self.sum_points/float(self.num_points + 1)
		self.point_indices.append(data_point_idx)
		self.num_points += 1


def clustering_algo(cluster_list, cluster_type_num, data_points):
	""" 
	Algorithm to cluster data points based on individual features indexed by  cluster_type_num 
	cluster_l
	"""
	num_clusters = 0
	idx = 0
	label_pred = []												# Predicted cluster labels for data points
	threshold = 0.7
	for point in data_points:
		if cluster.num_clusters[cluster_type_num] == 0:
			print ('New cluster')
			first_cluster = cluster(cluster_type_num, point, idx)
			cluster_list.append(first_cluster)					# Maintaining list of all clusters	 	
			num_clusters += 1
			label_pred.append(0)								
		else:
			max_sim = -1										# Parameter to measure max similarity between data point and a cluster centroid
			cnum = 0											# Parameter to measure current cluster number
			maxcnum = 0											# Parameter to measure cluster with max similarity
			for cluster_ in cluster_list:
				sim = cluster_.similarity(point) 
				if sim > max_sim:
					max_sim = sim
					maxcnum = cnum
				cnum += 1

			if max_sim >= threshold:
				cluster_list[maxcnum].add_point(point, idx)
				label_pred.append(maxcnum)
			else:
				# print ('Adding cluster')
				new_cluster = cluster(cluster_type_num, point, idx)
				cluster_list.append(new_cluster)
				label_pred.append(cnum)
		idx += 1
	return label_pred 

def super_clustering_algo(cluster_list, sample_num, feature_num, weight, pred_lists):
	"""The final clustering based on binary weighted votes from each feature for a data point"""
	final_label_pred_list = []
	for data_point in range(0,sample_num):
		if final_cluster.num_clusters == 0:
			first_final_cluster = final_cluster(data_point, tfidf_titles[data_point])
			cluster_list.append(first_final_cluster)
			final_label_pred_list.append(0)
		else:
			max_score = -1.0
			cnum = 0
			ideal_cnum = 0
			for cluster_ in cluster_list:
				score = cluster_.score(data_point, weight, pred_lists)
				if score > max_score:
					ideal_cnum = cnum
					max_score = score
				cnum += 1
			# Here, we use majority voting and take threshold as 0.5
			threshold = 0.5
			if max_score >= threshold:
				cluster_list[ideal_cnum].add_point(data_point, tfidf_titles[data_point])
				final_label_pred_list.append(ideal_cnum)
			else:
				new_cluster = final_cluster(data_point,tfidf_titles[data_point])
				cluster_list.append(new_cluster)
				final_label_pred_list.append(cnum)

	min_tweets_per_cluster = floor(sqrt(sample_num / len(cluster_list)))
	for fcl in cluster_list:
		if fcl.num_points < min_tweets_per_cluster:
			cluster_list.remove(fcl)
	
	return final_label_pred_list

def generate_all_label_pred(tweets_dict):
	"""Returns prediction labels for all features in a list"""   
	return [generate_title_label_pred(tweets_dict), generate_date_label_pred(tweets_dict)]


def generate_title_label_pred(tweets_dict):
	"""Generate predicted cluster labels for data points based on title similarity"""
	global tfidf_titles
	# Get the list of sample_num number of titles from the dict corpus
	print("sample number:" , sample_num)
	title_list = corpus_list(tweets_dict, 0, sample_num)
	clean_docs(title_list)

	# Generating tf-idf data for titles
	print("title_list_number", len(title_list))
	tfidf_titles = gen_tfidf_array(title_list)
	# List to store cluster objects, based on similarity function for titles
	print ('**********************Clusters based on feature = Text*************************')
	ideal_cluster_list = []
	# Getting prediction label (cluster number assigned) for each data point in titles data
	title_label_pred = clustering_algo(ideal_cluster_list, 0, tfidf_titles)
	print ('Number of Clusters = ',len(ideal_cluster_list))
	return title_label_pred


def generate_date_label_pred(tweets_dict):
	"""Generate predicted cluster labels for data points based on description similarity"""
	days_list = get_days_list(tweets_dict, 1)
	difference = max(days_list) - min(days_list)
	print ('**********************Clusters based on feature = DateTime*************************')
	ideal_cluster_list = []
	# Getting prediction label (cluster number assigned) for each data point in date data
	date_label_pred = clustering_algo(ideal_cluster_list, 2, days_list)
	print ('Number of Clusters based on Datetime = ', len(ideal_cluster_list))
	return date_label_pred

def binary_voting(sample_num, feature_num, weights, label_pred_list):
	print ('\n\n**************************Getting into final data*******************************\n')
	final_cluster_list = []
	final_label_pred_list = []
	final_label_pred_list = super_clustering_algo(final_cluster_list, sample_num, feature_num, weights,
													 label_pred_list)
	print ('\nFinal Number of Clusters :',len(final_cluster_list) )
	return [final_cluster_list, final_label_pred_list]


def ranked_cluster_list(cluster_list):
	# ranked_cluster_list = list(cluster_list.values())
	rcl = sorted(cluster_list, key =lambda x: x.num_points, reverse = True)
	return rcl

def cluster_popular_words(cluster,k):
	"""return popular words in the cluster centroid"""
	# popular_words_tuple_list = [(x,y) for (y,x) in sorted(zip(cluster.centroid,cluster.tfidf_features),reverse=True)]
	popular_words_tuple_list = [(x,y) for (y,x) in sorted(zip(cluster.centroid,feature_names),reverse=True)]
	popular_word_list = [x for (x,y) in popular_words_tuple_list if y>0 ]
	# for t in range(0,len(cluster.centroid)):
	# 	print(cluster.tfidf_features[t],cluster.centroid[t])
	# for t in popular_words_tuple_list:
	# 	print(t[0],t[1])
	if(len(popular_word_list) <= k ):
		return popular_word_list
	else:
		return popular_word_list[:k]

d = {}
# with open('nepal_TWEB_SITUATIONAL.txt') as f:
with open(sys.argv[1]) as f:
    # reader = csv.reader(f, delimiter="\")
    # d = list(reader)
    # temp_d = list(reader)
    i = 0
    for line in f:
    # for idx in range(0, len(temp_d)):
    # the following is according to the format of the Nepal Dataset, please change it according to the input dataset
    	lst = line.split("\t")
    	d[i] = [lst[3], dt.strptime(lst[0], '%a %b %d %H:%M:%S %z %Y')]
    	if(i==0):
    		print(d[i])
    	i+=1
pickle.dump(d, open( "tweets_dict.p", "wb" ) )

tweets_dict = pickle.load( open( "tweets_dict.p", "rb" ) )
sample_num= len(tweets_dict)
# sample_num = 100
print ('Number of Tweet Samples : ', sample_num)

# label_pred_title = generate_title_label_pred(tweets_dict)
# label_pred_date = generate_date_label_pred(tweets_dict)
# for sample_num in range(1,total_num):

label_pred_list= generate_all_label_pred(tweets_dict)
feature_num = 2
weights = [0.70, 0.30]
final_cluster_list, final_label_pred_list = binary_voting(sample_num, feature_num, weights, label_pred_list)

ranked_cluster_list = ranked_cluster_list(final_cluster_list)
# num_of_top_clusters = 10
num_of_top_clusters = int(sys.argv[2])
# num_of_words_per_cluster =int(sys.argv[3])
num_of_words_per_cluster =int(sys.argv[3])
# num_of_top_clusters = sys.argv[3]

# print("TOTAL CLUSTER COUNT: ", cluster_text.num_of_clusters)
print("Showing Top %d Clusters" %(num_of_top_clusters))
for idx, cluster in enumerate(ranked_cluster_list):
	if(idx>=(num_of_top_clusters)):
		break
	# clust_key = get_cluster_key(cluster_list,cluster)
	pop_word_list = cluster_popular_words(cluster, num_of_words_per_cluster)
	# pop_word_list = cluster_popular_words(cluster, len(cluster.tfidf_features))
	print("Cluster with rank: ", idx)
	print("%d popular words" %(num_of_words_per_cluster))
	# print("%d popular words" %(len(cluster.tfidf_features)))
	print(pop_word_list)


