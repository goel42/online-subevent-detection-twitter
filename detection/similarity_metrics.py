from sklearn.metrics.pairwise import cosine_similarity
from math import *
import numpy as np
def date_similarity_metric(m1, m2):
	"""Used to find similarity metric between no. of minutes since epoch -> m1 and m2, based on methods of Strehl et al."""
	# If time difference  b/w tweets > 1/2 a day => 0 similarity metric
	fraction_of_day = 0.5
	minutes_difference = abs(m2 - m1)
	minutes_limit = 1440 * fraction_of_day
	if minutes_difference >= minutes_limit:
		return 0
	else:
		return 1 - minutes_difference/float(minutes_limit)

def location_similarity_metric(lat1, long1, lat2, long2):
	""" Determination of location similarity metrics based on Haversine formula"""
	long1, lat1, long2, lat2 = map(radians, [long1, lat1, long2, lat2])
	# print long1, lat1, long2, lat2
	X1 = (sin((lat2 - lat1)/2.0))**2
	X2 = cos(lat1)*cos(lat2)*(sin((long2 - long1)/2.0)**2)
	return 2*asin(sqrt(X1 + X2))


def cosine_sim_metric(a1, a2):
	return cosine_similarity(np.asarray(a1).reshape(1,-1), a2.reshape(1,-1))[0][0]			#This is a 1x1 array, we get the float value from it through [0][0] indexing
