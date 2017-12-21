import csv
import pickle
from datetime import datetime
d = {}
with open('nepal_TWEB_SITUATIONAL.txt') as f:
    # reader = csv.reader(f, delimiter="\")
    # d = list(reader)
    # temp_d = list(reader)
    i = 0
    for line in f:
    # for idx in range(0, len(temp_d)):
    	lst = line.split("\t")
    	d[i] = [lst[3], datetime.strptime(lst[0], '%a %b %d %H:%M:%S %z %Y')]
    	if(i==0):
    		print(d[i])
    	i+=1
pickle.dump(d, open( "tweets_dict.p", "wb" ) )
