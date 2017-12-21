# online-subevent-detection-twitter
Given a corpus of tweets pertaining to a given major event (Eg. Nepal earthquake). Broadly, The task is defined as 
  
  1) Important Sub-event Detection
  2) Generating a summary report of the event based on these subevents
  
For subevent detection we use two approaches for clustering(each cluster denotes a unique subevent)
  
    1. Complete Dataset is fed at once (Static Dataset Approach)
    2. Tweets are fed one after another (Online Dataset Approach, more realistic)
 
We explore the following three features for a given tweet
 
    1) Tweet text
    2) DateTime
    3) Location

Initially, clusters are formed and analysed using individual features only. Finally the superclusters/finalclusters are formed using weighted binary voting.

For summarisation:-

  1. We first use a greedy algorithm to select representative tweets (focus set) from each cluster to form summaries. Using this, our     summarisation tends  to select tweets such that it can cover most information of the whole tweet set.
  2. From this focus set of tweets, we generate a cosine-similarity graph. In this graph, a tweet from the focus set represents a node and tweets this similarity greater than a minimum threshold are connected by an edge
  3. We obtain a set of tweet centrality scores using the PageRank algorithm, based on this graph that it takes as an input.
  4. However, a potential problem of LexRank is that some top-ranked tweets may have similar contents. So, we choose one tweet with the highest LexRank score from each cluster which is sufficiently unique(based on a threshold value) in the summary generated so far.  
  
References:-
  1. Sumblr: continuous summarization of evolving tweet streams. Conference: SIGIR
  2. Sub-Event Detection During Natural Hazards Using Features of Social Media Data. Conference: WWW
