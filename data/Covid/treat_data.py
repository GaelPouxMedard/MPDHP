import json
import gzip
import os
import time
import sys
import re
import datetime

s = "Thu Aug 05 16:31:43 +0000 2021"
date = datetime.datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y")
print(date.timestamp())
pause()

# Fonction affichage hiérarchie dict
def showDictStruct(d):
    def recursivePrint(d, i):
        for k in d:
            if isinstance(d[k], dict):
                print("-"*i, k)
                recursivePrint(d[k], i+2)
            else:
                print("-"*i, k, ":", d[k])
    recursivePrint(d, 1)

# it=350.000 ; fr=380.000 ; en= ; es=330.000
retweet_count_per_lg = {"_it": 1, "_fr": 3, "_en": 50, "_es": 10}

def treatAll():
    thres = None
    for folder in os.listdir("./"):
        numTweets = 0
        setWords = set()
        if ".txt" in folder: continue
        for k in retweet_count_per_lg:
            if k in folder:
                thres = retweet_count_per_lg[k]
        output = open(f"./{folder.replace('Tweets-treated-wo-retweets', 'events')}.txt", "w+")
        for month in os.listdir(f"./{folder}/"):
            for file in os.listdir(f"./{folder}/{month}/"):
                with gzip.open(f"./{folder}/{month}/{file}", 'r') as f:
                    for line in f:
                        d = json.loads(line)
                        retweet_count = d["retweet_count"]

                        if retweet_count<thres: continue

                        date = d["created_at"]
                        timestamp = int(datetime.datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp())
                        text = d["full_text"]

                        output.write(f"{timestamp}\t{text.replace(' ', ',')}\n")

                        # print(date)
                        # print(text)
                        # print(retweet_count)

                        numTweets += 1
                        setWords|=set(text.split(" "))

                print(folder, month, file, numTweets, len(setWords))


        output.close()


treatAll()
