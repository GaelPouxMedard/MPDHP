import json
import gzip
import os
import time
import sys
import re
import datetime

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
        if not os.path.isdir(folder): continue

        for k in retweet_count_per_lg:
            if k in folder:
                thres = retweet_count_per_lg[k]
        output = open(f"./{folder.replace('Tweets-treated-wo-retweets', 'events')}.txt", "w+", encoding="utf-8")
        for month in os.listdir(f"./{folder}/"):
            for file in os.listdir(f"./{folder}/{month}/"):
                with gzip.open(f"./{folder}/{month}/{file}", 'r') as f:
                    for line in f:
                        d = json.loads(line)
                        retweet_count = d["retweet_count"]

                        if retweet_count<thres: continue

                        date = d["created_at"]
                        timestamp = datetime.datetime.strptime(date, "%a %b %d %H:%M:%S %z %Y").timestamp()/60  # Minutes
                        text = d["full_text"]
                        text = re.sub(r'[0-9]+', '', text)
                        text = re.sub(r'\b\w{1,3}\b', '', text)
                        text = text.strip()
                        text = re.sub(r' +', ',', text)

                        output.write(f"{timestamp}\t{text}\n")

                        # print(date)
                        # print(text)
                        # print(retweet_count)

                        numTweets += 1
                        setWords|=set(text.split(","))

                print(folder, month, file, numTweets, len(setWords))

            #     if numTweets>2000: break
            # if numTweets>2000: break

        output.close()


treatAll()
