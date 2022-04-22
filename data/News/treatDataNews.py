import pickle
stopwords_english = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stopwords_english += ["say", "new", "news", "til"
                        , "be"
                        , "have"
                        , "use"
                        , "/"
                        , "get"
                        , "2019"
                        , "man"
                        , "$"
                        , "so"
                        , "one"
                        , "%"
                        , "go"
                        , "do"
                        , "make"
                        , "when"
                        , "year"
                        , "how"
                        , "why"
                        , "only"
                        , "-"
                        , "|"]

import re


import matplotlib.pyplot as plt
import numpy as np
import datetime


def treatText(txt):
    # convert to lower case
    lower_string = txt.lower()
    lower_string = lower_string.replace("\n", " ").replace("&amp;", "&")
    lower_string = re.sub(r'\w*http[s]?://t.co/\w*', '', lower_string)
    lower_string = re.sub(r'\w*@\w*', '', lower_string)

    # remove all punctuation except words and space
    lower_string = re.sub(r'[^\w\s]', ' ', lower_string)

    # remove white spaces
    lower_string = lower_string.strip()

    # convert string to list of words
    tabWds = [word for word in lower_string.split() if word not in stopwords_english]
    tabWds = [word for word in tabWds if len(word)>=4]
    final_string = ' '.join(tabWds)

    return final_string


# counts = {}
# cntSub = {}
# cntCons = 0
# totalEntries = 0
# for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
#     with open(f"news_titles_RS_2019-{month}.txt", "r", encoding="utf-8") as f:
#         for line in f:
#             infos = line.split("\t")
#
#             totalEntries += 1
#
#             try:
#                 if int(infos[5])<20:  # At least a difference of +20 upvotes
#                     continue
#             except:
#                 continue
#
#             time = float(infos[1])/60  # In minutes
#
#             cntCons += 1
#
#             text = treatText(infos[3])
#
#             if infos[2] not in cntSub: cntSub[infos[2]] = 0
#             cntSub[infos[2]] += 1
#
#             for wd in text.split(" "):
#                 if wd not in counts: counts[wd] = 0
#                 counts[wd] += 1
#
# print(month, counts, cntSub)
#
#
# with open("counts.pkl", "wb") as f:
#     pickle.dump(counts, f)
#
# pause()

with open("counts.pkl", "rb") as f:
    counts = pickle.load(f)

# efflines = 0
# allentries = 0
# cntSub = []
# allWords = []
# popularity, times = [], []
# allData = 0
# avglen = []
# for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
#     with open(f"news_titles_RS_2019-{month}.txt", "r", encoding="utf-8") as f:
#         for line in f:
#             infos = line.split("\t")
#             allentries += 1
#
#
#             try:
#                 if int(infos[5])<20:  # At least a difference of +20 upvotes
#                     continue
#             except:
#                 continue
#
#
#
#             time = float(infos[1])/60  # In minutes
#
#             text = treatText(infos[3])
#             #print(line)
#             allData += 1
#
#             text = [wd for wd in text.split(" ") if counts[wd]>3]
#             if not len(text)>=3:
#                 continue
#
#             cntSub.append(infos[2])
#             times.append(float(infos[1])/60)
#             popularity.append(int(infos[5]))
#             allWords += text
#             avglen.append(len(text))
#
#             efflines += 1
#
#         print(month, allentries, efflines, len([wd for wd in counts if counts[wd]>3]))
#         #break
#
#
# plt.figure(figsize=(10,5))
#
# plt.subplot(2,2,2)
# plt.hist(times, bins=50)
# plt.xlabel("Date")
# plt.ylabel("Count")
# dt = 30*60*24
# x_ticks = [np.min(times)]
# while x_ticks[-1]<np.max(times):
#     x_ticks.append(x_ticks[-1]+dt)
# x_ticks = x_ticks[:-1]
# x_labels = [datetime.datetime.fromtimestamp(float(ts)*60).strftime("%d %b") for ts in x_ticks]
# plt.xticks(x_ticks, x_labels, rotation=45, ha="right")
# plt.title("Times distribution")
#
# plt.subplot(2,2,3)
# plt.hist(popularity, bins=100)
# plt.semilogy()
# plt.ylabel("Count")
# plt.xlabel("Popularity")
# plt.title("Popularity distribution")
#
# plt.subplot(2,2,1)
# un, cnt = np.unique(cntSub, return_counts=True)
# un = [u for _, u in sorted(zip(cnt, un), reverse=True)]
# cnt = [c for c, u in sorted(zip(cnt, un), reverse=True)]
# xticks = []
# for i, (u,c) in enumerate(zip(un,cnt)):
#     plt.bar(i, c, 0.8)
#     xticks.append(u.capitalize())
#
# plt.xticks(list(range(len(xticks))), xticks, rotation=45, ha="right")
# plt.ylabel("Count")
# plt.title("Subreddits distribution")
#
#
# plt.subplot(2,2,4)
# un, cnt = np.unique(allWords, return_counts=True)
# un = [u for _, u in sorted(zip(cnt, un), reverse=True)]
# cnt = [c for c, u in sorted(zip(cnt, un), reverse=True)]
# plt.hist(cnt, bins=100)
#
# print(len(allWords), len(un))
#
# plt.semilogy()
# plt.ylabel("Density")
# plt.xlabel("Count")
# plt.title("Word counts distribution")
#
#
#
# for a in plt.gcf().get_axes():
#     a.label_outer()
#
#
# plt.tight_layout()
# plt.savefig("Stats_DS_after_red.pdf")
# plt.show()
#
pause()

efflines = 0
with open("allNews.txt", "w+", encoding="utf-8") as o:
    with open("metadata.txt", "w+", encoding="utf-8") as m:
        allData = 0
        for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
            with open(f"news_titles_RS_2019-{month}.txt", "r", encoding="utf-8") as f:
                for line in f:
                    infos = line.split("\t")

                    try:
                        if int(infos[5])<20:  # At least a difference of +20 upvotes
                            continue
                    except:
                        continue

                    time = float(infos[1])/60  # In minutes

                    text = treatText(infos[3])
                    #print(line)
                    allData += 1

                    text = [wd for wd in text.split(" ") if counts[wd]>3]
                    if not len(text)>=3:
                        continue

                    o.write(f"{time}\t{','.join(text)}\n")
                    m.write(f"{infos[2]}\n")
                    efflines += 1

                print(month, efflines, len([wd for wd in counts if counts[wd]>3]))