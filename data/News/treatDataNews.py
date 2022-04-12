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


counts = {}
cntSub = {}
cntCons = 0
for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
    with open(f"news_titles_RS_2019-{month}.txt", "r", encoding="utf-8") as f:
        for line in f:
            infos = line.split("\t")

            try:
                if int(infos[5])<20:  # At least a difference of +20 upvotes
                    continue
            except:
                continue
            cntCons += 1

            text = treatText(infos[3])

            if infos[2] not in cntSub: cntSub[infos[2]] = 0
            cntSub[infos[2]] += 1

            for wd in text.split(" "):
                if wd not in counts: counts[wd] = 0
                counts[wd] += 1

    print(month, cntCons, cntSub)

efflines = 0
with open("allNews.txt", "w+", encoding="utf-8") as o:
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
                efflines += 1

            print(month, efflines, len([wd for wd in counts if counts[wd]>3]))