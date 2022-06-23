# Databricks notebook source
pip install nltk

# COMMAND ----------

text = sc.textFile("dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/Electric_Light.txt")
text.collect()

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
nltk.download("all")

# COMMAND ----------

import re

# COMMAND ----------

special="[\[\]# ./\"'-:;|?*“”’‘]"

# COMMAND ----------

stopWords = sc.textFile("dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/stopwords-2.txt")
stopWords = stopWords.flatMap(lambda x: x.split('\n'))
stopWords = stopWords.collect()

# COMMAND ----------

text=text.flatMap(lambda context:re.split(special,context))
text=text.filter(lambda context:len(context)>0)
text.collect()

# COMMAND ----------

simplifiedText = text.filter(lambda x: x.lower() not in stopwords.words("english"))
simplifiedText.collect()

# COMMAND ----------

tagText=simplifiedText.filter(lambda x: nltk.pos_tag([x])[0][1]=="NN")
tagText.collect()

# COMMAND ----------

textMap=tagText.map(lambda x: (x.lower(),1))
textMap.collect()

# COMMAND ----------

textFreq = textMap.reduceByKey(lambda x,y: x+y).sortBy(lambda x: -x[1])
textFreq.collect()

# COMMAND ----------


