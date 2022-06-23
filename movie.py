# Databricks notebook source
pip install nltk

# COMMAND ----------

#dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/movie_metadata-4.tsv
#dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/searchwords-6.txt
#dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/stopwords-2.txt
#dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/plot_summaries-4.txt

# COMMAND ----------

plot = sc.textFile("dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/plot_summaries-4.txt")
plot.collect()

# COMMAND ----------

movies = sc.textFile("dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/movie_metadata-4.tsv")
movies.collect()

# COMMAND ----------

import nltk
from nltk.corpus import stopwords
nltk.download('all')
from nltk.tokenize import word_tokenize

# COMMAND ----------

import re

# COMMAND ----------

size = plot.count()
plots = plot.map(lambda x: x.split("\t"))
plots.count()

# COMMAND ----------

special="[\[\]# ./\"'-:;|?*“”’‘]"

# COMMAND ----------

plotText = plots.flatMap(lambda x : [((x[0], word.lower()), 1) for word in re.split(special, x[1]) if len(word)>1])
plotText.collect()

# COMMAND ----------

stopWords = sc.textFile("dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/stopwords-2.txt")
stopWords = stopWords.flatMap(lambda x: x.split('\n'))
stopWords = stopWords.collect()
plotWords = plotText.filter(lambda x: x[0][1] not in stopWords)
plotWords.count()

# COMMAND ----------

plotText = plotWords.reduceByKey(lambda x,y: x+y)
plotText.collect()

# COMMAND ----------

tfPlot = plotText.map(lambda x : (x[0][1], (x[0][0], x[1])))
idf = plotText.map(lambda x: (x[0][1], (x[0][0], x[1], 1)))
idf.collect()

# COMMAND ----------

idf = idf.map(lambda x : (x[0], x[1][2]))
idf.collect()

# COMMAND ----------

idf = idf.reduceByKey(lambda x,y: x+y)
idf.collect()

# COMMAND ----------

import math

# COMMAND ----------

idfPlot = idf.map(lambda x: (x[0], math.log10(size/x[1])))
idfPlot.collect()

# COMMAND ----------

tfIdf = tfPlot.join(idfPlot)
tfIdf.collect()

# COMMAND ----------

tfIdfPlot = tfIdf.map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1], x[1][0][1]*x[1][1] )))
tfIdfPlot.collect()

# COMMAND ----------

movies = sc.textFile("#dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/movie_metadata-4.tsv")
movieL = movies.map(lambda x : x.split('\t'))
movies_map = movieL.map(lambda x : (x[0], x[1], x[2])).map(lambda x : (x[0], x[2]))
tfIdfMovies = tfIdfPlot.join(movies_map)
tfIdfMovies.collect()

# COMMAND ----------

searchWords = sc.textFile("dbfs:/FileStore/shared_uploads/rxk200082@utdallas.edu/searchwords-6.txt")
searchWords.collect()

# COMMAND ----------

idfWords= tfIdfMovies.map(lambda x : (x[1][0][0], (x[1][1], x[1][0][2])))
for query in searchWords.collect():
    print("Searching for {} movies:".format(query))
    query = query.lower()
    queryTerms = query.split()
    if len(queryTerms)>1:  #For queries having more than a word
        queryTf = sc.parallelize(queryTerms).map(lambda x:(x,1)).reduceByKey(lambda x,y: x+y)
        queryTfIdf = queryTf.leftOuterJoin(idfWords)
        queryTfIdf = queryTfIdf.map(lambda x:(x[0], 0 if x[1][1] is None  else  x[1][0]*x[1][1][1]))
        queryTfIdf = queryTfIdf.map(lambda x: ( (x[0], x[1])))
        
        queriesTfIdf = tfIdfMovies.map(lambda x: (x[1][0][0], (x[1][1], x[1][0][3])))
        queriesTfIdf = queriesTfIdf.join(queryTfIdf)
        queriesTfIdf = queriesTfIdf.map(lambda x : (x[1][0][0], x[1][0][1] , x[1][1]) )
        queriesTfIdf = queriesTfIdf.map(lambda x: (x[0], (x[1]*x[2], x[2]*x[2], x[1]*x[1])))
        queriesTfIdf = queriesTfIdf.reduceByKey(lambda x,y : ((x[0] + y[0], x[1] + y[1], x[2] + y[2])))
        cosSimilar = queriesTfIdf.map(lambda x: (x[0], x[1][0]/(math.sqrt(x[1][1])*math.sqrt(x[1][2])))).sortBy(keyfunc = lambda x: x[1], ascending = False).take(10)
    else:
        cosSimilar = tfIdfMovies.filter(lambda x : x[1][0][0] == queryTerms[0]).sortBy(keyfunc = lambda x: x[1][0][3], ascending = False).map(lambda x :(x[1][1], x[1][0][3])).take(10)
    for movie in cosSimilar:
        print("\t", movie[0])

# COMMAND ----------


