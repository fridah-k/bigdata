import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql.session import SparkSession
from pyspark.streaming import StreamingContext
import pyspark.sql.types as tp # List of data types available.
from pyspark.ml import Pipeline # it is for transformation and estimators
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import Row
import sys
# initializing spark session
sc= SparkContext(appName="PysparkShell")
spark= SparkSession(sc)
# define the schema
my_schema= tp.StructType([tp.StructField(name= 'id', dataType= tp.IntegerType(), nullable= True),
    tp.StructField(name= 'label', dataType= tp.IntegerType(), nullable= True),
    tp.StructField(name= 'tweet', dataType= tp.StringType(), nullable= True)])
# Read the data set 
my_data= spark.read.csv('/home/fridah/Downloads/twitter-sentiments.csv', schema= my_schema, header=True)
my_data.show(5)
# print the schema of the file
my_data.printSchema()

# load the data and stream it into various transformations
# First we tokenize the data  that is converting the tweets into words
stage_1= RegexTokenizer(inputCol= 'tweet', outputCol= 'tokens', pattern= '\\W')
# Remove the stop words
stage_2= StopWordsRemover(inputCol= 'tokens', outputCol= 'filtered_words')
#define stage 3: create a word vector of the size 100
stage_3= Word2Vec(inputCol= 'filtered_words', outputCol= 'vector', vectorSize= 100)
#  define stage 4: Logistic Regression Model
model= LogisticRegression(featuresCol= 'vector', labelCol= 'label')
# set up the pipeline
pipeline= Pipeline(stages= [stage_1, stage_2, stage_3, model])
# fit the pipeline model with the training data
pipelineFit= pipeline.fit(my_data)
# initialize the Spark Streaming context and define a batch duration of 3 seconds.
#  This means that we will do predictions on data that we receive every 3 seconds

# define a function to compute sentiments of the received tweets
def get_prediction(tweet_text):
    try:
# filter the tweets whose length is greater than 0
        tweet_text= tweet_text.filter(lambda x: len(x) > 0)
# create a dataframe with column name 'tweet' and each row will contain the tweet
        rowRdd= tweet_text.map(lambda w: Row(tweet=w))
# create a spark dataframe
        wordsDataframe= spark.createDataFrame(rowRdd)
# transform the data using the pipeline and get the predicted sentiment
        pipelineFit.transform(wordsDataframe).select('tweet', 'prediction').show()
    except:
        print('No data')
# initialize the streaming context

ssc= StreamingContext(sc, batchDuration= 3)
# HOST= sys.argv[0]
# PORT= int(sys.argv[8080])

# Create a DStream that will connect to hostname:port, like localhost:8080
print('dgfshmlnmx cfn nv c')
print(sys.argv)
lines= ssc.socketTextStream(sys.argv[1], int(sys.argv[2]))

# split the tweet text by a keyword 'TWEET_APP' so that we can identify which set of words is from a single tweet
words= lines.flatMap(lambda line: line.split('TWEET_APP'))
# #get the predicted sentiments for the tweets received
words.foreachRDD(get_prediction)
# # start the computation
ssc.start()
# # wait for the computation to terminate
ssc.awaitTermination()
