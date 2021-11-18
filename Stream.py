import findspark
findspark.init()
from operator import add, sub
from time import sleep
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
sc= SparkContext(appName="PysparkNotebook")
ssc = StreamingContext(sc, 1)
# we have input some data
inputData = [
    [1,2,3],
    [0],
    [4,4,4],
    [0,0,0,25],
    [1,-1,10],
]
rddQueue= []
for datum in inputData:
  rddQueue += [ssc.sparkContext.parallelize(datum)]
inputStream= ssc.queueStream(rddQueue)
inputStream.reduce(add).pprint()

ssc.start()
sleep(5)
ssc.stop(stopSparkContext= True, stopGraceFully=True)




