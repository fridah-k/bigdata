import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
spark= SparkSession.builder.appName('walmart').getOrCreate()
df= spark.read.csv('/home/fridah/Downloads/adult.csv', inferSchema= True, header= True)
print(df.columns)
# df_rdd = df.rdd
# type(df_rdd)
# print(df_rdd.count())
# print(df_rdd.take(5))

# created a new RDD
sc = SparkContext.getOrCreate()
rdd=sc.textFile('/home/fridah/Downloads/adult.csv')
# we have used the map() to transform the Rdd string to RDD array string.
# whereby it is separated with commas
mappedRdd= rdd.map(lambda x:x.split(";"))
# use the take() to view the data in a big dataset
print(mappedRdd.take(5))
# we are storing our headers for next use 
headerRdd= mappedRdd.take(1)[0]
#o get rid of headers and data types, we use FI:LTER()
mappedRddWithoutHeader= mappedRdd.filter(lambda x:x[0]!= 'age' and x[0]!='String')
# we can use the three commands as follows
mappedRddWithoutHeader= sc.textFile('/home/fridah/Downloads/adult.csv').map(lambda x:x.split(";")).filter(lambda x:x[0]!= 'age' and x[0]!='String')

# CONVERTING RDD TO DF
# this converting the whole RDD to DataFrame
df= mappedRddWithoutHeader.toDF(headerRdd)
print(df)

# converting selected columns of RDD to DF
# df= mappedRddWithoutHeader.map(lambda x:Row(age=[82],workclass=[Private],fnlwgt=[132870],education=[HS-grad],education.num=[9],marital.status=[Widowed],occupation=[Exec-mangerial],relationship=[Not-in-family],race=[White],sex=[Female],capital.gain=[0],capital.loss=[4356],hours.per.week=[],native.country=[United-States],income=[<=50K])).toDF()
# print(df)

# Create RDD from parallelize
df = spark.sparkContext.parallelize([(1, 2, 3, 'a b c'),
             (4, 5, 6, 'd e f'),
             (7, 8, 9, 'g h i')]).toDF(['col1', 'col2', 'col3','col4'])
df.show()

