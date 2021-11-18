import findspark
findspark.init()
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import format_number
from pyspark.sql.functions import mean
from pyspark.sql.functions import min, max
from pyspark.sql.functions import corr
from pyspark.sql.functions import (dayofmonth, hour,
                                   dayofyear, month,
                                   year, weekofyear,
                                  format_number, date_format)

spark= SparkSession.builder.appName('walmart').getOrCreate()
df= spark.read.csv('/home/fridah/Downloads/WMT.csv', inferSchema= True, header= True)
print(df.columns)
df.printSchema()
# print the first five columns
for line in df.head(5):
    print(line, '\n')
# lets describe the data contents
df.describe().show()
# there are too many decimals in the mean and standard deviation.
# how to convert it into 2 decimal places
# summary= df.describe()
# summary.select(summary['summary'],
# format_number(summary['Open'].cast('float'), 2).alias('Open'),
# format_number(summary['High'].cast('float'), 2).alias('High'),
# format_number(summary['Low'].cast('float'), 2).alias('Low'),
# format_number(summary['Close'].cast('float'), 2).alias('Close'),
# format_number(summary['Volume'].cast('int'),0).alias('Volume'),
# ).show()
# # creating a new dataframe with a column called HV ratio
# df_hv= df.withColumn('HV Ratio', df['High']/df['Volume']).select(['HV Ratio'])
# df_hv.show()
# # which day had the highest price
# df.orderBy(df['High'].desc()).select(['Date']).head(1)[0]['Date']
# # mean of the close column
# df.select(mean('Close')).show()
# #Get the min and max of the volume
# df.select(max('Volume'), min('Volume')).show()
# # how days was close lower than 60 dollars
# df.filter(df['Close']<60).count()
# df.filter(df['High'] > 80).count() * 100/df.count()
# # getting correlation of the items
# df.corr('High','Volume')
# df.select(corr(df['High'], df['Volume'])).show()

# year_df= df.withColumn('Year', year(df['Date']))
# year_df.groupBy('Year').max()['Year', 'max(High)'].show()

# # what is the average close of each month in the calendar
# # create a new column of month from column date
# month_df= df.withColumn('Month', month(df['Date']))
# # group by month and the average of the months
# month_df= month_df.groupBy('Month').mean()
# # sort by month
# month_df= month_df.orderBy('Month')
# month_df['Month', 'avg(Close)'].show()


#creating Rdds with text file data
val (dataRDD)= spark.read.csv('/home/fridah/Downloads/WMT.csv').rdd
val dataRDD = spark.read.csv("path/of/csv/file").rdd





    


                





