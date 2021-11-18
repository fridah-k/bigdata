# Data analysis on the Uber trips that are done on dairly basis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# load data
rides= pd.read_csv("/home/fridah/Downloads/uber_rides.csv")
print(rides.head(10))
print(rides.describe())
# the mean of dropoff_lattitude is the highest while humidity is thwe lowest
print(rides.columns)
rides.drop["rider_uid"]

