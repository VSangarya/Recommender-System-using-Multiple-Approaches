import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, desc , col, max
from pyspark.ml.feature import  StringIndexer
from pyspark.ml import Pipeline



ratingpath=r'C:\Users\V Sangarya\Downloads\rating\ratingsSmall.csv'
moviepath=r'C:\Users\V Sangarya\Downloads\movies.csv'


df = pd.read_csv(ratingpath)
movie_titles = pd.read_csv(moviepath)
df=df.drop(['timestamp'],axis=1,)

n_users = df.userId.nunique()
n_items = df.movieId.nunique()

print('Num. of Users: '+ str(n_users))
print('Num of Movies: '+str(n_items))

train_data, test_data = train_test_split(df, test_size=0.25)

train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]


UMatrix = train_data_matrix


def findFeatures(UMatrix2, EigenVal, EigenVec):
    k = EigenVal.argsort()[::-1]
    EigenVal = EigenVal[k]
    EigenVec = EigenVec[:, k]
    V = EigenVec
    j = 0

    for i in EigenVal:
        if j == x.size:
            break
        elif i >= 0:
            x[j] = np.sqrt(i)
            j += 1

    U = np.transpose(EigenVec)

    for i in EigenVal:
        if j == x.size:
            break
        elif i >= 0:
            x[j] = np.sqrt(i)
            j += 1
    x[::-1].sort()
    return U, x, V

  
x = np.zeros(UMatrix.shape[1])
UMatrix2 = np.matmul(UMatrix, np.transpose(UMatrix))
EigenVal, EigenVec = la.eigh(UMatrix2)
u, d, vt = findFeatures(UMatrix2, EigenVal, EigenVec)


print("Left Singular Matrix is \n", u, "\n")
print("Latent factors strength is \n", d, "\n")
print("Right Singular Matrix is \n", vt)




def ALSRec(uid, n):
    sparks = SparkSession.builder.appName("ALS_RecSys").getOrCreate()

    df_movies = sparks.read.format('csv').option('header', True).option('inferSchema', True).load(ratingpath)
    df_movies = df_movies.drop('timestamp')
    df_movies = df_movies.na.drop()

    row = df_movies.count()
    col = len(df_movies.columns)
    print(row, col)

    (training, test) = df_movies.randomSplit([0.8, 0.2])

    USERID = 'userId'
    TRACK = 'movieId'
    COUNT = 'rating'

    als = ALS(maxIter=5, regParam=0.01, userCol=USERID, itemCol=TRACK, ratingCol=COUNT)
    model = als.fit(training)

    recs = model.recommendForAllUsers(n)
    res = recs.take(uid)

    pred = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="pred")
    err = evaluator.evaluate(pred)
    print("Root-mean-square error = " + err)

    return res



print("Enter a user id and number of recommendations")
uid = int(input())
n=int(input())
uidRes=ALSRec(uid,n)
print("Top 3 predictions for user ID :", uid, "is ", uidRes)




def userRec(uid,n):
    train_data2 = train_data.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

    trainPro = train_data.copy()
    testPro = test_data.copy()
    trainPro['rating'] = trainPro['rating'].apply(lambda x: 0 if x > 0 else 1)
    testPro['rating'] = testPro['rating'].apply(lambda x: 1 if x > 0 else 0)
    trainPro = trainPro.pivot(index='userId', columns='movieId', values='rating').fillna(1)
    testPro = testPro.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    PredUser = 1 - pairwise_distances(train_data2, metric='cosine')
    PredUser[np.isnan(PredUser)] = 0
    user_predicted_ratings = np.dot(PredUser, train_data2)
    user_final_ratings = np.multiply(user_predicted_ratings, trainPro)

    uidRes = user_final_ratings.iloc[uid].sort_values(ascending=False)
    uidRes.to_csv(r'C:\Users\V Sangarya\Desktop\Results.csv')
    Res=pd.read_csv(r'C:\Users\V Sangarya\Desktop\Results.csv')
    ResMovies=Res.loc[:,"movieId"]
    return ResMovies[0:n]


  
print("Enter a user id and number of recommendations")
uid = int(input())
n=int(input())
uidRes=userRec(uid,n)
print("Top 3 predictions for user ID :", uid, "is ")
for i in range(len(uidRes)):
    print(movie_titles.loc[[i], ['title']])



