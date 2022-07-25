from pandas import read_csv, DataFrame #for data frames
from numpy import ravel #for matrices
import matplotlib.pyplot as plt
import seaborn as sns #for plotting data

from sklearn.model_selection import train_test_split


#various pre-processing steps
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, Normalizer, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import GridSearchCV #used for optimization


#read the ecoli dataset
df = read_csv(
        '/home/machio_b/Documents/Data_Sets/ecoli.data',
        sep='\s+',
        header=None
)

#print(df.head())

#separate features from the labels
#split the datasets into 2/3 -> training instance and 1/3 test instance

#the data matrix X - iloc integer-location based indexing -> selects by position
x = df.iloc[:,1:-1]
#the labels
y = df.iloc[:,-1:]

#encode these labels into unique integers
encoder = LabelEncoder()
y = encoder.fit_transform(ravel(y))

#split the data set into test and train
x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=1/3,
        random_state=0
)

# print(x_train.shape)
# print(x_test.shape)