import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dataframe = None
x = None
y = None
train = None
test = None
model = None

def from_csv(filename):
    global dataframe
    dataframe = pandas.read_csv(filename)
    return dataframe

def preprocessing():
    global dataframe, x, y
    dataframe = dataframe.drop(["UDI","Product ID","TWF","HDF","PWF","OSF","RNF"], axis=1)
    
    encoder = LabelEncoder()
    dataframe['Type'] = encoder.fit_transform(dataframe['Type'])

    x = dataframe.drop('Machine failure', axis=1)
    y = dataframe['Machine failure']  

def split():
    global train, test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    train = (x_train, y_train)
    test = (x_test, y_test)

def fit():
    global model
    model = LogisticRegression()
    model.fit(train[0], train[1])
