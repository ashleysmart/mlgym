import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def load_titanic(datadir = None):
    if datadir is None:
        datadir = "~/data/kaggle/titanic/"
    testfile = datadir + "test.csv"
    trainfile = datadir + "train.csv"

    test  = pd.read_csv(testfile)
    train = pd.read_csv(trainfile)

    return train,test

def clean_titanic(train,test):
    # Step 0 
    # save ourselves the time and merge it all togther

    alldata = pd.concat([train,test], axis=0)

    # Step 1 
    #delete useless stuff

    alldata = alldata.drop(["Name", "Ticket", "Cabin","PassengerId"], 1)
    alldata.head()

    # Step 2 
    # Expand the catogrical data into boolen indications of presence or not
    # This remove the need for the model to learn the "meaning" of the value and uncomplicates the situation

    # dummy cols.. convert the "class" values into attibutes that are true/false
    dummy_cols=["Embarked","Sex","Pclass"]
    for column in dummy_cols:
        dummies = pd.get_dummies(alldata[column])
        alldata[dummies.columns] = dummies
    alldata = alldata.drop(dummy_cols, 1)
    #delete MALE its just the inverse of female.. 
    alldata = alldata.drop(["male"], 1)


    # Step 3 
    # Handling missing data
    # fill in Nan data with mean values
    #clean up the Nan(bad) data
    nan_cols = ["Age","Fare"]
    for column in nan_cols:
        coldata = alldata[column]
        coldata = coldata.fillna(coldata.mean())
        alldata[column] = coldata
    alldata.head()

    # slice the data apart again
    out_cols = ["Survived"] 

    xtrain = alldata[0:len(train)]
    ytrain = xtrain[out_cols] 
    xtrain = xtrain.drop(out_cols, 1)

    xtest  = alldata[len(train):]
    ytest  = xtest[out_cols] 
    xtest  = xtest.drop(out_cols, 1)

    return xtrain, ytrain, xtrain, ytrain


def plotboundary(inputs, outputs, x1, x2, predict):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = inputs[x1].min(), inputs[x1].max()
    y_min, y_max = inputs[x2].min(), inputs[x2].max()
    x_step = (x_max - x_min)/30.0
    y_step = (y_max - y_min)/30.0

    #basis_tag = [ "", "min", "mean", "max"]
    #basis = [inputs.min(), inputs.mean(), inputs.max()]
    basis_idx = [3,4,5,6,7]
    basis = inputs.describe() 
        
    xx, yy = np.meshgrid(np.arange(x_min-x_step, x_max+x_step, x_step), 
                         np.arange(y_min-y_step, y_max+y_step, y_step))
    
    plt.rcParams['figure.figsize'] = (16, 4)
    #plt.figure(figsize=(20,9))
    #plt.subplots_adjust(hspace=.7)
    f, ax = plt.subplots(1, 6)
    fig = 0
    
    # Plot also the training points
    ax[fig].scatter(inputs[x1], inputs[x2], c=outputs, edgecolors='k', cmap=plt.cm.Paired)
    
    ax[fig].set_xlim(xx.min(), xx.max())
    ax[fig].set_ylim(yy.min(), yy.max())
    ax[fig].set_xticks(())
    ax[fig].set_yticks(())
        
    ax[fig].set_xlabel(x1)
    ax[fig].set_ylabel(x2)
    fig += 1
    
    for idx in basis_idx:
        base = basis.iloc[idx]
        tag  = basis.index[idx]
        mockin = pd.concat([base] * xx.ravel().shape[0], axis=1).transpose()

        mockin[x1] = xx.ravel()
        mockin[x2] = yy.ravel()
        
        Z = predict(mockin)
        Z = Z.reshape(xx.shape)
        ax[fig].pcolormesh(xx, yy, Z, cmap='RdBu')

        ax[fig].set_xlim(xx.min(), xx.max())
        ax[fig].set_ylim(yy.min(), yy.max())
        ax[fig].set_xticks(())
        ax[fig].set_yticks(())

        ax[fig].set_xlabel(tag)

        fig += 1


    plt.show()

