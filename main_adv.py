# import dataset
import pandas as pd
import numpy as np
from numpy import ndarray
from pandas.core.frame import DataFrame
from gradient_descent_rss import gradient_descent

adv: DataFrame = pd.read_csv("./data/Advertising.csv")

y: ndarray = adv.loc[:,"sales"].to_numpy()
x: ndarray = adv.loc[:,["TV", "radio", "newspaper"]].to_numpy()

gradient_descent(y,x, 0.00001)
