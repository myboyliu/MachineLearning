import os
import pandas as pd

class DataLoad():
    def __init__(self, filename):
        self.filename = filename

    def getfilename(self):
        basedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        datadir = os.path.join(basedir, 'data')
        datafile = os.path.join(datadir, self.filename)
        return datafile

    def getFullData(self):
        return pd.read_csv(self.getfilename())

    def getRawData(self):
        return self.getFullData().values[:, 1:].tolist()

    def getColumns(self):
        return self.getFullData().columns[1:-1].tolist()

if __name__ == '__main__':
    load = DataLoad('watermelon30.txt')
    data = load.getdata()
    readData = data.values[:, 1:].tolist()
    columns = data.columns[1:-1].tolist()
    print(data)
    print(readData)
    print(columns)
