import sys

import pandas as pd

def readcsv(name):
    return pd.read_csv(name)

def main(argv):
    print(argv)
    df = readcsv('/Users/faraday/Documents/Projects/mwdb/phase1_dataset/mlratings.csv')
    #print(df.ix[:,:2])
    print(df.iloc[:5,[1,3]])
    df['sum'] = df.apply(lambda row: row[1]+row[3], axis=1)
    print(df)


if __name__ == '__main__':
    main(sys.argv)







