import random
import pandas as pd


def generateRandNumbers(length):
    result = []
    for n in range(length):
        result.append(round(random.uniform(0, 100), 2))
    return result


df = pd.DataFrame(columns=['numbers1', 'numbers2', 'numbers3'])
df['numbers1'] = generateRandNumbers(20)
df['numbers2'] = generateRandNumbers(20)
df['numbers3'] = generateRandNumbers(20)

path = '.\preprocessing\mockData'
df.to_csv(path + '\mock.csv')

print(df)
