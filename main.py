import pandas as pd

from scipy.spatial import distance

from utility import text_preprocessing, tokenization, create_domain_count, create_domain_tf, create_idf, create_tf_idf

with open('football-train') as f:
    football_train = f.read()

with open('os-train') as f:
    os_train = f.read()

with open('football-test') as f:
    football_test = f.read()

with open('os-test') as f:
    os_test = f.read()

football_train = text_preprocessing(football_train)
os_train = text_preprocessing(os_train)
domains_dict = football_train + ' ' + os_train
domains_dict = tokenization(domains_dict)

# Training data
football_train_count = create_domain_count(domains_dict, football_train)
os_train_count = create_domain_count(domains_dict, os_train)

football_train_tf = create_domain_tf(football_train_count)
os_train_tf = create_domain_tf(os_train_count)

idf_train = create_idf(football_train_count, os_train_count)

football_train_tf_idf = create_tf_idf(football_train_tf, idf_train)
os_train_tf_idf = create_tf_idf(os_train_tf, idf_train)

# Merge all dictionaries in one
final_train = {}
for word in football_train_count.keys():
    final_train[word] = [football_train_count[word]]
    final_train[word].append(os_train_count[word])
    final_train[word].append(football_train_tf[word])
    final_train[word].append(os_train_tf[word])
    final_train[word].append(idf_train[word])
    final_train[word].append(football_train_tf_idf[word])
    final_train[word].append(os_train_tf_idf[word])

dataframe_train = pd.DataFrame.from_dict(final_train, orient='index')

dataframe_train = dataframe_train.rename(
    columns={0: 'COUNT1', 1: 'COUNT2', 2: 'TF1', 3: 'TF2', 4: 'IDF', 5: 'TF-IDF1', 6: 'TF-IDF2'})

print(dataframe_train.head(10))

# Test data
test_count = create_domain_count(domains_dict, football_test)

test_tf = create_domain_tf(test_count)

# idf_test = create_idf(football_test_count, football_test_count)

# Use idf of train dictionary
test_tf_idf = create_tf_idf(test_tf, idf_train)

# Merge all dictionaries in one
final_test = {}
for word in test_count.keys():
    final_test[word] = [test_count[word]]
    final_test[word].append(test_tf[word])
    final_test[word].append(idf_train[word])
    final_test[word].append(test_tf_idf[word])

dataframe_test = pd.DataFrame.from_dict(final_test, orient='index')

dataframe_test = dataframe_test.rename(columns={0: 'COUNT', 1: 'TF', 2: 'IDF', 3: 'TF-IDF'})

print(dataframe_test.head(10))

# Calculate sum of every TF-IDF column in train dictionary
sum_football_train = sum(football_train_tf_idf.values())
sum_os_train = sum(os_train_tf_idf.values())

# Calculate sum of TF-IDF column in test dictionary
sum_test = sum(test_tf_idf.values())

# Calculate euclidean distance between train TF-IDFs and test IF-IDFy
football_distance = distance.euclidean(sum_football_train, sum_test)
os_distance = distance.euclidean(sum_os_train, sum_test)

# The nearest the winner
print(football_distance, os_distance)
if football_distance < os_distance:
    print("Football Domain")
else:
    print("OS Domain")
