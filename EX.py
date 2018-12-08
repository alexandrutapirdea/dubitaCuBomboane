import pandas as pd
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('good.csv')

col = ['Genre', 'Lyrics']

df = df[col]
# print(df.head())

# df.info()


df['genre_id'] = df['Genre'].factorize()[0]

genre_id_df = df[['Genre', 'genre_id']].drop_duplicates().sort_values('genre_id')
genre_to_id = dict(genre_id_df.values)
id_to_genre = dict(genre_id_df[['genre_id', 'Genre']].values)

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(df.head())

# df = df.groupby('Genre').head(1000)


# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(8,6))
# df.groupby('Genre').Lyrics.count().plot.bar(ylim=0)
# plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.Lyrics).toarray()
labels = df.genre_id
# print(features.shape)

# N = 2
# for Genre, genre_id in sorted(genre_to_id.items()):
#   features_chi2 = chi2(features, labels == genre_id)
#   indices = np.argsort(features_chi2[0])
#   feature_names = np.array(tfidf.get_feature_names())[indices]
#   unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
#   bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
#   print("# '{}':".format(Genre))
#   print("  . Most correlated unigrams:\n       . {}".format('\n       . '.join(unigrams[-N:])))
#   print("  . Most correlated bigrams:\n       . {}".format('\n       . '.join(bigrams[-N:])))


X_train, X_test, y_train, y_test = train_test_split(df['Lyrics'], df['Genre'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# clf = MultinomialNB().fit(X_train_tfidf, y_train)

#
# models = [
#     RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#     LinearSVC(),
#     MultinomialNB(),
#     LogisticRegression(random_state=0),
# ]
# CV = 5
# # = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# for model in models:
#   model_name = model.__class__.__name__
#   accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
#   for fold_idx, accuracy in enumerate(accuracies):
#     entries.append((model_name, fold_idx, accuracy))
#
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])




# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df,
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()
#
# print(cv_df.groupby('model_name').accuracy.mean())

# def top_n_accuracy(preds, truths, n):
#     best_n = np.argsort(preds, axis=1)[:,-n:]
#     ts = np.argmax(truths, axis=1)
#     successes = 0
#     for i in range(ts.shape[0]):
#       if ts[i] in best_n[i,:]:
#         successes += 1
#     return float(successes)/ts.shape[0]

#

X_train, X_test, y_train, y_test = train_test_split(df['Lyrics'], df['Genre'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

values = clf.predict_proba(count_vect.transform(["I touch my eyes, i`m dreaming for end, and enchanting woods is only a dream, I feel the sun my eyes will burn, for my soul I see anything."]))
# print(clf.predict_proba(count_vect.transform(["I touch my eyes, i`m dreaming for end, and enchanting woods is only a dream, I feel the sun my eyes will burn, for my soul I see anything."])))
print(values)
print(clf.classes_)
order = values.argsort()[-2:]
print(order)
print("prima " + clf.classes_[order[0][0]])
print("a doua " + clf.classes_[order[0][1]])
# print(clf.predict_proba(count_vect.transform(["Love Angels"])[2]))