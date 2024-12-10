import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd
import huspacy
from string import punctuation
from spacy.lang.hu import stop_words
from sklearn.neural_network import MLPClassifier
from prettytable import PrettyTable
from wordcloud import WordCloud
import spacy
from spacy import displacy
from sklearn import metrics

from sklearn.metrics import ConfusionMatrixDisplay

nlp = huspacy.load()
stop_words = set(stop_words.STOP_WORDS)
punctuations = set(punctuation)

def tokenize(sentence):
    #print("sentence before tokenize:", sentence)
    sentence = nlp(sentence)
    sentence = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in sentence]
    sentence = [word for word in sentence if word not in stop_words and word not in punctuations]
    sentence = " ".join(sentence).strip()    
    #print("sentence after tokenize: ",sentence)
    
    return sentence

sentenceExample = nlp("Talán ideje lenne fogyni, megszabadulni attól a sok plusz kilótól.")
#displacy.serve(sentenceExample, style="dep",auto_select_port=True)
print("sentence before tokenize:", sentenceExample)
sentenceafter = tokenize(sentenceExample)
print("sentence after tokenize: ",sentenceafter)

df = pd.read_csv('bodyShamingComments.txt', sep='\t')

print(df.head())

print(df.isna().sum())
labels = df['label'].unique()
print(labels)
df = df.dropna()
labels = df['label'].unique()
print(df['label'].value_counts())
df['comment_length'] = df['comment'].apply(len)
lens = df['comment_length']
print("karakterek")
print(lens.mean(), lens.std(), lens.max())
print("----------------")
print("szavak")
df['comment_length_words'] = df['comment'].apply(lambda x: len(x.split()))
label_counts = df['label'].value_counts()
lens = df['comment_length_words']
print(lens.mean(), lens.std(), lens.max())
df['filtered_and_lemmatized_comments'] = ""

df['filtered_and_lemmatized_comments'] = df['comment'].apply(tokenize)

#count by label 
plt.figure(figsize=(10, 6))
sns.barplot(x=label_counts.index, y=label_counts.values, palette='tab10')
plt.title('Az egyes megszégyenítő címkék száma összesen', fontsize=16)
plt.xlabel('Megszégyenítés címke')
plt.ylabel('Darab')
plt.show()

plt.xscale('log')
bins = 1.15**(np.arange(0,50))
for l in labels:
    plt.hist(df[df['label']==l]['comment_length'], bins=bins, alpha = 0.8, rwidth=0.85)
plt.legend(labels)
plt.title("Megjegyzések hosszának eloszlása")
plt.xlabel("Megjegyzések hossza")
plt.ylabel("Gyakoriság")
plt.show()

toxic_comments = ' '.join(df['filtered_and_lemmatized_comments'])

plt.figure(figsize=(12, 8))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(toxic_comments)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("A megszégyenítő megjegyzések szófelhője")
plt.show()

plt.figure(figsize=(10, 6))
sns.distplot(df['comment_length_words'], bins=50, kde=True, color='purple')
plt.title("Megjegyzésekben szereplő szavak számának eloszlása")
plt.xlabel("Szavak száma a megjegyzésben")
plt.ylabel("Frekvencia")

plt.show()

plt.figure(figsize=(10, 6))
sns.distplot(df['comment_length'], bins=50, kde=True, color='blue')
plt.title("Megjegyzések karakterszámának eloszlása")
plt.xlabel("Karakterek száma a megjegyzésben")
plt.ylabel("Frekvencia")

plt.show()

X = df['filtered_and_lemmatized_comments']
y = df['label']


vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=0)

smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

#logisztikus regresszió
logclf = LogisticRegression()
logclf.fit(X_train_smote, y_train_smote)

y_pred = logclf.predict(X_test)
print("Logistic regression")
print(classification_report(y_test, y_pred))
print("------------------------------------")
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
cm_display.plot()
plt.title("Logisztikius regresszió tévesztési mátrix")
plt.xlabel("Prediktált címkék")
plt.ylabel("Valódi címkék")
plt.show()

#mlp
sc=StandardScaler(with_mean=False)

scaler = sc.fit(X_train)
trainX_scaled = scaler.transform(X_train)
testX_scaled = scaler.transform(X_test)

mlp_clf = MLPClassifier(hidden_layer_sizes=(100,50,25),
                        max_iter = 300,activation = 'relu',
                        solver = 'adam', random_state=0)

#mlp_clf.fit(trainX_scaled, y_train)
mlp_clf.fit(X_train_smote, y_train_smote)
#y_pred = mlp_clf.predict(testX_scaled)
y_pred = mlp_clf.predict(X_test)

print("MLP")
print(classification_report(y_test, y_pred))
print("------------------------------------")
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
cm_display.plot()
plt.title("MLP tévesztési mátrix")
plt.xlabel("Prediktált címkék")
plt.ylabel("Valódi címkék")
plt.show()


#-------------------------------mondat alapú
from sentence_transformers import SentenceTransformer

#model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('SZTAKI-HLT/hubert-base-cc')

sentences = df['filtered_and_lemmatized_comments']

sentence_embeddings = model.encode(sentences)

#print(sentence_embeddings)
X = sentence_embeddings
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

clf = LogisticRegression()
clf.fit(X_train_smote, y_train_smote)

y_pred = clf.predict(X_test)

print("SBERT - Logistic regression")
print(classification_report(y_test, y_pred))
print("------------------------------------")
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
cm_display.plot()
plt.title("SBERT + logisztikus regresszió tévesztési mátrix")
plt.xlabel("Prediktált címkék")
plt.ylabel("Valódi címkék")
plt.show()

mlp_clf_sentence = MLPClassifier(hidden_layer_sizes=(100,50,25),
                        max_iter = 300,activation = 'relu', random_state=0,
                        solver = 'adam')

mlp_clf_sentence.fit(X_train_smote, y_train_smote)

y_pred = mlp_clf_sentence.predict(X_test)

print("SBERT - MLP")
print(classification_report(y_test, y_pred))
print("------------------------------------")
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = labels)
cm_display.plot()
plt.title("SBERT + MLP tévesztési mátrix")
plt.xlabel("Prediktált címkék")
plt.ylabel("Valódi címkék")
plt.show()

from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(
    estimators=[('lr', logclf), ('mlp', mlp_clf), ('slr', clf), ('smlp', mlp_clf_sentence)],
    voting='soft'
)

ensemble_model.fit(X_train_smote, y_train_smote)
y_pred_ensemble = ensemble_model.predict(X_test)

print('Ensemble Model')
print(classification_report(y_test, y_pred_ensemble))

t = PrettyTable(['comment', 'Logistic Regression', 'MLP', 'SBERT + Logistic Regression', 'SBERT + MLP'])

 # Kategória előrejelzése
def predict(new_text):
    filtered_and_lemmatized_sentence_sentence = tokenize(new_text)
    new_vector = vectorizer.transform([filtered_and_lemmatized_sentence_sentence])
    predicted_categoryLOG = logclf.predict(new_vector)
    new_text_scaled = scaler.transform(new_vector.toarray())  
    predicted_categoryMLP = mlp_clf.predict(new_text_scaled)
    new_sentence_embedding = model.encode([filtered_and_lemmatized_sentence_sentence])
    predictionSentence = clf.predict(new_sentence_embedding)
    predictionSentenceMLP = mlp_clf_sentence.predict(new_sentence_embedding)
    
    t.add_row([new_text, predicted_categoryLOG[0],predicted_categoryMLP[0],predictionSentence[0],predictionSentenceMLP[0]])

    #print(f"{new_text}\t{predicted_categoryLOG[0]}\t{predicted_categoryMLP[0]}\t{predictionSentence[0]}\t{predictionSentenceMLP[0]}")
    

testcomments = ["Ha nem lennél ilyen dagadt akkor talán nem kellene két jegyet venned a repülőre", "már már szánalmasnak tűnsz, mindenki legalább egy fejjel magasabb nálad",
                "Értem, miért részesíted előnyben az alacsony bútorokat!", "miért nem próbálod fehéríteni a bőröd? ", "szőrös majom vagy", "el ne fújjon téged a szél te kóró" ]

for c in testcomments:
    predict(c)

print(t)





