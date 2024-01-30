# Neleptcu Daniel-Andrei 332AB

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = pd.read_csv("fetal_health.csv")

# toate feature-urile mai putin fetal_health
x = data.drop(['fetal_health'], axis=1)

# fetal_health - este ceea criteriul pe care incercam sa il clasificam
y = data['fetal_health']

# liste pentru a pastra acuratetile  si testsize-urile pe care le vom
# reprezenta grafic mai tarziu pentru a arata diferenta dintre seturile
# reduse de date si cele nereduse.
accuracies = []
testsizes = []
accuracies_reduced = []

for i in range(int(x.shape[0] / 10)):
    print(i)
    # apelez functia train_test_split pentru a imparti random in subseturi pentru antrenare si testare
    # variez paramentru test_size pentru a observa diferenta de acuratete in mod grafic
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(i + 1) * 10, random_state=0)

    # apelez clasificatorul RandomForestClassifier cu criteriul gini si 290 de estimatori, din mai multe teste
    # 290 este numarul care a produs cele mai bune rezultate.
    Model = RandomForestClassifier(criterion='gini', n_estimators=290, max_depth=4, random_state=0)

    # antrenam RFC pentru setul nostru de date de antrenare
    Model.fit(x_train, y_train)

    # calculam predictia
    y_pred = Model.predict(x_test)

    # calculam acuratetea in procentajul pe care acestea le reprezinta
    # din numarul total de cazuri
    AccScoreFinal = accuracy_score(y_test, y_pred, normalize=True)

    # adaug rezultatele si testsize-ul actual la listele declarate la inceput
    # pentru reprezentare grafica
    accuracies.append(AccScoreFinal)
    testsizes.append((i + 1) * 10)

print("TESTSIZES")
print(testsizes)
print("-------")
# reduc numarul de feature-uri "necesare" din setul meu de date eliminand cele care
# nu au o importanta la fel de ridicata ( din documentatia setului de date si explicatiile
# altor utilizatori care au folosit acest set de date )

x = data.drop(
    ['fetal_health', 'histogram_width', 'histogram_max', 'histogram_min', 'histogram_median', 'histogram_mean',
     'histogram_mode', 'baseline value', 'abnormal_short_term_variability'], axis=1)
y = data['fetal_health']

# aplic MinMaxScaler pentru a scala datele pe o scala de la 0 la 1
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
x = scaler.fit_transform(x)

# reduc datele folosind PCA pastrand o mare parte din variabilitatea datelor
pca = PCA(n_components=10)
x = pca.fit_transform(x)

# aplic acelasi algoritm ca si pentru datele reduse
for i in range(int(x.shape[0] / 10)):
    print(i)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(i + 1) * 10, random_state=0)

    Model = RandomForestClassifier(criterion='gini', n_estimators=290, max_depth=4, random_state=0)
    Model.fit(x_train, y_train)

    # calculam predictia
    y_pred = Model.predict(x_test)

    AccScoreFinal = accuracy_score(y_test, y_pred, normalize=True)
    accuracies_reduced.append(AccScoreFinal)


# reprezint grafic acuratetea pentru seturile de date fara preprocesare si
# cu preprocesare de date raportat la testsizes
plt.plot(testsizes, accuracies, label="Fara preprocesare", color='blue')
plt.plot(testsizes, accuracies_reduced, label="Cu preprocesare", color='green')
plt.xlabel('Testsize')
plt.ylabel('Accuracy')
plt.title('Dependenta intre testsize si acuratete')
plt.legend()
plt.show()

combo = zip(accuracies, testsizes)
combo_reduced = zip(accuracies_reduced, testsizes)
sorted_combo = sorted(combo, key=lambda x: x[0], reverse=True)
sorted_combo_reduced = sorted(combo_reduced, key=lambda x: x[0], reverse=True)

sorted_accuracies, sorted_testsize = zip(*sorted_combo)
sorted_accuracies_reduced, _ = zip(*sorted_combo_reduced)


x = data.drop(['fetal_health'], axis=1)
y = data['fetal_health']

print("Primele 3 cele mai bune acurateti pentru datele nereduse: ")
for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=sorted_testsize[i], random_state=0)

    Model = RandomForestClassifier(criterion='gini', n_estimators=290, max_depth=4, random_state=0)
    Model.fit(x_train, y_train)

    print('Model Train Score: ', Model.score(x_train, y_train))
    print('Model Test Score: ', Model.score(x_test, y_test))
    print('Importanta feature-uri model: \n', Model.feature_importances_)

    ModelSVM = SVC()
    ModelSVM.fit(x_train, y_train)
    ModelLog = LogisticRegression(solver='newton-cholesky', max_iter=10000, C=1e-2)
    ModelLog.fit(x_train, y_train)
    y_predSVM = ModelSVM.predict(x_test)
    y_predLog = ModelLog.predict(x_test)
    AccScoreFinalSVM = accuracy_score(y_test, y_predSVM, normalize=True)
    AccScoreFinalLog = accuracy_score(y_test, y_predLog, normalize=True)

    y_pred = Model.predict(x_test)

    print('Testsize: ', sorted_testsize[i])
    AccScore = accuracy_score(y_test, y_pred, normalize=False)
    print('Accuracy Score: ', AccScore)
    AccScoreFinal = accuracy_score(y_test, y_pred, normalize=True)
    print('Final Accuracy: ', AccScoreFinal)
    print('---')
    print('SVM Acc:', AccScoreFinalSVM)
    print('Log Acc:', AccScoreFinalLog)
    print('----------------------------------------------------')


x = data.drop(
    ['fetal_health', 'histogram_width', 'histogram_max', 'histogram_min', 'histogram_median', 'histogram_mean',
     'histogram_mode', 'baseline value', 'abnormal_short_term_variability'], axis=1)
y = data['fetal_health']

# aplic MinMaxScaler pentru a scala datele pe o scala de la 0 la 1
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
x = scaler.fit_transform(x)

# reduc datele folosind PCA pastrand o mare parte din variabilitatea datelor
pca = PCA(n_components=10)
x = pca.fit_transform(x)

print("Primele 3 cele mai bune acurateti pentru datele reduse: ")
for i in range(3):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=sorted_testsize[i], random_state=0)

    Model = RandomForestClassifier(criterion='gini', n_estimators=290, max_depth=4, random_state=0)
    Model.fit(x_train, y_train)

    print('Model Train Score: ', Model.score(x_train, y_train))
    print('Model Test Score: ', Model.score(x_test, y_test))
    print('Importanta feature-uri model: \n', Model.feature_importances_)

    ModelSVM = SVC()
    ModelSVM.fit(x_train, y_train)
    ModelLog = LogisticRegression(solver='newton-cholesky', max_iter=10000, C=1e-2)
    ModelLog.fit(x_train, y_train)
    y_predSVM = ModelSVM.predict(x_test)
    y_predLog = ModelLog.predict(x_test)
    AccScoreFinalSVM = accuracy_score(y_test, y_predSVM, normalize=True)
    AccScoreFinalLog = accuracy_score(y_test, y_predLog, normalize=True)

    y_pred = Model.predict(x_test)
    print('Testsize: ', sorted_testsize[i])
    AccScore = accuracy_score(y_test, y_pred, normalize=False)
    print('Accuracy Score: ', AccScore)
    AccScoreFinal = accuracy_score(y_test, y_pred, normalize=True)
    print('Final Accuracy: ', AccScoreFinal)
    print('---')
    print('SVM Acc:', AccScoreFinalSVM)
    print('Log Acc:', AccScoreFinalLog)
    print('----------------------------------------------------')
