# Classification Algorithms Techniques

Neste etapa, serão definidos os principais algoritmos de classificação supervisionados ([link](https://towardsdatascience.com/top-machine-learning-algorithms-for-classification-2197870ff501)).  Para isto, serão utilizados os seguintes algoritmos de classficação supervisionados**:**

1. **[Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html):**
    
    O Logistic Regression utiliza a função de Sigmoid para gerar a probabilidade de uma determinada classificação. Este algoritmo é largamente utilizado quando o dataset possui uma classificação binária (0/1, verdadeiro/falso, etc). 
    
    Para o caso de um dataset multiclasse, o algoritmo utiliza a abordagem um versus restante. É possível configurar isto definindo o parâmetro multi_class como ovr.  Também é possível ajustar o solver (solver), penalidade (penalty) e a distribuição da classe (class_weight). 
    
    ```python
    from sklearn.linear_model import LogisticRegression
    
    reg = LogisticRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    ```
    
2. **[Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html):** 
    
    O algoritmo Desicion Tree Classifier é um algoritmo de classificação onde as características mais importantes da dos dados vão sendo subdivididos em avaliações de verdadeiro ou falso. A partição dos dados permite montar um padrão de classificação por meio das ramificação da árvore, onde um determinado registro pode ser classificado ao ser comparado com a árvore de decisão.
    
    ```python
    from sklearn.tree import DecisionTreeClassifier
    
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    y_pred = dtc.predict(X_test)
    ```
    
3. **[Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html):** 
    
    O algoritmo Random Forest é uma combinação de vários Decisions Tree. Este algoritmo utiliza a técnica de sub-amostragem e aplica a média das amostras para melhorar a previsão e contralar o problema de overfiting. 
    
    ```python
    from sklearn.ensemble import RandomForestClassifier
    
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    ```
    
4. **[k-Neighbor Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html):**
    
    Este algoritmo utiliza a abordagem de distância entre os registro em um espeço vetorial correspondente. Registro que possuem uma distância média pequena tendem a ter a mesma classificação. Este tipo de algoritmo também é bastante utiliado como sistema de recomendação. 
    
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    ```
    
5. **[Gaussian Naiye-Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html):**
    
    O algoritmo Gaussian Navye-Bayes é baseado no Teorema de Bayes onde a probabilidade condicional é calculada com base em uma informação dada. A condição de Naive assume que as informações dos registros são independentes (outro motivo para a eliminação das variáveis altamente correlacionadas). A grande vantagem deste algoritmo é que, diferentemente dos demais algoritmos de aprendizado, este não necessita de um grande banco de dados para apresentar uma boa performance nas previsões.
    
    ```python
    from sklearn.naive_bayes import GaussianNB
    
    gnb = KNeighborsClassifier()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    ```
    
6. **[Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC):**
    
    O Support Vector Machine é um algoritmo onde utiliza de um hiper palno para separar os registros em duas classes: negativa e positiva. Este hiper plano é calculado com base na maximização da distância entre os pontos. Com o hiper plano definindo a borda entre as duas classes, o algoritmo realiza a previsão da classificação com base na posição de um determinado registro neste espaço vetorial.
    
    Este algortimo, assim como Decision Tree e o Random Forest, pode ser utilizado tanto para classificação quanto para regresão.
    
     
    
    ```python
    from sklearn.svm import SVC
    
    svc = SVC()
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    ```
    
7. **[Ada Boost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html):**
    
    O algoritmo Ada Boost é nada mais que um compilado de vários algoritmos de previsão mais simples (e menos eficientes) organizados de forma iterativa de modo a fornecer, no conjunto, um algoritmo de previsão com alta eficiência. A ideia é avaliar uma combinação linear entre os resultados entre vários sub-modelos e otimizar os pesos de tais modelos de forma a fornecer a maior acurácia possível. 
    
    ```python
    from sklearn.ensemble import AdaBoostClassifier
    
    abc = AdaBoostClassifier()
    abc.fit(X_train, y_train)
    y_pred = abc.predict(X_test)
    ```
    
8. [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html):
    
    De forma semelhante ao Ada Boost, o Gradient Boosting Classifier configura-se como um compilado de árvores de decisão (Decission Trees). 
    
    ```python
    from sklearn.ensemble import GradientBoostingClassifier
    
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    ```