# UnderSampling Techniques

---

# Sumário

---

# 1. Função **UnderSampling** da biblioteca **imblearn**

A função **[UnderSampling**](https://imbalanced-learn.org/stable/under_sampling.html) tem por finalidade diminuir a desigualdade da distribuição entre as variáveis de análise. Esta técnica consiste em diminuir a proporção entre duas respostas (sim/não ou 1/0) de 1:100 para 1:10 ou até menos ao deletar algumas variáveis do grupo majoritário.  

Outra técnica bastante conhecida é o **OverSampling** onde o próopóosito é aumentar a proporção das variáveis ao incrementar a quantidade das variáveis minoritárias.

A técnica de *UnderSampling* pode ser usada diretamente nos dados de treino para ser ajustado no algoritmo de Machine Learning. Uma boa prática é combinar a téxcnica de **UnderSampling** com a técnica de **OverSampling** para promover uma performance melhor. Esta costuma ser uma boa prática pois a técnica de UnderSampling mais comum é o descarte aleatório de dados do grupo marjoritário. O que, não leva em consideração nenhuma informação a respeito do dado e não há nenhum parâmetros para estabelecer se o dado será descartado ou não. 

Uma saída para esta limitação é a utilização de técnicas de heurísticas e aprendizado de modelo para tentar identificar dados redundantes do grupo marjoritário. Tais técnicas serão avaliadas a seguir:  

# 2. Técnicas de balanceamento **UnderSampling**

## 2.1 - Near Miss UnderSampling

[**Near Miss](https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.NearMiss.html)** é um conjunto de técnicas UnderSampling the seleciona os exemplos baseado na distância dos dados marjoritários. Existem 3 versões desta técnica:

→ **NearMiss-1**: Seleciona os exemplos da classe marjoritária que possuem a menor distância média com os três exemplos mais próximos da classe minoritária.

→ **NearMiss-2**: Seleciona os exemplos da classe marjoritária que possuem a menor distância média com os três exemplos mais distantes da classe minoritária.

→ **NearMiss-3**: Selleciona um dado número de exemplos da classe marjoritária para cada exemplo da classe minoritária próximo

```python
from imblearn.under_sampling import NearMiss   
versao = 1
n=3
undersample = NearMiss(version=versao, n_neighbors=n) 
```

```python
# Undersample imbalanced dataset with NearMiss
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NearMiss
from matplotlib import pyplot
from numpy import where

# define an imbalaced dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# summarize class distribution
counter = Counter(y)
print(counter)

# define the undersampling method
vs = 1 # Near Miss version 1, 2 or 3
undersample = NearMiss(version=vs, n_neighbors=3)

# transform the dataset
X, y = undersample.fit_resample(X, y)

# summarize the new class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

## 2.2 - Condensed Nearest Neighbor Rule UnderSampling

O método **[Condensed Nearest Neighbor](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.CondensedNearestNeighbour.html)** (CNN) é uma técnica de de UnderSampling que procura o conjunto de amostras de uma classe que resulta em perda mínima da performance do modelo, referenciado como conjunto minimamente consistente. 

O método consiste em enumerar os exemplos e adicionar em um banco apenas se não for possível classificar-lo corretamente pelos exemplos presente neste banco. Esta abordagem foi inicialmente proposta para reduzir a memória necessária para o algoritmo k-Nearest Neighbors (KNN). 

Quando aplicado para classificação desbalanceada, o banco é preenchido de todos os exemplos da classe minoritária e apenas os exemplos da classe majortitária que não podem ser corretamente classificadas são adicionadas a este banco. Isto permite reduzir exemplos semelhantes e redundantes, do ponto de vista computacional. 

```python
from imblearn.under_sampling import CondensedNearestNeighbour

undersample = CondensedNearestNeighbour(n_neighbors=1)
```

```python
# Undersample and plot imbalanced dataset with the Condensed Nearest Neighbor Rule
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import CondensedNearestNeighbour
from matplotlib import pyplot
from numpy import where

# define an imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# summarize class distribution
counter = Counter(y)
print(counter)

# define the undersampling method
undersample = CondensedNearestNeighbour(n_neighbors=1)

# transform the dataset
X, y = undersample.fit_resample(X, y)

# summarize the new class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

# 3. Técnicas de exclusão de exemplos

Nesta seção iremos ver algumas técnica de exclusão de exemplos da classe majoritária.

## 3.1 - Tomek Links for UnderSampling

A maior crítica ao método Condensed Nearest Neighbor Rule é que os exemplos são selecionados aleatoriamente, o que pode levar a perda de informação relevante ao modelo.

Uma modificação proposta por Ivan Tomek sugere uma regra que procure pares de exemplos, um de cada classe, que tenham a menor distância euclidiana entre um e outro no espaço amostral. Em outras palavras, as regras de seleção dos pares são, dados dois exemplos *a* e *b*:
→ (i)   O vizinho mais próximo de *a* é o exemplo *b*;
→ (ii)  O vizinho mais próximo de *b* é o exemplo *a*;
→ (iii) Os exemplos *a* e *b* pertencem a classes diferentes.  

Por mais contra intuitivo que possa parecer, esta regra permite definir o contorno ou limite de cada classe.  

O método **[Tomek Link](https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.TomekLinks.html)** pode ser utilizado para localizar todos os exemplos que cruzam os limites das classes. Se, por exemplo, os exemplos da classe minoritária forem aproximadamente constantes, o método irá encontrar todos os exemplos da classe marjoritária próximos do contorno/limite da classe minoritária e remove-los. Estes exemplos a serem deletados seriam os exemplos ambíguos. 

Observe que esta técnica não irá transformar os dados, ela apenas irá excluir os dados ambíguos. 

```python
from imblearn.under_sampling import TomekLinks

undersample = TomekLink()
```

```python
# Undersample and plot imbalanced dataset with Tomek Links
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import TomekLinks
from matplotlib import pyplot
from numpy import where

# define an imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# summarize class distribution
counter = Counter(y)
print(counter)

# define the undersampling method
undersample = TomekLinks()

# transform the dataset
X, y = undersample.fit_resample(X, y)

# summarize the new class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

## 3.2 - Edited Nearest Neighbors Rule for UnderSampling

Outra técnica de exclusão de exemplos ambíguos é o **[Edited Nearest Neighbors](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.EditedNearestNeighbours.html)** (ENN). Esta regre consiste em setar *k=3* nearest neighbors para localizar os exemplos do dataset que são classificados incorretamente e que são removidos antes de setar *k=1* classitication rule.   

Quando utilizado para UnderSampling, a regra pode ser aplicado para cada exemplo na classe marjoritária, permitindo que tais exemplos que são classificados incorretamente (como pertencentes a outra classe) sejam removidos. Este procedimento também pode ser aplicado para cada exemplo na classe minoritária.

Assim como o método Tomek Link, este método não transforma os dados. Apenas elimina os dados ambíguos. 

```python
from imblearn.under_sampling import EditedNearestNeighbours

undersample = EditedNearestNeighbors(n_neighbors=3)
```

```python
# Undersample and plot imbalanced dataset with the Edited Nearest Neighbor rule
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import EditedNearestNeighbours
from matplotlib import pyplot
from numpy import where

# define an imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# summarize class distribution
counter = Counter(y)
print(counter)

# define the undersampling method
undersample = EditedNearestNeighbours(n_neighbors=3)

# transform the dataset
X, y = undersample.fit_resample(X, y)

# summarize the new class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

# 4. Combinação dos métodos Keep and Delete

## 4.1 - One-Sided Selection for UnderSampling

O método One-Sided Selection (OSS) é uma combinação dos métodos Tomek Link e Condensed Nearest Neighbor (CNN). 

O método OSS então  utiliza o Tomek Link para remover os exemplos ambíguos e depois utiliza o método CNN para remover os exemplos redundantes que estão distantes dos contornos das classes.

A função **[OneSidedSelection](https://imbalanced-learn.org/dev/references/generated/imblearn.under_sampling.OneSidedSelection.html)** pode ser importada na biblioteca imblearn.under_sampling e seus argumentos são: n_neighbors e n_seeds_S. O argumento n_neighbors já é conhecido da função CNN visto acima. Enquanto n_seeds_S indica o tamanho da amostragem avaliada. Como a avaliação da função CNN ocorre em apenas um bloco, é interessante configurar o valor de n_seeds_S com um valor relativamente alto.  

```python
from imblearn.under_sampling import OneSidedSelection

undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
```

```python
# Undersample and plot imbalanced dataset with One-Sided Selection
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import OneSidedSelection
from matplotlib import pyplot
from numpy import where
# define dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)
# summarize class distribution
counter = Counter(y)
print(counter)
# define the undersampling method
undersample = OneSidedSelection(n_neighbors=1, n_seeds_S=200)
# transform the dataset
X, y = undersample.fit_resample(X, y)
# summarize the new class distribution
counter = Counter(y)
print(counter)
# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

## 4.2 - Neighborhood Cleaning Rule for UnderSampling

O método Neighborhood Cleaning Rule (NCR) consiste na combinação do método ENS para o tratamento de exemplos ambíguos e o método CNN para o tratamento dos exemplos redundantes. 

Entretanto, diferentemente do método OSS, o método CNR esta mais focado na limpeza dos exemplos que necessariamente na remoção de exemplos redundantes. Portanto, o método NCR está mais focado no tratamento dos exemplos ambíguos.  

A função [**NeighbourhoodCleaningRule](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.NeighbourhoodCleaningRule.html)** da biblioteca imblearn.under_sampling possuem dois argumentor: n_neighbors e threshold_cleaning. O argumento threshold_cleaning contrala se o algoritmo CNN será aplicado ou não para uma determinada classe. O valor padrão é 0.5 e indica que uma determinada classe será avaliada pelo algoritmo CNN se ele tiver mais que 50% do número de exemplos da classe marjoritária.

```python
# Undersample and plot imbalanced dataset with the neighborhood cleaning rule
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import NeighbourhoodCleaningRule
from matplotlib import pyplot
from numpy import where

# define an imbalanced dataset
X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
	n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=1)

# summarize class distribution
counter = Counter(y)
print(counter)

# define the undersampling method
undersample = NeighbourhoodCleaningRule(n_neighbors=3, threshold_cleaning=0.5)

# transform the dataset
X, y = undersample.fit_resample(X, y)

# summarize the new class distribution
counter = Counter(y)
print(counter)

# scatter plot of examples by class label
for label, _ in counter.items():
	row_ix = where(y == label)[0]
	pyplot.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()
```

# Referências

[Undersampling Techniques](https://machinelearningmastery.com/undersampling-algorithms-for-imbalanced-classification/#:~:text=Undersampling%20refers%20to%20a%20group,has%20a%20skewed%20class%20distribution.)

[The Condesed nearest neighbor rule](https://ieeexplore.ieee.org/document/1054155)

[kNN Approach to Unbalanced Data Distribution](https://www.site.uottawa.ca/~nat/Workshop2003/jzhang.pdf)

[Two Modifications of CNN](https://ieeexplore.ieee.org/document/4309452)

[Asymptotic Properties of Nearest Neighbor Rules Using Edited Data](https://ieeexplore.ieee.org/document/4309137)

[An Experiment with the Edited Nearest-Neighbor Rule](https://ieeexplore.ieee.org/document/4309523)

[Improving Identification of Difficult Smal Classes by Balancing Class Distribution](https://link.springer.com/chapter/10.1007/3-540-48229-6_9)