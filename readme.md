# **1º Challenge de Data Science Alura** 

![Badge em Desenvolvimento](http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge)


Neste repositório estão os códigos, arquivos e resultados da análise de dados da empresa [Alura Voz](https://www.alura.com.br/challenges/data-science). O projeto foi desenvolvido durante os meses de Maio até Julho de 2022.


## Sumário 
**[1. Descrição do projeto](#1-descrição-do-projeto)**

**[2. Extração, limpeza e tratamento dos dados](#2-extração-limpeza-e-tratamento-dos-dados)**

- **[2.1 - Avaliação dos dados](#21---avaliação-dos-dados)**
- **[2.2 - Inspeção e limpeza dos dados](#22---inspeção-e-limpeza-dos-dados)**
- **[2.3 - Padronização dos dados](#23---padronização-dos-dados)**

---

# **1. Descrição do projeto** 

Neste projeto, é desenvolvido um conjunto de análises e modelos de machine learning supervisionados, para auxiliar na fidelização dos seus clientes de maneira mais assertiva. As atividades a serem realizadas são:
- Extração, limpeza e tratamento de dados.
- Análise exploratória e quantitativa dos dados.
- Interpretação dos dados e levantamento de hipóteses com bases nas análises.
- Criação de modelos de machine learning supervisionado para predizer a tendência de um novo cliente pedir cancelamento do plano.


# **2. Extração, limpeza e tratamento dos dados**

## **2.1 - Avaliação dos dados**
Esta é a primeira etapa a ser realizada do projeto. A base de dados é obtida por meio do seguinte [link](https://github.com/sthemonica/alura-voz/blob/main/Dados/Telco-Customer-Churn.json). Para melhor descrever a base de dados, a tabela a seguir exibe o dicionário dos dados fornecidos.

Nome da coluna | Dicionário da empresa
-------|------------------
`customerID`| Código único para o cliente da empresa
`Churn`| se o cliente deixou ou não a empresa
`gender`| gênero (masculino e feminino)
`SeniorCitizen`| informação sobre um cliente ter ou não idade igual ou maior que 65 anos
`Partner`| se o cliente possui ou não um parceiro ou parceira
`Dependents`| se o cliente possui ou não dependentes
`tenure`| meses de contrato do cliente
`PhoneService`| assinatura de serviço telefônico
`MultipleLines`| assinatura de mais de uma linha de telefone
`InternetService`| assinatura de um provedor internet
`OnlineSecurity`| assinatura adicional de segurança online
`OnlineBackup`| assinatura adicional de backup online
`DeviceProtection`| assinatura adicional de proteção no dispositivo
`TechSupport`| assinatura adicional de suporte técnico, menos tempo de espera
`StreamingTV`| assinatura de TV a cabo
`StreamingMovies`| assinatura de streaming de filmes
`Contract`| tipo de contrato
`PaperlessBilling`| se o cliente prefere receber online a fatura
`PaymentMethod`| forma de pagamento
`Charges.Monthly`| total de todos os serviços do cliente por mês
`Charges.Total`| total gasto pelo cliente

## **2.2 - Inspeção e limpeza dos dados**

Sabendo das variáveis presente nos dados, a próxima etapa é realizar a inspeção de seus valores. Usando a função *df.info()*, têm-se as seguintes ifnormações do dataframe

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7267 entries, 0 to 7266
Data columns (total 21 columns):
Column           Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7267 non-null   object 
 1   Churn             7267 non-null   object 
 2   gender            7267 non-null   object 
 3   SeniorCitizen     7267 non-null   int64  
 4   Partner           7267 non-null   object 
 5   Dependents        7267 non-null   object 
 6   tenure            7267 non-null   int64  
 7   PhoneService      7267 non-null   object 
 8   MultipleLines     7267 non-null   object 
 9   InternetService   7267 non-null   object 
 10  OnlineSecurity    7267 non-null   object 
 11  OnlineBackup      7267 non-null   object 
 12  DeviceProtection  7267 non-null   object 
 13  TechSupport       7267 non-null   object 
 14  StreamingTV       7267 non-null   object 
 15  StreamingMovies   7267 non-null   object 
 16  Contract          7267 non-null   object 
 17  PaperlessBilling  7267 non-null   object 
 18  PaymentMethod     7267 non-null   object 
 19  Monthly           7267 non-null   float64
 20  Total             7267 non-null   object 
dtypes: float64(1), int64(2), object(18)
memory usage: 1.2+ MB
```

Apesar de não apresentar nenhum valor não nulo, algumas linhas podem estar sendo preenchidas com caracteres do tipo em branco. Assim, por garantia, é realizada uma última inspeção rodando o seguinte código:

```python
# Verificação de dados em branco
for column in df.columns:
  print('coluna {0}: {1} vazios. '.format( column, df.loc[ (df[column] == '') | (df[column].isnull()) | (df[column] == ' ') ][column].count()))
```
Obtendo assim a seguinte saída:

```python
coluna customerID: 0 vazios. 
coluna Churn: 224 vazios. 
coluna gender: 0 vazios. 
coluna SeniorCitizen: 0 vazios. 
coluna Partner: 0 vazios. 
coluna Dependents: 0 vazios. 
coluna tenure: 0 vazios. 
coluna PhoneService: 0 vazios. 
coluna MultipleLines: 0 vazios. 
coluna InternetService: 0 vazios. 
coluna OnlineSecurity: 0 vazios. 
coluna OnlineBackup: 0 vazios. 
coluna DeviceProtection: 0 vazios. 
coluna TechSupport: 0 vazios. 
coluna StreamingTV: 0 vazios. 
coluna StreamingMovies: 0 vazios. 
coluna Contract: 0 vazios. 
coluna PaperlessBilling: 0 vazios. 
coluna PaymentMethod: 0 vazios. 
coluna Monthly: 0 vazios. 
coluna Total: 11 vazios.
```
Com isso, sabe-se agora que a coluna *Churn* e *Total* possuem valores em branco/nulos. Como a quantidade é relativamente pequena (224 e 11) em relação ao total de registros (7267), então opta-se pelo descarte de tais linhas em branco.

Em adição, a coluna *customerID* também é descartada pois não haverá necessidade de identificar um determinado cliente nas análises seguintes.

## **2.3 - Padronização dos dados**

Este última etapa consiste em padronizar os dados. A primeira tarefa é de traduzir as colunas, de modo a facilitar a leitura e interpretação dos dados para o leitor que não esteja familiarizado com a língua inglesa. Após a tradução das colunas, são feitas duas cópias. A primeira cópida do dataframe contém os valores escritos em forma de texto (Yes/No) para facilitar na análise gráfica e relacional dos dados. A segunda cópia contém os dados em forma numérica (1/0) para facilitar na análise e treinamento dos algoritmos de aprendizado de máquina.

Mais informações e detalhes dos procedimentos realizados pode ser obtidos no [Notebook 1](https://github.com/goto-95/Challenge_DataScience_AluraVoz/blob/main/Challenge_DS_Alura_Semana01.ipynb) criado para esta primeira etapa

# **3. Análise gráfica e relacional dos dados**

Realizado o tratamento dos dados e a exportação em um arquivo *csv* mais simples, a próxima etapa é avaliar a relação que as variáveis têm entre si e, principalmente, com a taxa de cancelamento (descrito pela coluna `Churn` ou `Status_Cliente` na sua versão traduzida).

## **3.1 - Análise correlacional**

O primeiro é avaliar qual é o grau de correlação que as variáveis têm entre si. Para isto, é feita o carregamento do dataframe com os valores numérico [dados_tratados_obj.csv](https://github.com/goto-95/Challenge_DataScience_AluraVoz/blob/main/dados_tratados_num.csv). A matriz de correlação obtida é exibida abaixo. As relações entre as variáveis são apresentadas na forma gráfica utilizando círculos e elispses. Quanto maior a correlação entre duas variáveis, negativa ou positiva, maior será a distorção da elipse (tendendo a uma reta para o caso de `df.corr()= abs(1)`). Por outro lado, quanto menor for a correlção, mais próoximo de um círculo será o gráfico. As variáveis que contém uma correlação positiva entre si, apresentam uma elipse azul. Enquanto que as variáveis com correlação negativa apresentam uma elipse vermelha.

Com base no gráfico apresentado na Fig. 3.1, é possível observar que:

- Os clientes que possuem ***assinatura de TV*** também tendem a assinar os serviços de ***backup***, ***suporte técnico***, ***internet***, ***seguro online***, ***seguro do dispositivo móvel*** e o de ***filmes.***
- Em particular, os clientes que assinam os serviços de ***seguro,*** tanto o online quanto o do dispositivo móvel, tendem a preferir pela a optar pela cobrança online.
- Os clientes que assinam os serviços de telefone, tanto o da linha normal quanto o multi-linha, não apresentam interesse em adquirir os demais serviços
- O perfil dos clientes inativos indicam uma significativa correlação negativa com o valor anual do plano. Enquanto que apresentam uma correlação positiva com o valor mensal. Este comportamento precisa ser avaliado com mais detalhes.
- A coluna de clientes inativos apresenta uma correlação negativa com a coluna de ***Meses_Contrato*** que indica o tempo do contrato em meses. Enquanto apresenta uma correlação positiva com a coluna ***dependentes***.
<figure>
    <img src= "Imagens-Analise_Grafica/corrplot.png" />
    <figcaption align = 'center'>
        <b> Figura 3.1: Matriz de correlação das variáveis da base de dados. </b>
    </figcaption>
</figure>


## **3.2 - Análise gráfica de evasão de clientes**

A próxima etapa é a avaliação do perfil dos clientes para entender o percentual de evasão, a relação do cancelemento com o perfil do cliente e coms os serviços prestados pela empresa.

### **Taxa total de evasão**
Primeiramente é avaliado a taxa total de evasão dos clientes. Como observado no gráfico da Fig. 3.2, é possível observar que o percentual de cancelamento é de em torno de 26.6%, o que é considerado relativamente alto.

<figure>
    <img src = "Imagens-Analise_Grafica/barplot_percentual_evasao.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.2: Percentual de evasão total dos clientes. </b>
    </figcaption>
</figure>

### **Perfil de gastos dos clientes**
A primeira hipótese está no fator financeiro. Avalia-se se o custo dos planos fornecidos possam influenciar na tendência de cancelamento. O primeiro gráfico, Fig. 3.3 avalia os valores dos planos mensais. Como é possível observar que:

- Os planos entre $60,00 e $100,00 apresentam maiores índices de cancelamento, maiores até que a média geral.
- Os planos abaixos de $20,00 apresentam a menor taxa de evasão, cerca de 10%. 

<figure>
    <img src = "Imagens-Analise_Grafica/hist_custo_mensal.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.3: Histograma dos valores dos planos mensais agrupados por status dos clientes. </b>
    </figcaption>
</figure>

A média dos planos mensais é apresentada na Fig. 3.4 confirma a hipótetse inicial que o fator financeiro têm forte influência no processo de cancelamento. Em síntese, observa-se que os valores dos planos mensais dos clientes que pediram cancelamento é maior que os clientes que não pediram cancelamento. Sugere-se que os clientes inativos tenham migrado para planos mais baratos da concorrência.

<figure>
    <img src = "Imagens-Analise_Grafica/barplot_custo_mensal_medio.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.4: Gráfico de barras dos valores médios dos planos mensais agrupados por status dos clientes. </b>
    </figcaption>
</figure>

Agora para os valores totais dos planos, Fig. 3.5, têm-se que os planos mais baratos apresentam a taxa de cancelamento de 37%. Isto pode indicar que, para os planos anuais, os clientes que aderem a menos serviços tem maior tendência de pedir cancelamento.  

<figure>
    <img src = "Imagens-Analise_Grafica/hist_custo_total.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.5: Histograma dos valores dos planos totais agrupados por status dos clientes. </b>
    </figcaption>
</figure>

Para confirmar tal avaliação, a Fig. 3.6 exibe o gráfico de barra dos valores médios dos planos anuais separados por status dos clientes. Nota-se que, de fato, os clientes com planos anuais mais baratos tem a maior chance de pedir o cancelamento.  

<figure>
    <img src = "Imagens-Analise_Grafica/barplot_custo_total_medio.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.6: Gráfico de barras dos valores médios dos planos anuais agrupados por status dos clientes. </b>
    </figcaption>
</figure>

### **Tempo de permacência dos clientes**

Sabendo das tendências dos clientes em relação a cada tipo de plano, agora deseja-se avaliar o tempo de permanência dos clientes com a empresa. No histograma de tempo total, Fig. 3.7, a maior parte dos clientes que pediram cancelamento ficaram menos de2 anos com a empresesa. Em especial, cerca de 50% do clientes pediram cancelamento antes do primeiro ano. 

<figure>
    <img src = "Imagens-Analise_Grafica/hist_duracao_plano.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.7: Histograma do tempo de contrato (em meses) dos planos agrupados por status dos clientes. </b>
    </figcaption>
</figure>

Olhando com mais detalhes no primeiro ano de contrato, Fig. 3.8, observa-se que cerca de 62% dos clientes pedem cancelamento no primeiro mês de serviço. Isto pode indicar uma falha na instalação, cobrança ou atendimento que comprometam a satisfação do cliente ao aderir ao plano. A medida que o tempo de contrato vai aumentando, o percentual de clientes que vão pedindo o cancelamento vai diminuindo e se aproximando da média, que é de 26%.

<figure>
    <img src = "Imagens-Analise_Grafica/hist_duracao_plano_1ano.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.8: Histograma dos primeiros 12 meses dos planos agrupados por status dos clientes. </b>
    </figcaption>
</figure>

## **3.3 - Análise da qualidade dos serviços prestados**
Esta etapa tem por objetivo entender se a qualidade dos serviços fornecidos pela empresa têm influência na taxa de cancelamento. Como a empresa dispõem de disversos serviços, serão mostrados os resultados mais importantes.

### **Serviço de TV**

Para o serviço de assinatura de TV, os dados são descrito como clientes que possuem ou não o serviço e os clientes que não possuem o serviço de internet. Pela Fig. 3.9, observa-se que a taxa de cancelamento entre os cliente que possuem ou não o serviço de TV apresentam valores próximos entre si, em torno de 30 à 33%. Enquanto os clientes que não tem o serviço de internet apresentam uma taxa de cancelamento muito menor, em torno de 7.4%. Isto indica que o serviço de internet fornecido pode não ser satisfatório. A próxima análise irá verificar esta hipótese.

<figure>
    <img src = "Imagens-Analise_Grafica/barplot_tv.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.9: Gráfico de barras categórico para o fornecimento do serviço de assinatura de TV.</b>
    </figcaption>
</figure>

### **Serviço de internet**

Para o serviço de internet, Fig. 3.10,  observa-se que a taxa de cancelamento para o serviço de internet via fibra óptica é consideralvemente alta (cerca de 42%) comparada ao fornecimento via DSL. Isto indica que a qualidade da internet via fibra óptica não está atendendo aos clientes.

<figure>
    <img src = "Imagens-Analise_Grafica/barplot_internet.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.10: Gráfico de barras categórico para o fornecimento do serviço de assinatura de internet.</b>
    </figcaption>
</figure>

### **Serviço de suporte técnico**

Para o serviço de suporte técnico, Fig. 3.11, observa-se que a taxa de cancelamento é significativamente alto (cerca de 42%) para os planos que não possuem este serviço. Enquanto que a taxa de cancelamento para os planos com o suporte é menor, cerca de 16%.

<figure>
    <img src = "Imagens-Analise_Grafica/barplot_suporte_tecnico.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.11: Gráfico de barras categórico para o fornecimento do serviço de assinatura do serviço de suporte técnico.</b>
    </figcaption>
</figure>

## **3.4 - Forma de pagamento**
Também pe verificado se há alguma relação entre a forma de pagamento e a taxa de cancelamento. Com base na Fig. 3.12, as formas de pagamento via boleto, credit card e transferência bancária apresentam a taxa de cancelamento menor que a média geral. Enquanto que a forma de boleto eletrônico apresenta uma taxa significativamente maior, em torno de 46%. Isto pode ser um indicador a possíveis problemas de geração de boletos online.

<figure>
    <img src = "Imagens-Analise_Grafica/barplot_forma_pagamento.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.12: Gráfico de barras categórico para o a forma de pagamento escolhido pelo cliente.</b>
    </figcaption>
</figure>

## **3.5 - Tipo de contrato**
Outro aspecto observado é o tipo de contrato escolhido pelo cliente. Pelo Fig. 3.13, foi observado que os clientes que optam pelo plano mensal tem mais chances de pedir o cancelamento. A taxa de evasão para este tipo de plano é de em torno de 43%.

<figure>
    <img src = "Imagens-Analise_Grafica/barplot_tipo_contrato.png"  />
    <figcaption align = 'center'>
        <b> Figura 3.13: Gráfico de barras para o tipo de contrato escolhido pelo cliente.</b>
    </figcaption>
</figure>


