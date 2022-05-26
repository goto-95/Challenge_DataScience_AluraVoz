# **1º Challenge de Data Science Alura** 

![Badge em Desenvolvimento](http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge)


Neste repositório estão os códigos, arquivos e resultados da análise de dados da empresa [Alura Voz](https://www.alura.com.br/challenges/data-science). O projeto foi desenvolvido durante os meses de Maio até Julho de 2022.


## Sumário 
1. [**Descrição do projeto**](#1-descrição-do-projeto)
2. [**Extração, limpeza e tratamento dos dados**](#2-extração-limpeza-e-tratamento-dos-dados) 


# **1. Descrição do projeto** 

Neste projeto, é desenvolvido um conjunto de análises e modelos de machine learning supervisionados, para auxiliar na fidelização dos seus clientes de maneira mais assertiva. As atividades a serem realizadas são:
- Extração, limpeza e tratamento de dados.
- Análise exploratória e quantitativa dos dados.
- Interpretação dos dados e levantamento de hipóteses com bases nas análises.
- Criação de modelos de machine learning supervisionado para predizer a tendência de um novo cliente pedir cancelamento do plano.


# **2. Extração, limpeza e tratamento dos dados**

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



<img src= "Imagens-Analise_Grafica/barplot_aposentado.png" />
