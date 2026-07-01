# II. Desenhos

O desenho é a decisão de **como** atribuir o tratamento. Ele determina a
comparabilidade dos grupos e, consequentemente, qual inferência é válida depois.
Esta seção cobre o desenho completamente randomizado, a blocagem, a
rerandomização e o fatorial, com a matemática de cada um e exemplos numéricos.

---

## 4. CRD (Completely Randomized Design)

### Intuição

É o desenho mais simples: as unidades são atribuídas ao acaso, sem nenhuma
estrutura imposta. A randomização funciona como um seguro estatístico, pois
distribui igualmente, em média, todos os fatores externos (conhecidos ou não)
entre os braços.

### Formalização: randomização completa vs. Bernoulli

Há duas formas comuns de randomizar dois braços, e a distinção importa:

1. **Bernoulli**: cada unidade é tratada de forma independente com probabilidade
   `p`. O número de tratados é aleatório, `n_T \sim \text{Binomial}(N, p)`.
2. **Randomização completa**: fixa-se o número de tratados `n_T` e sorteia-se
   **qual** subconjunto de tamanho `n_T` recebe o tratamento, com todos os

   $$\binom{N}{n_T}$$

   subconjuntos igualmente prováveis.

A randomização completa tem duas vantagens práticas: garante o tamanho de cada
braço (sem o risco de um sorteio Bernoulli azarado deixar um braço minúsculo) e
tem variância um pouco menor para a diferença de médias. O conjunto de sorteios
possíveis (as `C(N, n_T)` alocações) é também a base da inferência por
randomização (ver [IV. Inferência](04-inferencia.md), tópico 13).

Na tradição clássica de DoE, a análise de um CRD costuma ser apresentada por um
modelo aditivo `y_{ti} = η + τ_t + ε_{ti}` e o teste F da ANOVA, sob erros
normais e homocedásticos. Essa é uma moldura **paramétrica** e histórica.

> **Na biblioteca (importante).** Dois pontos de alinhamento:
> 1. A `skxperiments` faz **randomização completa com número fixo de tratados**:
>    `CRD(p=0.5)` resolve `n_treated = round(p · N)` e sorteia exatamente essa
>    quantidade; não é Bernoulli por unidade. Use `CRD(p=...)` ou
>    `CRD(n_treated=...)`, mutuamente exclusivos.
> 2. A inferência da biblioteca é **baseada em randomização** ou na variância de
>    Neyman, e **não** assume erros normais nem usa o teste F da ANOVA. O modelo
>    aditivo acima é contexto, não pressuposto da lib.

### Exemplo trabalhado

Com `N = 8` unidades e `n_T = 4`, existem `C(8,4) = 70` alocações possíveis,
todas com a mesma probabilidade `1/70`. A randomização escolhe uma delas. A
"validade" do experimento vem justamente desse espaço de 70 alocações: a
distribuição da estatística de teste sob a nula é obtida varrendo essas
alocações (no `RandomizationTest`, por amostragem de Monte Carlo quando o número
é grande).

### Quando usar e quando não usar

Use o CRD quando as unidades são homogêneas e não há informação para agrupá-las;
é o ponto de partida de qualquer teste A/B. Evite quando há uma fonte conhecida
de variação que poderia ser controlada por blocagem, ou quando o ruído de fundo
é tão alto que o CRD fica insensível a efeitos reais.

---

## 5. Blocagem e estratificação

### Intuição

A regra prática é "bloqueie o que puder, randomize o que não puder". Se uma
variável categórica conhecida (região, dispositivo, lote) explica boa parte da
variação do resultado, agrupamos as unidades por ela em **blocos** e
randomizamos **dentro** de cada bloco. Assim a variação entre blocos sai do erro
experimental, e o teste fica mais sensível.

### Formalização: por que a precisão aumenta

Decomponha a variância total do resultado em uma parte **entre blocos** e uma
parte **dentro dos blocos**:

$$\sigma^2_{\text{total}} = \sigma^2_{\text{entre}} + \sigma^2_{\text{dentro}}.$$

No CRD, toda a variação (incluindo a entre blocos) entra no erro do estimador.
Na blocagem, comparamos tratamento e controle **dentro** de cada bloco, então a
variação entre blocos é removida do erro. O ganho de precisão é tanto maior
quanto mais a variável de bloco explica o resultado (quanto maior
`sigma²_entre`). No limite em que cada bloco é um par (uma unidade tratada e uma
controle bem parecidas), temos o desenho de **pares casados**.

O estimador combina os efeitos por bloco numa média ponderada (tópico 9). Vale
um cuidado: o modelo aditivo clássico `y = η + β_i + τ_t + ε` supõe que o efeito
do tratamento é o mesmo em todos os blocos (ausência de interação
tratamento×bloco). O estimador ponderado da biblioteca **não** depende dessa
suposição: ele estima a média ponderada dos efeitos por bloco, qualquer que seja
a heterogeneidade entre blocos.

### Exemplo trabalhado

Suponha duas regiões com linhas de base muito diferentes. Região A tem resultado
típico em torno de 0, região B em torno de 3, e o efeito verdadeiro é `+0,5` em
ambas. Num CRD ingênuo, a enorme diferença A vs. B entra no erro e pode
"afogar" o efeito de `0,5`. Bloqueando por região, estimamos `+0,5` dentro de A
e `+0,5` dentro de B, e combinamos: o ruído A vs. B nunca entra na conta, e a
estimativa final fica muito mais precisa. Esse é exatamente o cenário do
notebook `05_blocking`.

> **Na biblioteca.** `BlockedCRD(block_col=..., p=...)` randomiza dentro de cada
> bloco preservando a proporção; `BlockedDifferenceInMeans` estima a SATE
> ponderada por tamanho de bloco. Dois alinhamentos:
> - O estimador **não exige** "ausência de interação" treatment×bloco.
> - **Pares casados** (um tratado e um controle por bloco) são um caso-limite
>   válido para estimar, mas o `BootstrapCI` precisa de pelo menos 2 unidades
>   por estrato (bloco×braço), então pares casados levantam
>   `InsufficientDataError` no bootstrap.

---

## 6. Rerandomização e distância de Mahalanobis

### Intuição

A randomização equilibra as covariáveis em média, mas um sorteio específico pode
sair desequilibrado por azar, especialmente com poucas unidades ou muitas
covariáveis. A rerandomização é um seguro contra esse azar: defina antes um
critério de equilíbrio, sorteie, e se o sorteio falhar no critério, **descarte e
sorteie de novo**, até passar.

### Formalização: distância de Mahalanobis

Precisamos resumir o desequilíbrio de **várias** covariáveis num único número,
levando em conta a escala e a correlação entre elas. A distância de Mahalanobis
faz isso (Morgan & Rubin, 2012):

$$
M = (\bar{X}_T - \bar{X}_C)^{\top}\,
      \big[\operatorname{cov}(\bar{X}_T - \bar{X}_C)\big]^{-1}\,
      (\bar{X}_T - \bar{X}_C).
$$

Aqui `X̄_T - X̄_C` é o vetor das diferenças de média das covariáveis entre os
braços, e a matriz no meio é a covariância dessa diferença, que sob randomização
completa vale `(1/n_T + 1/n_C)·S_X`, com `S_X` a covariância amostral das
covariáveis. Aceita-se o sorteio se `M ≤ a`, onde `a` é o limiar.

Sob normalidade aproximada, `M` segue uma qui-quadrado com `k` graus de
liberdade (`k` = número de covariáveis), então um limiar natural é
`a = chi2.ppf(p_a, df=k)`, onde `p_a` é a fração de sorteios que você aceita
(por exemplo `p_a = 0,05` aceita os 5% mais equilibrados). Quanto menor `p_a`,
mais rígido o critério e maior a redução de variância nas covariáveis (e, por
tabela, no estimador, na medida em que as covariáveis predizem o resultado).

### Inferência precisa respeitar o critério

Um ponto sutil e importante: se você rerandomiza no desenho mas analisa com um
teste padrão, o teste fica **conservador**. A análise correta usa um teste de
randomização que **gera as permutações sob o mesmo critério de aceitação** usado
no desenho.

### Exemplo trabalhado

Com duas covariáveis (`x1`, `x2`) e `p_a = 0,05`, o limiar é
`a = chi2.ppf(0,05, df=2) ≈ 5,99`. Você sorteia; se as mudas grandes caíram quase
todas no tratamento, `M` excede `a` e o sorteio é descartado; repete-se até
`M ≤ 5,99`. O resultado é um sorteio em que `x1` e `x2` estão equilibrados entre
os braços, e o efeito final fica isolado dessas variáveis.

> **Na biblioteca.** `ReRandomizedCRD(covariates=[...], threshold=..., p=...)`
> implementa o laço de aceitação/rejeição, cacheia a matriz de covariância e a
> **reusa** no `draw()`. O `RandomizationTest` respeita o critério ao reutilizar
> a mesma matriz/limiar nas permutações, mantendo o teste válido (não
> conservador). Use quando há muitas covariáveis; evite em alocação sequencial
> (unidades chegando uma a uma) ou sem acesso às covariáveis antes da
> intervenção.

---

## 7. Fatorial 2^K

### Intuição

Em vez de testar um fator por vez, o desenho fatorial varia `K` fatores binários
ao mesmo tempo e estima, num único experimento, os **efeitos principais** (o
impacto médio de cada fator) e as **interações** (quando o efeito de um fator
depende do nível de outro). É eficiente: com uma rodada de `2^K` células,
extraem-se todos os efeitos.

### Formalização: contrastes ortogonais

Há `2^K` células (combinações dos níveis). Com codificação `-1` (baixo) e `+1`
(alto), cada efeito é um **contraste** ortogonal nas médias das células. O
modelo de regressão equivalente, para dois fatores, é

$$
y = \beta_0 + \beta_1 x_A + \beta_2 x_B + \beta_{12}\,x_A x_B + \varepsilon,
\qquad x_A, x_B \in \{-1, +1\}.
$$

A ortogonalidade da matriz de contrastes faz com que cada efeito seja estimado
de forma independente dos outros. O número de efeitos é `2^K - 1`: são `K`
efeitos principais, `C(K,2)` interações de duas vias, e assim por diante até a
interação de `K` vias. A álgebra dos contrastes está detalhada em
[III. Estimação](03-estimacao.md), tópico 12.

### Exemplo trabalhado: um 2^2

Imagine dois fatores `A` e `B` (cada um baixo/alto) e as médias de resposta por
célula:

| `A` | `B` | média |
|---|---|---|
| - | - | 20 |
| + | - | 30 |
| - | + | 24 |
| + | + | 38 |

- Efeito principal de A: média no alto menos média no baixo de A,
  `(30+38)/2 - (20+24)/2 = 34 - 22 = 12`.
- Efeito principal de B: `(24+38)/2 - (20+30)/2 = 31 - 25 = 6`.
- Interação AB: `[(38+20) - (30+24)]/2 = (58 - 54)/2 = 2`.

A interação positiva de `2` significa que A e B juntos rendem um pouco mais do
que a soma dos efeitos isolados sugeriria. Sem o desenho fatorial, testando um
fator por vez, você jamais veria esse termo.

> **Na biblioteca.** `FactorialDesign(factors=[...], n_per_cell=...)` exige
> `n_per_cell · 2^K` unidades e usa codificação **little-endian** das células
> (`cell = Σ x_j · 2^j`). O `FactorialEstimator` devolve os `2^K - 1` efeitos.
> Para `K` grande, o número de execuções explode (`2^{10} = 1024`); fatoriais
> **fracionários**, que testam uma fração das células trocando interações de
> alta ordem por eficiência, são um item de v2.

### Quando usar e quando não usar

Use na fase de triagem, para descobrir quais fatores importam e como interagem,
ou em testes A/B multivariados. Evite quando a resposta é fortemente não linear
(dois níveis não capturam curvatura, situação em que se adicionam pontos
centrais ou se parte para superfície de resposta, ambos em v2).

---

Notebooks relacionados:
[`04_balance_rerandomization`](../../../examples/for_starters/pt-br/04_balance_rerandomization.ipynb),
[`05_blocking`](../../../examples/for_starters/pt-br/05_blocking.ipynb),
[`06_factorial`](../../../examples/for_starters/pt-br/06_factorial.ipynb).

