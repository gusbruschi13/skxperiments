# III. Estimação

Como transformar os dados observados numa estimativa do efeito. Todos os
estimadores desta seção produzem **apenas o ponto**; a medida de incerteza
(erro padrão, intervalo, p-valor) vem da inferência, tratada em
[IV. Inferência](04-inferencia.md). A separação é deliberada: na biblioteca, o
estimador resolve "qual o efeito" e a classe de inferência resolve "quão certo".

---

## 8. Diferença de médias e não-viés

### Intuição

Sob randomização, os grupos são comparáveis em média, então a diferença entre a
média do tratamento e a média do controle, após a intervenção, estima o efeito.

### Formalização e prova do não-viés

O estimador é

$$\hat{\tau} = \bar{Y}_T - \bar{Y}_C.$$

Por que ele é não-viesado para o SATE sob randomização completa? A chave é que,
nesse desenho, **toda** unidade tem a mesma probabilidade de ser tratada,
`P(T_i = 1) = n_T / N`. A média do braço tratado é uma média dos `Y_i(1)`
restrita às unidades sorteadas para tratamento. Tomando o valor esperado sobre o
sorteio:

$$
\mathbb{E}[\bar{Y}_T]
= \mathbb{E}\!\left[\frac{1}{n_T}\sum_{i:\,T_i=1} Y_i(1)\right]
= \frac{1}{n_T}\sum_{i=1}^{N} \mathbb{P}(T_i=1)\,Y_i(1)
= \frac{1}{n_T}\cdot\frac{n_T}{N}\sum_{i=1}^{N} Y_i(1)
= \overline{Y(1)}.
$$

Pelo mesmo argumento, `E[Ȳ_C] = mean(Y(0))`. Logo

$$\mathbb{E}[\hat{\tau}] = \overline{Y(1)} - \overline{Y(0)} = \text{SATE}.$$

Nenhuma suposição de modelo foi usada: só a estrutura do sorteio. Esse é o
sentido de "o desenho dita a inferência".

### Exemplo trabalhado

Na tabela da ciência da [seção I](01-fundamentos.md) (SATE `= 2,33`), suponha um
sorteio que coloca `{1, 4, 5}` no tratamento e `{2, 3, 6}` no controle.

- Tratados observam `Y(1)`: `12, 9, 16`, média `12,33`.
- Controles observam `Y(0)`: `8, 12, 7`, média `9,00`.
- `τ̂ = 12,33 - 9,00 = 3,33`.

Um outro sorteio daria outro número (a estimativa tem variância). Mas a média de
`τ̂` sobre os 20 sorteios possíveis (`C(6,3) = 20`) é exatamente `2,33`, como a
prova acima garante. A variância dessa distribuição é o que a inferência mede.

> **Na biblioteca.** `DifferenceInMeans(outcome_col=...)` para `CRDAssignment`
> (incluindo rerandomizado). É o estimador mais transparente e o ponto de
> partida. Evite em estudos observacionais (viés de seleção) e tenha cautela em
> amostras pequenas (não-viesado, mas com variância alta).

---

## 9. Estimador estratificado (ponderação por bloco)

### Intuição

Estima-se o efeito **dentro** de cada bloco e combina-se por uma média ponderada
pelo tamanho do bloco. Como a comparação é interna ao bloco, a variação entre
blocos não entra no erro.

### Formalização

Com blocos `b = 1, ..., B`, tamanho `N_b` e efeito interno `τ̂_b`,

$$
\hat{\tau}_{\text{strat}} = \sum_{b} \frac{N_b}{N}\,\hat{\tau}_b,
\qquad N = \sum_b N_b.
$$

Cada `τ̂_b = Ȳ_{T,b} - Ȳ_{C,b}` é não-viesado para o SATE do bloco `b` (pelo
argumento do tópico 8 aplicado dentro do bloco). A média ponderada é, então,
não-viesada para o SATE global. A variância, por independência entre blocos, é

$$
\operatorname{Var}(\hat{\tau}_{\text{strat}})
= \sum_b \left(\frac{N_b}{N}\right)^2 \operatorname{Var}(\hat{\tau}_b),
$$

fórmula que reaparece na [variância de Neyman estratificada](04-inferencia.md).

### Exemplo trabalhado

Dois blocos. Bloco A: `N_A = 40`, efeito interno `τ̂_A = 0,4`. Bloco B:
`N_B = 60`, efeito interno `τ̂_B = 0,7`. O estimador ponderado é

$$
\hat{\tau}_{\text{strat}} = \frac{40}{100}(0{,}4) + \frac{60}{100}(0{,}7)
= 0{,}16 + 0{,}42 = 0{,}58.
$$

Repare que ele difere de uma média simples `(0,4 + 0,7)/2 = 0,55`: a ponderação
dá mais peso ao bloco maior.

> **Na biblioteca.** `BlockedDifferenceInMeans` implementa exatamente essa média
> ponderada por tamanho. É equivalente à regressão com interações quando a
> variável de bloco é categórica. Exige cada bloco com ao menos um tratado e um
> controle (suporte comum).

---

## 10. Ajuste por regressão (Lin)

### Intuição

Covariáveis correlacionadas com o resultado podem reduzir a variância da
estimativa. Historicamente, temia-se (Freedman) que usar regressão para ajustar
experimentos pudesse **introduzir viés** se o modelo estivesse errado. Lin
(2013) mostra que, ao incluir as **interações tratamento×covariável** com as
covariáveis **centradas na média**, o ajuste nunca piora a precisão
assintótica, mesmo quando o modelo linear é só uma aproximação. A intuição
geométrica: permitir uma inclinação própria para cada braço impede que uma
relação covariável-resultado diferente entre os grupos contamine a estimativa do
efeito.

### Formalização

O estimador de Lin é o coeficiente `β` na regressão OLS

$$
Y_i = \alpha + \beta\,T_i + \gamma\,(z_i - \bar{z}) + \delta\,T_i\,(z_i - \bar{z}) + \varepsilon_i,
$$

onde `z_i - z̄` são as covariáveis centradas e `T_i` o indicador de tratamento.
A centralização é o que faz `β` ser a estimativa do ATE (e não um efeito num
ponto arbitrário das covariáveis). O termo de interação `T_i (z_i - z̄)` deixa
cada braço ter sua própria inclinação.

### Inferência correta

As propriedades de "não causar dano" são assintóticas e dependem do uso de uma
variância robusta à heterocedasticidade (estimador sanduíche de Huber-White,
tipicamente HC2). A diferença de médias simples e a regressão de Lin concordam
no valor esperado; Lin tende a ter erro padrão menor quando as covariáveis
predizem o resultado.

> **Na biblioteca (alinhamento).** `LinEstimator(outcome_col, covariates)`
> produz o **ponto**. O intervalo "correto" do Lin usa a variância robusta, que
> **não** sai do `NeymanCI` (a whitelist dele aceita só
> `DifferenceInMeans`/`BlockedDifferenceInMeans`). Para obter um IC com `Lin`,
> use `BootstrapCI` (que aceita qualquer estimador escalar) ou
> `RandomizationTest`. Esse é um ponto que o usuário precisa saber, para não
> procurar a combinação Neyman+Lin.

### Quando usar

Em alocação desigual (por exemplo 75% tratamento, 25% controle), onde a
regressão sem interação falha mais facilmente, e sempre que se queira mais
precisão de forma segura. Evite em amostras muito pequenas (há um viés que some
com `n`) ou quando a prioridade é a transparência total da diferença de médias.

---

## 11. CUPED

### Intuição

CUPED (*Controlled-experiment Using Pre-Experiment Data*) usa uma medida do
mesmo usuário **anterior** ao experimento para remover ruído. Se sabemos o
comportamento habitual do usuário, subtraímos a parte previsível e ficamos com a
parte que o tratamento de fato moveu.

### Formalização: variáveis de controle

CUPED é uma aplicação do método de **variáveis de controle**. Dada a métrica `Y`
e uma covariável pré-experimento `X`, defina o resíduo ajustado `Y - θX`. Sua
variância é

$$
\operatorname{Var}(Y - \theta X)
= \operatorname{Var}(Y) - 2\theta\operatorname{Cov}(Y,X) + \theta^2\operatorname{Var}(X).
$$

Minimizando em `θ` (derivada igual a zero):

$$
-2\operatorname{Cov}(Y,X) + 2\theta\operatorname{Var}(X) = 0
\quad\Longrightarrow\quad
\theta^{*} = \frac{\operatorname{Cov}(Y,X)}{\operatorname{Var}(X)}.
$$

Substituindo de volta, a variância mínima é

$$\operatorname{Var}(Y - \theta^{*} X) = \operatorname{Var}(Y)\,(1 - \rho^2),$$

onde `ρ = corr(Y, X)`. Ou seja, a **variância fica multiplicada por `1 - ρ²`**:
uma correlação de `0,7` entre passado e presente deixa a variância em
`1 - 0,49 = 0,51` da original, uma queda de cerca de 49% (na prática, "metade").
O estimador do efeito ajustado é

$$\Delta_{\text{CUPED}} = (\bar{Y}_T - \bar{Y}_C) - \theta\,(\bar{X}_T - \bar{X}_C).$$

Como `X` é pré-experimento, sob randomização `E(X_T) = E(X_C)`, então o termo
subtraído tem esperança zero e o ajuste **não introduz viés**.

### Exemplo trabalhado: Bing slowdown

No exemplo clássico (Deng et al., 2013), a equipe do Bing mediu o impacto de um
atraso proposital de 250 ms no engajamento. Com um teste t comum, foram precisas
duas semanas para alcançar significância. Usando como covariável a atividade dos
mesmos usuários nas duas semanas anteriores (com `ρ ≈ 0,7`, logo
`1 - ρ² ≈ 0,5`), a variância caiu cerca de 50% e o efeito ficou significativo já
no primeiro dia, com metade dos usuários. A matemática acima explica de onde vem
esse "50%": é o `1 - ρ²`.

> **Na biblioteca.** `CUPED(outcome_col, pre_experiment_col)` computa
> `θ = Cov(Y,X)/Var(X)` e expõe `theta` e `correlation` em `Results.extra`. v1
> aceita só `CRDAssignment`. A covariável **tem que ser anterior ao tratamento**
> (ou não afetada por ele); usar uma covariável contaminada pelo tratamento
> introduz viés.

---

## 12. Contrastes fatoriais (codificação ±1) e a escala dos efeitos

### Intuição

Com níveis codificados `-1` (baixo) e `+1` (alto), cada efeito é uma média de
respostas com sinais `±1`, que isola a mudança atribuível àquele fator (ou
combinação de fatores).

### Formalização: a álgebra dos contrastes

Para um subconjunto não vazio `S` de fatores, o efeito é

$$
\text{efeito}_S = \frac{1}{2^{K-1}}
\sum_{\text{células}} \bar{y}_{\text{célula}} \prod_{j \in S}(2x_j - 1),
$$

onde `x_j` é o nível do fator `j` na célula (em `{0,1}`) e `2x_j - 1` o converte
para `±1`. Casos particulares:

- Efeito principal de A: `ȳ(A+) - ȳ(A-)` (a diferença das médias entre o nível
  alto e o baixo de A).
- Interação AB: `[ȳ(++) + ȳ(--) - ȳ(+-) - ȳ(-+)] / 2`.

O divisor `2^{K-1}` é o número de células em cada "lado" do contraste.

### A sutileza da escala: ±1 vs. {0,1}

Esse ponto confunde quem simula dados. Suponha que você gere o resultado por um
modelo com codificação `{0,1}`:

$$y = b_A\,A + b_B\,B + b_{AB}\,(A\,B) + \text{ruído}, \qquad A,B \in \{0,1\}.$$

O contraste `±1` da biblioteca **não** devolve `b_A` para o efeito principal de
A. Fazendo a conta com as quatro médias de célula, obtém-se

$$
\text{efeito}_A = b_A + \tfrac{1}{2} b_{AB},
\qquad \text{efeito}_{AB} = \tfrac{1}{2} b_{AB}.
$$

Ou seja, o efeito principal estimado é a média do efeito de A sobre os dois
níveis de B, e a interação vem com um fator `1/2`. Isso não é bug: é a definição
de efeito fatorial. Ao escrever exemplos didáticos com simulação, deixe isso
explícito para o leitor não estranhar a diferença entre os `b` da simulação e os
efeitos estimados.

### Efeito fatorial não é Cohen's d

São duas escalas distintas que vale não confundir:

- O **efeito fatorial** (o que o estimador retorna) está em **unidade bruta** da
  resposta (por exemplo, "+12 pontos de rendimento").
- O **Cohen's d** é um tamanho de efeito **padronizado**,
  `d = (m_A - m_B)/σ`, usado em análise de **poder** (ver
  [IV. Inferência](04-inferencia.md), tópico 17), para dizer se um efeito é
  "pequeno, médio ou grande" independentemente da unidade.

> **Na biblioteca (correção).** O `FactorialEstimator` devolve os contrastes em
> **unidade bruta**, com o divisor `1/2^{K-1}` nos termos de ordem maior. Ele
> **não** padroniza por `σ` e, portanto, **não** retorna Cohen's d. A convenção
> da lib para efeitos principais é a **diferença completa** alto-menos-baixo
> (não a metade, que algumas referências chamam de coeficiente de regressão).

---

Notebooks relacionados:
[`03_reducing_variance`](../../../examples/for_starters/pt-br/03_reducing_variance.ipynb),
[`06_factorial`](../../../examples/for_starters/pt-br/06_factorial.ipynb).
