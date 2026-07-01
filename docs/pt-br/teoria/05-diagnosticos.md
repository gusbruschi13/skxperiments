# V. Diagnósticos e prática

Antes de confiar num resultado, é preciso verificar se o experimento foi bem
conduzido. Esta seção cobre os três diagnósticos centrais (equilíbrio, SRM e
A/A) e a composição de tudo num fluxo reproduzível.

---

## 18. Equilíbrio de covariáveis (SMD)

### Intuição

Os grupos de tratamento e controle devem ter distribuições de características
iniciais (idade, região, comportamento prévio) parecidas. Se diferem **antes**
da intervenção, o resultado fica confundido. A SMD (diferença de médias
padronizada) mede esse desequilíbrio numa escala que independe da unidade da
variável, permitindo comparar "idade em anos" e "renda em reais" na mesma régua.

### Formalização

Para uma covariável contínua,

$$\text{SMD} = \frac{\bar{x}_T - \bar{x}_C}{\sqrt{(s_T^2 + s_C^2)/2}},$$

onde `s_T²` e `s_C²` são as variâncias da covariável em cada grupo (com
`ddof = 1`). O denominador é o desvio padrão agrupado (*pooled*), que padroniza a
diferença sem ser inflado pelo tamanho da amostra. A regra prática mais comum
(Stuart, 2010) é considerar `|SMD| > 0,1` como desequilíbrio relevante.

### Por que não usar teste t para avaliar equilíbrio

Este é um ponto sutil e importante. Um teste t (ou um p-valor) para "as médias
são iguais?" depende do **tamanho da amostra**: com `n` enorme, uma diferença
minúscula e irrelevante vira "significativa"; com `n` pequeno, um desequilíbrio
real pode não atingir significância. Pior: após um pareamento, o `n` cai e o
p-valor sobe, dando a falsa impressão de equilíbrio. A SMD, por ser padronizada
e independente de `n`, não tem esse problema. Por isso a recomendação é avaliar
equilíbrio pela **magnitude da SMD**, não por p-valor. Quando o resultado é
sensível à dispersão, vale também checar a **razão de variâncias** entre os
grupos, pois médias iguais com variâncias muito diferentes ainda podem enviesar.

### Exemplo trabalhado e o Love plot

Estudo de um treinamento de vendas. No início, o grupo que aderiu tem 10 anos de
experiência média e o controle 2 anos. Com desvios padrão de, digamos, 4 e 3, o
denominador agrupado é `sqrt((16 + 9)/2) = sqrt(12,5) ≈ 3,54`, e a SMD inicial é

$$\frac{10 - 2}{3{,}54} \approx 2{,}26,$$

altíssima (muito acima de `0,1`): os grupos não são comparáveis. Após equilibrar
(por blocagem, rerandomização ou, em estudos observacionais, pareamento),
as médias se aproximam e a SMD cai para perto de zero. O **Love plot** desenha,
para cada covariável, a SMD antes (ponto aberto, longe de zero) e depois (ponto
fechado, perto de zero), tornando o equilíbrio visível de relance.

> **Na biblioteca.** `check_balance(assignment, covariates)` calcula a SMD com o
> desvio padrão agrupado `sqrt((var_T + var_C)/2)` e `ddof = 1`. O
> `BalanceReport` sinaliza `|SMD| > 0,1` e expõe a tabela, que o `plot_balance`
> desenha como Love plot. A biblioteca, por desenho, **não** usa teste t para
> equilíbrio. Em experimentos randomizados, este é um diagnóstico do **sorteio
> concretamente realizado** (a randomização garante equilíbrio em média, não em
> cada sorteio).

---

## 19. SRM (Sample Ratio Mismatch)

### Intuição

Se a proporção observada de unidades em cada braço não bate com a planejada,
algo quebrou: sorteio defeituoso, perda de dados, filtragem de bots
assimétrica. O SRM é um alarme de **bug de implementação**, não uma hipótese
científica. Por isso ele invalida tudo: se a alocação está errada, nenhuma
estimativa em cima dela é confiável.

### Formalização

Testa-se a hipótese de que a razão observada é igual à planejada com um
qui-quadrado de aderência. Para dois braços com `N` unidades e proporção
esperada `p`, as contagens esperadas são `E_T = N p` e `E_C = N(1-p)`, e a
estatística é

$$\chi^2 = \sum_{g \in \{T, C\}} \frac{(O_g - E_g)^2}{E_g},$$

com 1 grau de liberdade. Sinaliza-se SRM se o p-valor for **muito** baixo,
tipicamente `p < 0,001`.

### Por que o limiar é tão estrito

Em amostras grandes (milhões de unidades), a Lei dos Grandes Números faz a razão
ficar quase perfeita, então qualquer desvio sistemático produz um p-valor
minúsculo. O limiar estrito de `0,001` reflete que o SRM costuma vir de **perda
seletiva** de usuários específicos (por exemplo, os mais ativos sendo
classificados como bots e descartados), o que é suficiente para inverter o
sentido do efeito real. Um limiar frouxo (0,05) geraria alarmes falsos demais e,
ao mesmo tempo, um SRM verdadeiro quase sempre estoura o `0,001` com folga.

### Exemplo trabalhado: Bing

Plano de divisão 50/50. Observado: `821.588` no controle e `815.482` no
tratamento, total `N = 1.637.070`. As contagens esperadas são `N/2 = 818.535`
cada. A estatística é

$$
\chi^2 = \frac{(821588 - 818535)^2}{818535} + \frac{(815482 - 818535)^2}{818535}
= 2 \cdot \frac{3053^2}{818535} \approx 22{,}8,
$$

cujo p-valor (qui-quadrado, 1 g.l.) é cerca de `1,8 \times 10^{-6}`. A diferença
de contagem parece pequena (razão `0,993`), mas o p-valor minúsculo prova que
não foi acaso. A investigação revelou que o tratamento causava um bug que fazia
o sistema classificar usuários reais como bots, excluindo-os. O SRM evitou uma
decisão baseada em dados corrompidos.

> **Na biblioteca.** `SRMTest(threshold=0.001)` usa `scipy.stats.chisquare`
> comparando contagens observadas e esperadas (proporção `p` do design para dois
> braços, ou células uniformes no fatorial). Rode **sempre, antes** de analisar;
> se falhar, depure a causa antes de olhar qualquer outro resultado. O
> `ExperimentPipeline` roda o `SRMTest` automaticamente.

---

## 20. Teste A/A: calibração e uniformidade dos p-valores

### Intuição

Num teste A/A, os dois grupos recebem **a mesma** experiência. É uma balança
calibrada no zero: como não há diferença real, o sistema não deveria acusar
"significância" mais do que o acaso permite. Se acusa, há erro no código, na
estatística ou no sorteio.

### Formalização: por que os p-valores ficam uniformes

Sob a hipótese nula verdadeira (garantida num A/A), o p-valor de um teste bem
calibrado segue uma distribuição `Uniforme[0, 1]`. Isso é consequência da
transformação integral de probabilidade: se a estatística de teste tem a
distribuição assumida sob a nula, então `p = F(T)` é uniforme. A consequência
operacional é

$$\mathbb{P}(p \le \alpha \mid H_0) = \alpha.$$

Em 1000 testes A/A, espera-se cerca de 50 com `p < 0,05` puramente por sorte. Um
histograma dos p-valores deve ficar **plano**. Um pico perto de zero indica que
o sistema está descalibrado e produzindo falsos-positivos em excesso.

### Exemplo trabalhado

A equipe suspeita que o erro padrão da métrica "cliques por usuário" está
subestimado por causa de usuários-robô (outliers) e do fato de a unidade de
sorteio (usuário) diferir da de medida (clique). Eles simulam 1000 testes A/A
sorteando usuários do passado para dois grupos idênticos (*offline replay*, para
não gastar tráfego real) e plotam o histograma dos 1000 p-valores. Se 15% derem
`p < 0,05` (em vez dos 5% esperados), o sistema está descalibrado. Após corrigir
o cálculo (por exemplo, com o método Delta para a métrica de razão), repetem a
simulação até o histograma ficar plano, provando que a taxa de erro do Tipo I
voltou a `α`.

> **Na biblioteca.** `AATest(design, inference, n_simulations, ...)` rerandomiza
> sobre dados sem efeito e verifica a **taxa de falso-positivo** (teste binomial
> contra `α`) e a **uniformidade** dos p-valores (Kolmogorov-Smirnov). O caso de
> unidade de sorteio diferente da de medida, que viola a independência e quebra
> a uniformidade, motiva o método Delta (item de v2).

---

## 21. Composição: pipeline e relatório

### Intuição

Depois de entender cada peça, compõe-se um fluxo que vai do desenho ao relatório,
de forma reproduzível e com os diagnósticos rodando antes da leitura dos
resultados.

### O que a prática da indústria inclui (contexto)

Em plataformas de experimentação maduras (Kohavi et al.), o fluxo completo passa
por *data cooking* (juntar logs, filtrar bots, limpar duplicatas, computar
métricas), métricas de proteção (guardrails) e um OEC (*Overall Evaluation
Criterion*) que combina várias métricas num índice único,

$$\text{OEC} = \sum_i w_i \cdot \text{Metric}_i,$$

com `w_i` os pesos de importância. Um princípio recorrente (de Shewhart) é
"preservar a evidência": o relatório mostra o SRM e os guardrails **antes** dos
ganhos, para evitar leitura enviesada.

### O que a biblioteca faz (escopo)

A `skxperiments` cobre a parte estatística **a partir do `Assignment`**, não a
ingestão e a limpeza de dados nem o cálculo de OEC.

- `ExperimentPipeline(inference, diagnostics=[SRMTest(), ...])` roda os
  diagnósticos e a inferência sobre um `Assignment` e devolve um
  `PipelineResult` que reúne resultado, relatório de diagnósticos e flags. O
  `SRMTest` roda por padrão, e uma flag não interrompe a estimativa (a menos que
  você peça `raise_on_flag=True`).
- `ExperimentComparison` compara experimentos independentes aplicando a correção
  de múltiplos testes na família.
- `ExperimentReport` gera um HTML autocontido com a tabela de resultados, os
  diagnósticos e os gráficos embutidos.

### Exemplo trabalhado

Um teste de cor de botão no LinkedIn. O `ExperimentPipeline` recebe o
`Assignment` (já com o outcome), roda o `SRMTest` (verde), roda o `NeymanCI`
para o efeito, e empacota tudo. O `ExperimentReport` mostra, no topo, o
diagnóstico de SRM; se passou, exibe o efeito e o intervalo. Se o efeito for
`+1%` com IC que exclui zero, o relatório evidencia um resultado robusto, com a
ordem "diagnóstico primeiro, ganho depois" que a boa prática recomenda.

> **Na biblioteca (escopo).** O `ExperimentPipeline`/`ExperimentReport` cuidam da
> composição estatística e do relatório; o *data cooking* e o OEC são
> **metodologia** de plataforma, fora do que a lib implementa. O princípio de
> surfar os diagnósticos antes dos ganhos está refletido no design do
> `PipelineResult`.

---

Notebooks relacionados:
[`08_diagnostics`](../../../examples/for_starters/pt-br/08_diagnostics.ipynb),
[`09_putting_it_together`](../../../examples/for_starters/pt-br/09_putting_it_together.ipynb).
