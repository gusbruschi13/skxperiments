# I. Fundamentos

Esta seção estabelece o vocabulário e a lógica da inferência causal sob o
framework de *potential outcomes* (Rubin Causal Model), que é a base de toda a
`skxperiments`. A ideia central, que se repete em cada módulo da biblioteca, é
simples de enunciar e poderosa em consequências: o **mecanismo de atribuição do
tratamento** é o ponto de partida da análise, e não o modelo estatístico.

> Estes textos são teóricos. Para o uso prático lado a lado com a API, veja os
> notebooks em [`examples/for_starters/pt-br/`](../../../examples/for_starters/pt-br/).

---

## 1. O problema fundamental e os potential outcomes

### Intuição

Falar de "efeito causal" é falar de uma comparação entre dois mundos: o mundo em
que a unidade recebeu o tratamento e o mundo em que ela não recebeu. Para um
paciente, o efeito de um remédio é a diferença entre o tempo de recuperação
**com** o remédio e o tempo de recuperação **sem** o remédio, para o mesmo
paciente, nas mesmas condições. O obstáculo é que, na realidade, cada paciente
ou toma o remédio ou não toma. O outro mundo, o contrafactual, nunca é
observado.

### Formalização

Para cada unidade `i` definimos dois resultados potenciais:

- `Y_i(1)`: o resultado que seria observado se `i` recebesse o tratamento.
- `Y_i(0)`: o resultado que seria observado se `i` não recebesse o tratamento.

O efeito causal individual é a diferença entre eles:

$$\delta_i = Y_i(1) - Y_i(0).$$

A atribuição é representada por `T_i` (1 para tratado, 0 para controle). O
resultado observado é apenas um dos dois potenciais, selecionado por `T_i`:

$$Y_i = T_i\,Y_i(1) + (1 - T_i)\,Y_i(0).$$

O **problema fundamental da inferência causal** (Holland, 1986) é que, para cada
unidade, observamos `Y_i(1)` ou `Y_i(0)`, nunca os dois. Logo `delta_i` nunca é
diretamente mensurável. Sob essa ótica, a inferência causal é um problema de
**dados ausentes**: metade da "tabela da ciência" (a tabela com `Y(0)` e `Y(1)`
de todas as unidades) está sempre faltando.

A saída é abandonar o efeito individual e estimar **médias** (ver tópico 2),
porque médias de quantidades não observadas podem, sob randomização, ser
estimadas sem viés a partir do que observamos.

### Pressupostos

Para que essa estrutura seja bem definida, assumem-se duas condições, reunidas
sob a sigla SUTVA (*Stable Unit Treatment Value Assumption*):

1. **Ausência de interferência**: o resultado de uma unidade não depende do
   tratamento atribuído a outras unidades. O `Y_i(t)` depende só do tratamento
   de `i`.
2. **Ausência de versões ocultas do tratamento** (consistência): existe um único
   "tratamento" bem definido, de modo que o resultado observado é exatamente o
   resultado potencial do braço recebido.

SUTVA falha, por exemplo, em redes sociais e marketplaces, onde o tratamento de
um usuário "vaza" para outro. Quando isso ocorre, os estimadores padrão ficam
viesados (esse é um tema de v2 na biblioteca).

### Exemplo trabalhado: a tabela da ciência

Considere seis unidades com a seguinte tabela de resultados potenciais (que na
prática nunca observamos por inteiro):

| unidade | `Y(0)` | `Y(1)` | `delta` |
|---|---|---|---|
| 1 | 10 | 12 | 2 |
| 2 | 8 | 11 | 3 |
| 3 | 12 | 13 | 1 |
| 4 | 9 | 9 | 0 |
| 5 | 11 | 16 | 5 |
| 6 | 7 | 10 | 3 |

O efeito médio verdadeiro nestas seis unidades é

$$\text{SATE} = \frac{2+3+1+0+5+3}{6} = \frac{14}{6} \approx 2{,}33.$$

Repare que ele também é a diferença das médias dos dois potenciais:
`mean(Y(1)) = 71/6 ≈ 11,83` e `mean(Y(0)) = 57/6 = 9,5`, cuja diferença é
`2,33`. Essa identidade vai reaparecer no tópico 8: é por isso que a diferença
de médias estima o efeito médio.

O ponto a internalizar: nunca veremos as duas colunas. Se a unidade 5 for
tratada, observamos `16` e perdemos o `11`; o efeito `5` dela é para sempre uma
inferência, não uma medida.

> **Na biblioteca.** `skxperiments.core.potential_outcomes.PotentialOutcomes`
> representa `Y(0)`, `Y(1)`, o efeito individual e o ATE, para fins didáticos
> (em um experimento real você nunca teria as duas colunas). A "ignorabilidade
> forte", que em estudos observacionais precisa ser **suposta** (dado um
> conjunto de covariáveis, a atribuição é independente dos potenciais), na lib é
> **garantida pelo desenho**: a randomização produz essa independência por
> construção. Por isso a biblioteca parte do `Assignment`, e não de um modelo de
> seleção. A formulação de Pearl via grafos e do-calculus é logicamente
> equivalente; a lib adota a linguagem de potential outcomes por ser a mais
> direta para experimentos.

---

## 2. Estimandos: ATE, SATE vs. PATE

### Intuição

Como o efeito individual é inacessível, definimos o alvo da inferência (o
"estimando") como um **efeito médio**. Mas média sobre qual conjunto? Sobre as
unidades efetivamente no experimento, ou sobre uma população maior da qual elas
seriam uma amostra? Essa escolha não é cosmética: muda a fonte de aleatoriedade,
a fórmula da variância e a interpretação do intervalo de confiança.

### Formalização

O *Average Treatment Effect* é a média dos efeitos individuais:

$$
\text{ATE} = \frac{1}{n}\sum_{i=1}^{n} \big(Y_i(1) - Y_i(0)\big)
            = \overline{Y(1)} - \overline{Y(0)}.
$$

A partir daí, duas leituras:

- **SATE** (*Sample Average Treatment Effect*): a média dos efeitos para as `n`
  unidades **específicas** do estudo. Aqui os resultados potenciais são tratados
  como números fixos, e a **única** fonte de aleatoriedade é o sorteio do
  tratamento. Essa é a visão de **população finita** (Neyman, Fisher).
- **PATE** (*Population Average Treatment Effect*): o efeito médio numa
  **superpopulação**, da qual as `n` unidades são uma amostra aleatória. Há duas
  fontes de aleatoriedade: a amostragem das unidades e o sorteio do tratamento.

Formalmente, o PATE é o valor esperado do SATE quando reamostramos as unidades,
e no limite `n` grande os dois coincidem em valor. A diferença aparece na
**variância**: a incerteza sobre o PATE inclui a variabilidade de quais unidades
entraram no estudo, que não existe na visão de população finita.

### Quando cada um

- Use **SATE** (população finita) quando o interesse é exatamente nestas
  unidades. Exemplo clássico: o efeito de um fertilizante **neste** campo
  dividido em parcelas. Não há "superpopulação de campos" relevante.
- Use **PATE** (superpopulação) quando o objetivo é generalizar. É o padrão em
  testes A/B online: os usuários de hoje são vistos como amostra de um fluxo
  contínuo de usuários futuros.

Há ainda o ATT (*Average Treatment Effect on the Treated*), a média do efeito
restrita às unidades tratadas, útil quando o efeito é heterogêneo e a decisão é
sobre quem de fato recebe o tratamento.

### Exemplo trabalhado

Na tabela da ciência do tópico 1, o SATE é `2,33`: é o efeito médio **naquelas
seis unidades**. Se elas fossem uma amostra de uma população grande de
unidades similares, e quiséssemos prever o efeito médio em qualquer amostra
futura, estaríamos atrás do PATE. O ponto estimado seria o mesmo `2,33`, mas o
intervalo de confiança para o PATE seria mais largo, porque precisa absorver a
incerteza de "estas seis poderiam ter sido outras seis".

> **Na biblioteca.** A distinção mapeia direto na escolha de inferência:
> **SATE → `NeymanCI`** (população finita) e **PATE → `BootstrapCI`**
> (superpopulação). Escolher entre os dois é escolher **sobre quem** você quer
> concluir. O `ExperimentComparison` e o `MultipleTestingCorrection` operam no
> nível dos efeitos estimados, independentemente dessa escolha.

---

## 3. Por que randomizar

### Intuição

Se deixarmos as unidades escolherem (ou um processo não controlado decidir)
quem recebe o tratamento, os grupos tendem a diferir em características que
também afetam o resultado. A diferença observada entre os grupos passa a
misturar o efeito do tratamento com essas diferenças pré-existentes. Esse é o
**confundimento**. Randomizar quebra a ligação entre as características da
unidade e a atribuição, tornando os grupos comparáveis em média.

### Formalização: o viés de seleção

Decomponha a diferença de médias observada. Seja o grupo de tratamento `T=1` e
o de controle `T=0`. O que observamos é

$$
\mathbb{E}[Y \mid T=1] - \mathbb{E}[Y \mid T=0]
= \underbrace{\mathbb{E}[Y(1) \mid T=1] - \mathbb{E}[Y(0) \mid T=1]}_{\text{ATT (efeito real nos tratados)}}
+ \underbrace{\mathbb{E}[Y(0) \mid T=1] - \mathbb{E}[Y(0) \mid T=0]}_{\text{viés de seleção}}.
$$

O segundo termo é o **viés de seleção**: a diferença de linha de base (`Y(0)`)
entre quem foi tratado e quem não foi, **na ausência** de tratamento. Em estudos
observacionais ele costuma ser não nulo (quem busca o tratamento já era
diferente). A randomização força

$$T \perp \big(Y(0), Y(1)\big),$$

ou seja, a atribuição é independente dos resultados potenciais. Isso zera o viés
de seleção (`E[Y(0)|T=1] = E[Y(0)|T=0]`) e iguala ATT e ATE, de modo que a
diferença de médias passa a estimar o efeito causal.

### Exemplo trabalhado: confundimento vs. randomização

Volte à tabela da ciência. Suponha um processo de **auto-seleção** em que as
unidades com maior linha de base `Y(0)` tendem a buscar o tratamento. Digamos
que os tratados sejam `{1, 3, 5}` (que têm `Y(0)` de 10, 12 e 11) e os controles
`{2, 4, 6}` (com `Y(0)` de 8, 9 e 7).

- Tratados observam `Y(1)`: `12, 13, 16`, média `13,67`.
- Controles observam `Y(0)`: `8, 9, 7`, média `8,00`.
- Diferença observada: `13,67 - 8,00 = 5,67`.

Mas o efeito médio verdadeiro é `2,33`. O excesso de `3,3` é puro viés de
seleção: os tratados já partiam de uma linha de base mais alta (`E[Y(0)|T=1] =
11` contra `E[Y(0)|T=0] = 8`). A "evidência" de um efeito enorme é, em boa
parte, a diferença que já existia antes.

Agora randomize. Sob sorteio, qualquer unidade tem a mesma chance de cair em
cada braço, então `E[Y(0)|T=1] = E[Y(0)|T=0] = 9,5` (a média geral de `Y(0)`), o
viés desaparece, e a diferença de médias estima `2,33`. O exemplo do Netflix é o
mesmo mecanismo: sem randomizar, os *heavy users* (que cancelariam menos de
qualquer jeito) dominam o grupo da interface nova e inflam o efeito aparente.

### Limite: equilíbrio só em média

A randomização garante comparabilidade **em média**, sobre todos os sorteios
possíveis. Um sorteio específico, sobretudo em amostra pequena, pode sair
desequilibrado por azar (todos os *heavy users* num braço). Isso motiva a
**rerandomização** (tópico 6): restringir os sorteios aceitáveis a um critério
de equilíbrio.

> **Na biblioteca.** O design é o ponto de partida: `CRD`, `BlockedCRD`,
> `ReRandomizedCRD` e `FactorialDesign` produzem um `Assignment`, que é o
> contrato consumido pelos estimadores. Como a randomização garante a
> ignorabilidade, a biblioteca não precisa (e não tenta) modelar um processo de
> seleção. Para diagnosticar o equilíbrio de um sorteio concreto, use
> `check_balance` e `BalanceReport` (ver [V. Diagnósticos](05-diagnosticos.md)).

---

Notebooks relacionados:
[`00_why_randomize`](../../../examples/for_starters/pt-br/00_why_randomize.ipynb),
[`01_first_experiment`](../../../examples/for_starters/pt-br/01_first_experiment.ipynb).
