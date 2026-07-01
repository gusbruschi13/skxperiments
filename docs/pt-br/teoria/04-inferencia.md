# IV. Inferência

Estimar o ponto é metade do trabalho; a outra metade é quantificar a incerteza.
A `skxperiments` oferece três caminhos de inferência com interpretações
distintas (randomização, Neyman, bootstrap), mais a correção para múltiplos
testes e o planejamento de poder. Esta seção deriva o que cada um faz.

---

## 13. Inferência baseada em randomização (sharp null de Fisher)

### Intuição

Não assumimos que os dados seguem uma normal. Assumimos que o tratamento não teve
efeito algum e perguntamos: dado esse cenário, quão raro seria observar uma
diferença tão grande quanto a que observamos, só por causa do sorteio? Se for
muito raro, rejeitamos a hipótese de "efeito zero".

### Formalização: a hipótese nula forte e a distribuição de randomização

A **hipótese nula forte** (sharp null) de Fisher afirma que o efeito é zero para
**toda** unidade:

$$H_0:\quad Y_i(1) = Y_i(0) \quad \text{para todo } i.$$

Sob `H_0`, o resultado observado de cada unidade não dependeria do braço: o valor
`Y_i` é fixo. A única coisa aleatória é **qual** alocação saiu. Logo podemos, em
princípio, recalcular a estatística de teste (por exemplo, a diferença de
médias) para **cada** uma das `C(N, n_T)` alocações possíveis, mantendo os `Y_i`
fixos e só trocando os rótulos. Isso gera a **distribuição de randomização** da
estatística sob a nula, sem nenhuma suposição distribucional. O p-valor é a
fração dessa distribuição tão ou mais extrema que o observado.

Quando o número de alocações é grande demais para enumerar, amostramos `m`
alocações por Monte Carlo. O p-valor usa a correção de Phipson & Smyth (2010):

$$p = \frac{B + 1}{m + 1},$$

onde `B` é o número de permutações tão ou mais extremas que a observada. O `+1`
no numerador e no denominador inclui a **própria alocação observada** como uma
das possibilidades válidas sob a nula. Sem essa correção (`p = B/m`), o p-valor
poderia dar exatamente zero, o que é impossível num teste de permutação e causa
erros graves em análises de múltiplos testes.

### Exemplo trabalhado

Dez ratos, cinco recebem ração nova (tratamento) e cinco a antiga (controle).
Sob a sharp null, o peso de cada rato seria o mesmo em qualquer braço. Embaralhe
os dez pesos observados, sorteie cinco para um grupo "falso" de tratamento, e
calcule a diferença de médias. Repita `m = 1000` vezes. Se em apenas `B = 4`
desses 1000 sorteios a diferença "falsa" foi tão ou mais extrema que a real, o
p-valor é

$$p = \frac{4 + 1}{1000 + 1} \approx 0{,}005.$$

Como `0,5%` é raro, rejeitamos a nula.

> **Na biblioteca.** `RandomizationTest(estimator, n_permutations, ...)` usa
> exatamente `(1 + n_extreme) / (1 + n_permutations)`. O `draw()` rerandomiza
> pelo mesmo mecanismo do desenho (respeitando blocagem e rerandomização). O
> critério two-sided é `|T_perm| ≥ |T_obs|`.

---

## 14. Variância de Neyman e intervalo de Wald

### Intuição

A variância de Neyman mede a incerteza do efeito considerando só as unidades do
experimento (população finita). Ela é **conservadora** por uma razão precisa:
parte da variância verdadeira depende de algo que nunca observamos, e a fórmula
de Neyman lida com isso superestimando, para não prometer mais precisão do que
os dados sustentam.

### Formalização: a variância exata e por que dropamos um termo

Sob randomização completa, a variância exata do estimador da diferença de médias
para o SATE é (Neyman, 1923):

$$
\operatorname{Var}(\hat{\tau})
= \frac{S_1^2}{n_T} + \frac{S_0^2}{n_C} - \frac{S_{\tau}^2}{N},
$$

onde:

- `S_1²` é a variância dos `Y_i(1)` entre as `N` unidades,
- `S_0²` é a variância dos `Y_i(0)`,
- `S_τ²` é a variância dos **efeitos individuais** `δ_i = Y_i(1) - Y_i(0)`.

O problema: `S_τ²` envolve `Y_i(1)` e `Y_i(0)` **da mesma unidade**, que nunca
observamos juntos (o problema fundamental da [seção I](01-fundamentos.md)). Logo
`S_τ²` **não é identificável**. O estimador de Neyman simplesmente **descarta**
esse termo:

$$\hat{V}_{\text{Neyman}} = \frac{s_T^2}{n_T} + \frac{s_C^2}{n_C},$$

com `s²` as variâncias amostrais dentro de cada braço (`ddof = 1`). Como
`S_τ² ≥ 0`, descartar o termo negativo faz `V̂` ser, em média, **maior ou igual**
à variância verdadeira. Daí o nome "conservador". Há um caso em que é **exato**:
quando o efeito é **constante** (`δ_i = c` para todo `i`), tem-se `S_τ² = 0` e
nada é perdido.

Para desenhos **blocados**, a forma estratificada, consistente com a ponderação
por bloco do [tópico 9](03-estimacao.md), é

$$\hat{V} = \sum_b \left(\frac{N_b}{N}\right)^2 \hat{V}_b.$$

O intervalo de Wald é

$$\hat{\tau} \pm z_{1-\alpha/2}\,\sqrt{\hat{V}},$$

com `z` da **normal** (1,96 para 95%).

### Exemplo trabalhado: efeito constante vs. heterogêneo

Cenário A (efeito constante): a droga acelera a cura em exatamente 2 dias para
todos. Então `S_τ² = 0` e a variância de Neyman é exata; o IC tem a largura
"certa".

Cenário B (efeito heterogêneo): a droga acelera 10 dias em alguns e atrasa 2 dias
em outros. Agora `S_τ² > 0`, e como a fórmula de Neyman ignora o termo
`-S_τ²/N`, ela superestima a variância. O IC sai **mais largo** do que o
necessário: o método se recusa a declarar precisão alta quando há instabilidade
oculta nos efeitos individuais. Isso é uma característica, não um defeito.

> **Na biblioteca.** `NeymanCI(estimator)` aceita `DifferenceInMeans` e
> `BlockedDifferenceInMeans`, usa o quantil **normal** (não `t`) e a forma
> estratificada no caso blocado. É conservador para o SATE (população finita) e
> aproximadamente não-viesado para o PATE (superpopulação), porque na
> superpopulação o termo de covariância dos potenciais entra de outra forma.

---

## 15. Bootstrap: percentil vs. BCa

### Intuição

O bootstrap aproxima a distribuição amostral de quase qualquer estatística
reamostrando os próprios dados com reposição, sem fórmula fechada nem
normalidade. É a leitura de **superpopulação**: tratamos as unidades observadas
como amostra de uma população maior.

### Formalização

A partir dos dados, gera-se `B` reamostras (com reposição) e recalcula-se a
estatística em cada uma, obtendo a distribuição bootstrap `θ̂^*`.

- **Percentil**: o IC são os quantis `α/2` e `1 - α/2` de `θ̂^*` diretamente.
- **BCa** (*bias-corrected and accelerated*): ajusta os pontos de corte por dois
  parâmetros:
  - `z_0` (correção de viés): `Φ⁻¹` da **fração de réplicas bootstrap abaixo da
    estimativa observada**. Mede o viés de mediana.
  - `a` (aceleração): captura a assimetria, estimada por jackknife
    leave-one-out.

  Os percentis ajustados são

  $$\alpha_{\text{ajustado}} = \Phi\!\left(z_0 + \frac{z_0 + z^{(\alpha)}}{1 - a\,(z_0 + z^{(\alpha)})}\right),$$

  e o limite do IC é o quantil correspondente da distribuição bootstrap. Quando
  `z_0 = 0` e `a = 0` (distribuição simétrica, sem viés), o BCa coincide com o
  percentil.

### Exemplo trabalhado

Quer um IC para a correlação entre nota do vestibular e desempenho na faculdade,
com apenas 15 alunos. A correlação é limitada a `[-1, 1]` e sua distribuição
amostral é assimétrica nesse `n` pequeno, então a aproximação normal do teste t
é ruim. Você sorteia 1000 reamostras de 15 alunos (com reposição), calcula a
correlação em cada uma, e usa o BCa para corrigir o viés e a assimetria. Se a
correlação observada de `0,77` for puxada por um outlier, o IC por BCa reflete
essa instabilidade de forma mais honesta que um IC gaussiano.

> **Na biblioteca.** `BootstrapCI(estimator, method="bca"|"percentile", ...)`
> **reamostra dentro de cada braço** (e dentro de bloco×braço), preservando as
> margens do desenho. Não é um bootstrap IID genérico: para experimentos, a
> reamostragem por braço é o esquema correto. Aceita qualquer estimador escalar
> (incluindo `Lin` e `CUPED`, o que dá um IC para eles). O caso degenerado do
> BCa (quando todas ou nenhuma réplica fica abaixo do observado, deixando `z_0`
> indefinido) levanta erro sugerindo `method="percentile"`.

---

## 16. Múltiplos testes: FWER vs. FDR

### Intuição

Testar muitas hipóteses infla a chance de um falso-positivo aparecer só por
acaso. Sob independência, a probabilidade de **pelo menos um** falso-positivo em
`m` testes a nível `α` é

$$1 - (1 - \alpha)^m.$$

Para `α = 0,05` e `m = 10`, isso já é `1 - 0,95^{10} \approx 0,40`: 40% de chance
de uma falsa descoberta. A correção controla esse risco, com duas filosofias.

### Formalização

- **FWER** (*family-wise error rate*): probabilidade de ao menos um
  falso-positivo na família.
  - **Bonferroni**: rejeita se `p ≤ α/m`.
  - **Holm** (step-down): ordene `p_(1) ≤ ... ≤ p_(m)`; compare `p_(i)` a
    `α/(m - i + 1)`; rejeite enquanto passar e pare no primeiro que falhar. Holm
    é uniformemente mais poderoso que Bonferroni e controla o mesmo FWER.
- **FDR** (*false discovery rate*): proporção esperada de falsos-positivos entre
  as rejeições.
  - **Benjamini-Hochberg (BH)**: ordene os p-valores; ache o maior `i` com
    `p_(i) ≤ (i/m)\,q`; rejeite todos até esse `i`.

### Exemplo trabalhado: cinco testes

P-valores ordenados `[0,008, 0,012, 0,020, 0,041, 0,300]`, com `α = q = 0,05`,
`m = 5`.

- **Bonferroni** (limiar `0,05/5 = 0,01`): só `0,008` passa. **1 rejeição.**
- **Holm**: `p_(1)=0,008 ≤ 0,05/5=0,010` (rejeita); `p_(2)=0,012 ≤ 0,05/4=0,0125`
  (rejeita); `p_(3)=0,020 ≤ 0,05/3=0,0167`? Não, `0,020 > 0,0167`, **para**.
  **2 rejeições.**
- **BH**: maior `i` com `p_(i) ≤ (i/5)·0,05`: `i=1` `0,008≤0,010` ✓; `i=2`
  `0,012≤0,020` ✓; `i=3` `0,020≤0,030` ✓; `i=4` `0,041≤0,040` ✗; `i=5`
  `0,300≤0,050` ✗. Maior `i` válido é `3`, então rejeita `p_(1..3)`.
  **3 rejeições.**

O mesmo conjunto rende 1, 2 ou 3 descobertas conforme o método: Bonferroni é o
mais conservador, BH o mais poderoso.

> **Na biblioteca.** `MultipleTestingCorrection(method=..., alpha=...)` com
> `"holm"` (default), `"bonferroni"` e `"bh"`. O `ExperimentComparison` aplica a
> correção sobre uma família de experimentos. O BH assume dependência positiva
> (o caso típico); a variante Benjamini-Yekutieli para dependência arbitrária é
> um item de v2.

### Quando cada um

Use **FWER** quando um único falso-positivo é caro (decisões críticas, registros
regulatórios). Use **FDR** em exploração com muitas métricas, aceitando uma
proporção pequena de erros em troca de mais poder.

---

## 17. Power analysis

### Intuição

Planejar para ter unidades suficientes para enxergar o efeito que você procura.
Quatro quantidades estão amarradas: fixe três e a quarta fica determinada.

- Tamanho da amostra `n`.
- Efeito mínimo detectável (MDE, `δ`).
- Poder `1 - β` (chance de detectar um efeito que existe), por convenção 80% ou
  90%.
- Nível `α` (chance de falso-positivo), por convenção 5%.

### Formalização

Para um teste de diferença de duas médias, o efeito detectável a um dado poder é

$$\delta = (z_{1-\alpha/2} + z_{1-\beta})\,\sigma\,\sqrt{\tfrac{1}{n_T} + \tfrac{1}{n_C}}.$$

Isolando o tamanho com alocação igual (`n` por braço), chega-se a

$$n_{\text{por braço}} = \frac{2\,\sigma^2\,(z_{1-\alpha/2} + z_{1-\beta})^2}{\delta^2}.$$

Com `α = 0,05` (`z = 1,96`) e poder 80% (`z = 0,84`), `(1,96 + 0,84)^2 ≈ 7,85`,
e `2 × 7,85 ≈ 16`, o que dá a regra de bolso

$$n_{\text{por braço}} \approx \frac{16\,\sigma^2}{\delta^2}.$$

### Exemplo trabalhado

MDE `δ = 0,2`, `σ = 1`, `α = 0,05` two-sided, poder 80%, alocação 50/50.

$$
n_{\text{por braço}} \approx \frac{16 \cdot 1}{0{,}04} = 400,
\qquad n_{\text{total}} \approx 800.
$$

A forma exata (a que a biblioteca usa) com `σ_eff = σ\sqrt{1/0,5 + 1/0,5} = 2σ`
arredonda o resultado para cima:

$$n_{\text{total}} = \left\lceil \left(\frac{(z_{1-\alpha/2} + z_{1-\beta})\cdot 2}{0{,}2}\right)^2 \right\rceil.$$

Com os quantis arredondados (`1,96` e `0,84`) o argumento é `(2,80\cdot 2/0,2)^2
= 28^2 = 784`. Com os quantis exatos (`1,95996` e `0,84162`) ele é
`28{,}016^2 \approx 784{,}9`, e o teto leva a `n_total = 785` (392 tratados, 393
controles), próximo do `800` da regra de bolso (o `16` arredonda `15,7`).

> **Na biblioteca.** `power_analysis(n=, mde=, power=, std=, alpha=, allocation=, ...)`
> resolve uma das quantidades dadas as outras, pela forma **exata** com os
> quantis normais (a regra `16σ²/δ²` é só uma checagem rápida). Escopo v1: dois
> grupos, outcome contínuo, aproximação normal. O `σ²` costuma vir de dados
> históricos ou de um teste A/A (ver [V. Diagnósticos](05-diagnosticos.md)).

---

Notebooks relacionados:
[`02_inference_three_ways`](../../../examples/for_starters/pt-br/02_inference_three_ways.ipynb),
[`07_many_tests`](../../../examples/for_starters/pt-br/07_many_tests.ipynb),
[`09_putting_it_together`](../../../examples/for_starters/pt-br/09_putting_it_together.ipynb).
