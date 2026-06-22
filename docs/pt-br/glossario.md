# Glossário

Termos centrais de experimentação e inferência causal, na linguagem da
biblioteca. A ordem vai do mais fundamental ao mais específico.

## Fundamentos

**Potential outcomes (`Y(0)`, `Y(1)`).** Os dois resultados que uma unidade
*teria* sob controle e sob tratamento. O problema fundamental da inferência
causal: só observamos **um** por unidade; o outro é contrafactual.

**Efeito individual (ITE).** `Y(1) - Y(0)` para uma unidade. Nunca observável
diretamente.

**ATE (*Average Treatment Effect*).** A média dos efeitos individuais na
população de interesse. É o alvo padrão dos estimadores da lib.

**SATE vs. PATE.** *Sample* ATE (efeito médio **nestas** unidades) vs.
*Population* ATE (efeito médio na população maior). A distinção liga ao ponto
seguinte.

**População finita vs. superpopulação.** Inferir sobre *estas* unidades
(finita; `NeymanCI`) vs. sobre uma população maior da qual elas são amostra
(superpopulação; `BootstrapCI`). Não é detalhe técnico: muda o que o intervalo
de confiança significa.

## Desenho

**Randomização.** Atribuir o tratamento ao acaso. É o que permite atribuir
diferenças de resultado ao tratamento, e não a confundidores.

**Confundidor.** Variável que afeta tanto a chance de receber o tratamento
quanto o resultado. Randomização quebra o confundimento por construção.

**CRD (*Completely Randomized Design*).** Metade (ou proporção `p`) das
unidades ao tratamento, ao acaso, sem estrutura.

**Blocagem.** Randomizar dentro de estratos (blocos) para garantir equilíbrio
em variáveis conhecidas. Implementado por `BlockedCRD`.

**Rerandomização.** Sortear de novo até a alocação ficar equilibrada por um
critério (distância de Mahalanobis). Implementado por `ReRandomizedCRD`.

**Fatorial 2^K.** Desenho com `K` fatores binários e todas as combinações.
Permite estimar efeitos principais e interações.

## Inferência

**Sharp null (Fisher).** Hipótese nula **forte**: efeito **zero em todas** as
unidades. É o que o `RandomizationTest` testa permutando o tratamento.

**Inferência baseada em randomização.** O p-valor vem do mecanismo de
randomização que você escolheu, não de pressupostos distribucionais.

**Variância de Neyman.** Estimador conservador da variância do ATE sob
população finita (`NeymanCI`).

**Bootstrap.** Reamostragem com reposição para aproximar a distribuição
amostral sob superpopulação (`BootstrapCI`): percentil ou BCa.

**p-valor.** Probabilidade, sob a nula, de um efeito tão ou mais extremo que o
observado. Pequeno sugere que a nula é implausível.

**Intervalo de confiança.** Faixa de valores plausíveis para o efeito, ao
nível `1 - alpha`.

## Múltiplos testes

**FWER (*family-wise error rate*).** Probabilidade de **pelo menos um**
falso-positivo numa família de testes. Controlada por Bonferroni e Holm.

**FDR (*false discovery rate*).** **Proporção** esperada de falsos-positivos
entre as descobertas. Controlada por Benjamini-Hochberg (BH).

## Diagnósticos

**SMD (*Standardized Mean Difference*).** Diferença de médias padronizada,
usada para checar equilíbrio de covariáveis. Regra prática: `|SMD| > 0.1`
indica desequilíbrio relevante.

**SRM (*Sample Ratio Mismatch*).** A alocação observada destoa da pretendida.
É um alarme de **bug de implementação** (logging assimétrico, filtro de bots),
não uma hipótese científica; por isso o limiar estrito de 0.001.

**Teste A/A.** Re-randomizar sobre dados sem efeito para checar a calibração
do pipeline: a taxa de falso-positivo deve bater com `alpha`.
