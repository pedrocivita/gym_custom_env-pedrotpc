# Experiment Log — APS CPP

Histórico cronológico das iterações de design e treino. Mantido aqui para auditabilidade da entrega.

---

## v1 (2026-05-04) — RecurrentPPO + LSTM + curriculum (FALHOU)

**Hipótese**: a baseline (PPO + 3x3 view, 1M tsteps) falha por falta de memória. Solução clássica: substituir por RecurrentPPO com LSTM, e usar curriculum learning 5x5 → 10x10 conforme sugerido no enunciado.

**Setup**:
- `RecurrentPPO("MultiInputLstmPolicy", ...)` do `sb3-contrib`
- LSTM hidden 128, 1 layer, separate actor/critic LSTMs
- Observação: `agent (3) + neighbors (5,5)` (já expandida para 5x5 na primeira iteração)
- Treinado em Google Colab Free (GPU T4)
- 5x5: 2M tsteps. Atingiu avg coverage ~96%.
- 10x10: curriculum a partir do 5x5, 1.5M tsteps adicionais.

**Resultado**: 10x10 colapsou. `ep_rew_mean` foi de +25 para -208 ao longo de 1.5M tsteps. Diagnose:
1. A política 5x5 já tinha entropia baixa (~-0.5) — pouco espaço para readaptar a 88 cells livres do 10x10.
2. LR=3e-4 alto demais para fine-tune de política convergida.
3. LSTM com BPTT é frágil em CPU e o sinal de objetivo é esparso (só recebe +bônus em episódios completos).

**Decisão**: abandonar LSTM e curriculum. Pivotar para memória explícita.

---

## v2 (2026-05-04) — Versão Colab nunca executada

Plano de 10M timesteps total (2M+5M+3M) overnight no Colab T4. Estimado 6-7h. Cancelado pelo risco de desconexão da sessão Free e pelo aprendizado do v1.

---

## v3 (2026-05-05) — MaskablePPO + visit-map global (VIOLAVA RUBRIC)

**Hipótese**: substituir LSTM por canal de mapa global na observação — memória explícita em vez de implícita. Adicionar action masking via `MaskablePPO`.

**Setup**:
- `MaskablePPO + MultiInputPolicy`
- Observação: `agent (7) + global_map (3, H, W)` com canais para obstáculos / visitados / agente
- Action masking baseado no obstacle_grid completo
- Reward limpo: -0.05 step, +1.0 nova, -0.25 revisita, +10×(size/5) full coverage, sem stuck/truncation penalty

**Resultado**: ~6× speedup vs RecurrentPPO em CPU. Métricas saudáveis. Mas...

**Problema descoberto durante revisão do enunciado**: o canal `obstacle_mask` global revelava todos os obstáculos do mapa de uma vez — **viola explicitamente** o requisito de observabilidade parcial. O `action_masks()` baseado em `obstacle_grid` completo também violava.

**Decisão**: refatorar para v3.1 mantendo arquitetura mas corrigindo a violação.

---

## v3.1 (2026-05-05) — partial-observability compliant

**Mudanças**:
- `obstacle_mask` global removido. Substituído por sensor local 5x5 (`neighbors`).
- `action_masks()` agora só bloqueia movimentos para fora do grid (geometria conhecida).
- Stuck penalty (-0.5) volta — o agente descobre obstáculos por colisão.
- `visited_map` global mantido com justificativa: é dado *gerado pelo agente ao explorar*, análogo a um occupancy map SLAM construído online.

**Hyperparams**:
- `n_envs=12, batch_size=256, n_epochs=6, ent_coef=0.01`
- `torch.set_num_threads=3` (evita contenção em DummyVecEnv)
- 5x5: 500k, 10x10: 2M, 20x20: 3M timesteps (~2h03min)

**Resultados (100 ep det + 100 ep stoch)**:

| Grid | Det Full | Stoch Full | Stoch Avg |
|------|---------:|-----------:|----------:|
| 5x5 | 35% | 87% | 99.0% |
| 10x10 | 0% | 8% | 92.9% |
| 20x20 | 0% | 0% | 93.1% |

**Diagnóstico**: avg coverage altíssimo (93-99%) mas full coverage rate insuficiente em 10x10/20x20. Causa: **sparse terminal reward**. O bônus de full coverage só é dado em episódios que completam, e episódios truncam quase sempre no início do treino — então o gradiente do objetivo final nunca chega à política. O agente otimiza para "cobrir muito" mas nunca aprende "fechar o último 5-10%".

**Outro problema**: `visited_map` global tem shape variável (size×size) → quebra transfer learning entre tamanhos (Linear final do CNN tem shape diferente).

**Decisão**: refatorar para v3.2 com janela 5x5 fixa (transfer learning trivial) + reward com bônus parciais (atacar sparse terminal reward).

---

## v3.2 (2026-05-07) — 5x5 windows + curriculum + bônus parciais

**Mudanças centrais**:
- `visited_map` global → `visited_neighbors` (5x5 fixo, janela centrada no agente)
- Param count idêntico em 5x5/10x10/20x20 (223k cada) → curriculum funciona
- Reward: bônus +2.0 a cada 25%, 50%, 75% de coverage (uma vez por episódio)
- Curriculum mode: 5x5 (scratch) → 10x10 (transfer com lr=1e-4, ent=0.03) → 20x20 (transfer)

**Pipeline**: `train_curriculum_pipeline.py` com 600k+1M+1.5M = 3.1M timesteps.

**Wall-time real**: 1h48min (5x5: 14min, 10x10: 27min, 20x20: 1h06m).

**Métricas de treino (sinais de saúde)**:
- 5x5 final: `explained_variance ≈ 0.6, entropy_loss ≈ -0.92`
- 10x10 final: `explained_variance ≈ 0.87, entropy_loss ≈ -0.82`
- 20x20 final: `explained_variance ≈ 0.98, entropy_loss ≈ -0.92` ← critic praticamente perfeito

**Resultados (100 ep det + 100 ep stoch)**:

| Grid | Det Full | Stoch Full | Stoch Avg | Stoch Steps |
|------|---------:|-----------:|----------:|------------:|
| 5x5 | 65.0% | **94.0%** ✅ | 99.7% | 36 |
| 10x10 | 2.0% | 77.0% | 98.1% | 272 |
| 20x20 | 0.0% | 1.0% | 97.8% | 1596 |

**Avanço vs v3.1**:
- 5x5 stoch full: 87% → **94%** (+7pp) — passa o critério de 90%
- 10x10 stoch full: 8% → **77%** (+69pp) — quase passa
- 20x20 stoch full: 0% → 1% — não fechou (ainda em loop nas últimas células)

**Diagnóstico para 20x20**: avg=97.8% significa que o agente cobre ~344 de 352 cells livres em média (perde ~7-8 cells). Em deterministic mode com max_steps=1600, sempre estoura o budget — `avg_steps=1600 std=0`. O policy não tem "endgame skill" para encontrar células isoladas em cantos / atrás de obstáculos.

**Causa estrutural**: bônus parciais 25/50/75 funcionam até 75%. Acima disso, agente fica sem sinal denso até o full coverage (+40 em 20x20) que ele raramente vê. Sparse reward novamente, mas agora só nos últimos 25% do estado.

**Decisão**: v3.3 com bônus densos no endgame (90/95/98%) + penalty de truncation proporcional à cobertura faltante.

---

## v3.3 (2026-05-07) — endgame reward shaping

**Mudanças no reward**:
- Bônus parciais 25/50/75 mantidos (+2.0 cada)
- **Adicionado**: bônus +5.0 a 90%, +5.0 a 95%, +5.0 a 98% (uma vez por episódio cada)
- **Adicionado**: truncation penalty proporcional `-5.0 * (1 - coverage_ratio)`. Episódio que termina com 50% de coverage leva -2.5; com 95% leva -0.25.

**Lógica**:
1. Bônus densos no endgame força o agente a aprender o "último mile" — sinal de gradiente continua até 98%.
2. Penalty proporcional puxa o agente a maximizar coverage mesmo em episódios que truncam, em vez de só desistir.
3. Mantém compatibilidade total com checkpoints v3.2 (mesma observação).

**Pipeline**: `train_curriculum_pipeline.py` re-executado com novo reward, mesmos hyperparams.

**Wall-time**: 1h14m48s (mais rápido que v3.2 1h48m — episódios completam mais cedo, aumentando throughput médio).

**Métricas finais de treino (sinais ENGANOSAMENTE bons)**:
- 5x5 final: `explained_variance≈0.85, entropy_loss≈-1.0`
- 10x10 final: `explained_variance=0.86, entropy_loss=-1.02, value_loss=15.8` (melhor que v3.2!)
- 20x20 final: `explained_variance=0.99, entropy_loss=-1.01, value_loss=2.0` (critic praticamente perfeito)

**Resultados de avaliação (PIORARAM vs v3.2)**:

| Grid | Det Full | Stoch Full | Stoch Avg |
|------|---------:|-----------:|----------:|
| 5x5 | 34.0% | 91.0% ✅ | 99.3% |
| 10x10 | 0.0% | **50.0%** ⬇ | 98.7% |
| 20x20 | 0.0% | **0.0%** ⬇ | 94.7% |

Comparado com v3.2: 5x5 caiu 3pp (94→91), 10x10 caiu **27pp** (77→50), 20x20 caiu 1pp (1→0). 5x5 ainda passa o critério de 90% mas margem encolheu.

**Diagnóstico do retrocesso** — três hipóteses convergentes:

1. **Truncation penalty proporcional rewardou conservadorismo**: o agente passou a evitar exploração arriscada (que poderia falhar em fechar coverage) em favor de manter-se em regiões já exploradas. Episódio truncado com 95% custa só -0.25, mas com 50% custa -2.5; o agente aprende a NUNCA ir abaixo de 90% mesmo que isso o impeça de "investir" em fechar 100%.

2. **Bônus 90/95/98% removeram pressão pra fechar**: com +15.0 disponíveis nesses três milestones (mais barato do que tentar full coverage), o policy aprendeu "atinge milestones, não tenta fechar". É a clássica falha do reward shaping: a agente otimiza o reward, não o objetivo verdadeiro.

3. **Sinais de treino enganaram**: `value_loss=2.0` no fim parecia excelente, mas refletia que o critic estava modelando bem um POLICY MAIS CONSERVADOR (que tem returns mais previsíveis pq foge de risco). Critic confiante + policy ruim = underdiagnosis.

**Decisão**: reverter o env para reward v3.2, apagar os modelos v3.3, e perseguir melhorias via **continue-training nos checkpoints v3.2** (que tinham 5x5=94%, 10x10=77%, 20x20=1%).

**Lição RL**: reward shaping é uma faca de dois gumes. Bônus densos podem ajudar exploração mas também distorcer o objetivo. A regra geral de Ng et al. (1999) sobre potential-based shaping NÃO foi seguida aqui (os bônus de v3.3 não são funções potenciais), o que abre espaço para "reward hacking" — exatamente o que aconteceu.

---

## Continue-training v3.2 com gamma corrigido (2026-05-07)

**Diagnóstico crítico antes de rodar**: revisão da configuração revelou que `gamma=0.99` provavelmente é a causa raiz do problema do 20x20 não fechar. O bônus de full coverage (+10×size/5) só é dado no fim do episódio. Com `gamma=0.99` e `max_steps=2000` no 20x20, o discount factor `0.99^2000 ≈ 0` faz esse sinal **invisível** para a política — o agente literalmente não enxerga o bônus de "fechar 100%" através de 2000 timesteps de discount. Isso é consistente com os resultados observados:

| Grid | max_steps | `0.99^max_steps` | Observado v3.2 stoch full |
|------|----------:|-----------------:|--------------------------:|
| 5x5 | 200 | 0.13 | 94% ✅ |
| 10x10 | 600 | 0.0024 | 77% (borderline) |
| 20x20 | 1500-2000 | ~1e-7 | 1% ❌ |

A correlação é direta: onde o discount factor é praticamente zero, o agente otimiza só os bônus parciais e ignora a recompensa terminal. Mudar `gamma → 0.997` faz `0.997^2000 ≈ 0.0025` — pequeno mas detectável.

**Setup ajustado**: `continue_pipeline.py` carrega os checkpoints v3.2 e continua o treino com:
- `lr=1e-4` (vs 5e-5 anterior — 5e-5 era conservador demais para ainda mover a política)
- `ent_coef=0.05` (combate loops em deterministic)
- **`gamma=0.997`** (o ajuste mais importante; override aplicado após `MaskablePPO.load`)

| Stage | Tsteps adicional | ETA |
|-------|------------------:|-----|
| 10x10 | +1.5M | ~36 min |
| 20x20 | +4.0M (era 3M; +1M extra) | ~2h47m |

Total: ~3h22m.

**Resultados**: (a preencher)

---

## Decisões rejeitadas e por quê

| Ideia | Por que rejeitada |
|-------|---|
| BFS frontier no policy | Algoritmo clássico, não RL. O professor proibiu explicitamente em conversa entre alunos (Matheus/Pedro, 05/05). |
| AdaptiveAvgPool no CNN para shape unificado | Destrói localização espacial — o `agent_mask` é 1 em apenas 1 célula, averaging dilui. |
| Treinar com obstáculos densos custom | Manteria avaliação igual ao enunciado e não generalizaria; tempo limitado. |
| Hybrid RL + BFS heuristic para endgame | Mesma razão de "BFS frontier no policy" — viola RL puro. |
| Reward intrinsic motivation (RND, count-based) | Implementação complexa, sem garantia de ganho marginal vs. shaping direto. |

---

## Hardware e infraestrutura

- Treino: Lenovo Yoga Book 9i, Intel Core Ultra 7, 14 cores, **sem GPU**.
- Tentativa de Colab T4 abandonada após v1 (instabilidade da sessão Free para runs longos).
- Decisões de design — sem LSTM, stride-2 (até v3.1), 5x5 windows fixos (v3.2+), torch threads cap, n_envs=12 — todas voltadas a maximizar throughput em CPU.
- v3.2 atinge ~700-800 fps em 10x10, ~480 fps em 20x20 (pipeline 1h48m end-to-end).
