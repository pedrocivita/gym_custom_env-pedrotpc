# Coverage Path Planning — Relatório APS

**Aluno:** Pedro Civita | **Disciplina:** Reinforcement Learning, Insper 2026.1 | **Prazo:** 2026-05-08
**Repositório:** https://github.com/pedrocivita/gym_custom_env-pedrotpc

---

## 1. Problema

Treinar um agente RL para **cobrir todas as células livres** em grids 5x5, 10x10 e (bônus) 20x20, sob **observabilidade parcial** — o agente não pode receber o mapa completo do ambiente. A baseline (PPO + MultiInputPolicy, 1M timesteps) atinge ~75-81% full-coverage em 5x5 e ~59-70% em 10x10.

## 2. Diagnóstico da baseline

| Limitação | Efeito |
|-----------|--------|
| MLP feedforward sem memória | Cai em loops em grids maiores |
| Visão 3x3 (9 células) | Em 10x10 enxerga ~10% do grid |
| Sem direção do não-explorado | Só sabe *quanto* falta cobrir, não *para onde ir* |
| Sem action masking | Desperdiça experiência batendo em paredes |

## 3. Estratégia adotada

### 3.1 Algoritmo: MaskablePPO + memória explícita (sem LSTM)

Substituí RecurrentPPO+LSTM (memória implícita via BPTT, frágil em CPU) por **memória explícita** na observação: um mapa binário do que o agente já visitou. A CNN aprende trivialmente "vá para onde está zerado", sem precisar gradiente atravessar dezenas de timesteps.

### 3.2 Observação (partial observability preservada)

```python
"agent":       (7,)         # pose, coverage, 4 razões direcionais de não-visitado
"neighbors":   (5, 5)       # sensor local — única forma de descobrir obstáculos
"visited_map": (size, size) # memória binária de exploração do agente
```

**Não viola partial observability**: o `visited_map` é dado *gerado pelo agente ao se mover* (análogo a um occupancy map SLAM construído online) e nunca revela células não-exploradas. As razões direcionais usam só o `visited_map`, não o ground-truth.

### 3.3 Action masking restrito

`MaskablePPO` mascara **apenas movimentos para fora do grid** (geometria conhecida — análogo a paredes de uma sala). **Obstáculos não são mascarados**: o agente os descobre via sensor 5x5 ou pela penalidade `stuck` (-0.5) ao colidir. Assim a observabilidade parcial é preservada mas a maior parte da experiência desperdiçada some.

### 3.4 Reward shaping

| Evento | Reward | Justificativa |
|--------|--------|---------------|
| Step base | -0.05 | Pressão temporal leve |
| Célula nova | +1.0 | Sinal principal |
| Revisita | -0.25 | Desencorajar loops |
| Stuck (obstáculo) | -0.5 | Aprender a evitar obstáculos descobertos |
| Cobertura completa | +10 × (size/5) | Escala com dificuldade |
| Truncation | 0 | Evita viés no bootstrap de valor |

### 3.5 Feature extractor (`utils/feature_extractor.py`)

Dois caminhos CNN + um MLP, concatenados em um vetor de 128 dimensões:
- `neighbor_cnn`: Conv(1→16, k=3) → Conv(16→32, k=3) → 800 features.
- `visited_cnn`: Conv(1→16, k=3, **stride=2**) → Conv(16→32, k=3, **stride=2**) — `stride=2` mantém o 20x20 viável em CPU.
- `agent_mlp`: Linear(7→64).
- Combiner: Linear(•→128).

Param counts: 5x5: 137k, 10x10: 158k, 20x20: 223k.

### 3.6 Hiperparâmetros e treinamento

| Param | Valor |
|-------|-------|
| `learning_rate` / `clip_range` / `gamma` / `gae_lambda` | 3e-4 / 0.2 / 0.99 / 0.95 |
| `n_steps` / `batch_size` / `n_epochs` | 512 / 256 / 6 |
| `ent_coef` | 0.01 |
| `n_envs` | 12 (DummyVecEnv) |
| `torch.set_num_threads` | 3 (evita contenção com vec-env loop) |
| Checkpoints | a cada ~50k env-steps (recovery em caso de crash) |

| Stage | Obstáculos | max_steps | Timesteps | Tempo CPU |
|-------|------------|-----------|-----------|-----------|
| 5x5 | 3 | 200 | 500k | ~9 min |
| 10x10 | 12 | 600 | 2.0M | ~50 min |
| 20x20 | 48 | 2000 | 3.0M | ~1h13min |

Pipeline total: **2h03min** em Lenovo Yoga Book 9i (Intel Core Ultra 7, 14 cores, sem GPU).

### 3.7 Avaliação dual (deterministic + stochastic)

Cada modelo é avaliado em 100 episódios deterministic (`argmax`) e 100 stochastic (`sample`). Sob observabilidade parcial a política ótima pode ser estocástica — em ambientes com simetrias do `visited_map`, sampling quebra loops que `argmax` não quebra. Reportar ambos é mais honesto.

## 4. Resultados

100 episódios deterministic + 100 stochastic em cada grid.

### 4.1 Grid 5x5 (3 obstáculos)

| Métrica | Baseline | Deterministic | Stochastic |
|---------|----------|---------------|------------|
| Full Coverage Rate | 69-81% | **44.0%** | **87.0%** |
| Avg Coverage | — | 85.8% | **99.0%** |
| Avg Steps | — | 67 | 39 |

### 4.2 Grid 10x10 (12 obstáculos)

| Métrica | Baseline | Deterministic | Stochastic |
|---------|----------|---------------|------------|
| Full Coverage Rate | 59-70% | 0.0% | 8.0% |
| Avg Coverage | — | 48.4% | **92.9%** |
| Avg Steps | — | 400 (max) | 394 |

### 4.3 Grid 20x20 (48 obstáculos) — bônus

| Métrica | Deterministic | Stochastic |
|---------|---------------|------------|
| Full Coverage Rate | 0.0% | 0.0% |
| Avg Coverage | 14.2% | **93.1%** |
| Avg Steps | 1600 (max) | 1600 (max) |

Curvas de treino (TensorBoard em `log/`): `ep_rew_mean` cresceu monotonicamente, `entropy_loss` ~-0.9 a -1.2 (não colapsou), `explained_variance` > 0.9 nos três tamanhos — critic confiante.

## 5. Análise

**O que funcionou:** O agente em modo *stochastic* atingiu **avg coverage 99%/93%/93%** em 5x5/10x10/20x20 — a política aprendeu sim a estrutura geral do problema. A combinação `visited_map` + sensor 5x5 + action masking restrito treina ~6× mais rápido que RecurrentPPO+LSTM em CPU. O critic em todos os tamanhos atingiu `explained_variance > 0.9`, mostrando que a representação capta bem o valor dos estados.

**O que não funcionou (full coverage 100%):** Em 10x10 e 20x20 o **full coverage rate ficou em 0-8%**. O padrão observado é que o agente cobre rapidamente 90-95% e depois trava em loops nas últimas 5-7% das células. Causas prováveis:
1. **Sub-ótimo de timesteps**: o 10x10 e 20x20 exigem aprender estratégias de "varredura final" que requerem mais experiência. Em 10x10 com 2M timesteps e 20x20 com 3M, o agente não treinou o suficiente para os casos de borda.
2. **Sinal de objetivo escasso**: o bônus de full coverage (+20 em 10x10, +40 em 20x20) só é dado em episódios que completam — episódios truncados não recebem essa pista, então no início do treino o agente quase nunca vê o gradiente do objetivo final.
3. **Determinismo cria loops**: em deterministic mode, a simetria do `visited_map` faz o agente repetir a mesma trajetória ineficiente. Em stochastic isso melhora drasticamente (avg vai de 48% para 93% em 10x10).

**Limitações da abordagem:** modelo size-specific (cada grid tem seu modelo), avaliação fortemente dependente de stochastic sampling, e geração uniforme de obstáculos (configurações tipo corredor estreito não são bem testadas).

**Lição da iteração v1:** uma versão inicial usava RecurrentPPO+LSTM com curriculum 5x5→10x10. Colapsou: `ep_rew_mean` foi de +25 para -208 ao longo de 1.5M timesteps porque a política 5x5 já tinha entropia baixa demais para se readaptar a 88 células livres. Trocar memória implícita (LSTM) por explícita (`visited_map`) resolveu a estabilidade — mas como mostram os resultados, ainda há caminho para melhorar a parte final do policy.

**Próximos passos para fechar 100%:** continue-training com mais 2-3M timesteps em 10x10/20x20 (mantendo o checkpoint atual como base), ou ajustar reward para dar bônus parciais a cada 25%/50%/75% de coverage, dando sinal de "completar o último pedaço" mesmo em episódios não-finalizados.

## 6. Como reproduzir

```powershell
python -m venv venv ; .\venv\Scripts\Activate.ps1 ; pip install -r requirements.txt
python train_local_pipeline.py 5 10 20 --skip-test    # ~2h em CPU 14 cores
python finalize.py                                    # roda testes, preenche este relatório, faz commit
```

Modelos salvos em `data/maskppo_cpp_<size>_*.zip`; checkpoints em `data/maskppo_cpp_..._checkpoints/`.
