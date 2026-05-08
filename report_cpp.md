# Coverage Path Planning — Relatório APS

**Aluno:** Pedro Civita | **Disciplina:** Reinforcement Learning, Insper 2026.1 | **Prazo:** 2026-05-08
**Repositório:** https://github.com/pedrocivita/gym_custom_env-pedrotpc

---

## 1. Problema

Treinar um agente RL para **cobrir todas as células livres** em grids 5x5, 10x10 e (bônus) 20x20, sob **observabilidade parcial** — o agente não pode receber o mapa completo do ambiente. Tem acesso a um sensor local 5x5 ao seu redor + informações que ele mesmo derivou da exploração (visited set, obstáculos descobertos pelo sensor).

Baseline (PPO + MultiInputPolicy, 1M timesteps, ent_coef=0.05): ~69-81% full coverage em 5x5 e ~59-70% em 10x10. Falha em 20x20.

## 2. Diagnóstico das limitações da baseline

| Limitação | Efeito |
|-----------|--------|
| Sem memória explícita ou feature de progresso espacial | Agente "esquece" onde já passou; cai em loops em grids maiores |
| `gamma=0.99` com `max_steps=2000` | `0.99^2000 ≈ 1e-7` — bônus de cobertura completa é invisível ao agente no 20x20 |
| `n_steps=512` em episódios de 2000 steps | PPO/GAE fragmenta rollouts em 25% do episódio; advantage estimation lossy |
| Sem action masking | Desperdiça experiência batendo em paredes (que são geometria conhecida) |
| Layouts impossíveis na geração | ~30% dos layouts 10x10 e ~80% dos 20x20 enclausuram células — ceiling natural < 100% |
| Sinal direcional fraco no endgame | Razões direcionais decaem para 1/N com 1 célula faltando — agente não sabe pra onde ir |

## 3. Estratégia adotada (v3.7)

### 3.1 Algoritmo: MaskablePPO

`sb3-contrib.MaskablePPO` com `MultiInputPolicy`. Action masking só de movimentos para fora do grid (geometria conhecida, análoga a paredes de uma sala). **Obstáculos não são mascarados** — o agente os descobre via sensor 5x5 ou pela penalidade `stuck` ao colidir, preservando observabilidade parcial.

### 3.2 Observação (partial observability preservada)

```python
"agent": (10,)  # pose (2), coverage (1), 4 razões direcionais (4),
                # nearest-unvisited compass dx, dy, dist (3)
"neighbors": (5, 5)  # sensor local — única forma de descobrir obstáculos
                     # (0=free, 1=obstáculo/wall, 2=visited)
```

**Compass nearest-unvisited (`dx, dy, dist`):** vetor para a célula não-visitada mais próxima que NÃO seja obstáculo conhecido (revelado pelo sensor). Computado por argmin Manhattan O(N²) sobre o `visited_set` do agente excluindo `_known_obstacles`. **Não é busca de planejamento** — é feature engineering sobre informação que o agente já possui, apresentada de forma mais útil (sinal forte vs ratio direcional fraco no endgame).

**Não viola observabilidade parcial:** todas as features são derivadas da exploração do próprio agente (visited set + sensor history). Nenhuma exposição do mapa global de obstáculos.

### 3.3 Filtro de solvability na geração do ambiente

Em `reset()`, regenera obstáculos até obter layout onde todas as células livres são alcançáveis a partir da posição inicial do agente (BFS reachability check, capped em 200 retries). Sem o filtro, o agente é punido por episódios objetivamente impossíveis de fechar.

**Aprovação do professor (08/05/2026, em aula):** o filtro é permitido desde que viva na geração do ambiente (`reset()`) e seja documentado. Ele NÃO dá nova informação ao agente — apenas remove episódios fora da distribuição válida.

### 3.4 Reward (potential-based shaping)

| Evento | Reward | Justificativa |
|--------|--------|---------------|
| Step base | -0.05 | Pressão temporal leve |
| Célula nova | +1.0 | Sinal principal |
| Revisita | -0.25 | Desencorajar loops |
| Stuck (obstáculo) | -0.5 | Aprender a evitar obstáculos descobertos |
| 25% / 50% / 75% milestone | +2.0 cada | Sinal de progresso intermediário |
| Cobertura completa | +10 × (size/5) | Escala com dificuldade (10/20/40 em 5/10/20) |
| Potential-based shaping | `+10·(γ·cov(s') - cov(s))` | Ng, Harada & Russell (1999): provavelmente reward-invariant; densifica gradiente |

### 3.5 Feature extractor (`utils/feature_extractor.py`)

```python
neighbor_cnn: Conv(1→16, k=3) → Conv(16→32, k=3) → Flatten  # 800 features
agent_mlp:    Linear(10→64) → ReLU                          # 64 features
combine:      Linear(864 → 128) → ReLU                      # 128 features
```

**Param count: 116k uniforme em 5x5, 10x10 e 20x20.** Como ambos inputs são size-invariant (5x5 sensor + 10-feature agent vec), pesos transferem 100% entre tamanhos sem buffer mismatch — viabiliza curriculum learning sem retraining de camada.

### 3.6 Curriculum learning (5x5 → 10x10 → 20x20)

| Stage | Modo | Tsteps | LR | ent_coef | Tempo |
|-------|------|--------|------|----------|-------|
| 5x5   | scratch | 1.0M | 3e-4 | 0.01 | ~10 min |
| 10x10 | transfer | 3.0M | 1e-4 | 0.01 | ~30 min |
| 20x20 | transfer | 1.5M | 5e-5 | 0.01 | ~22 min |

**Total wall-time: 1h06m** em CPU (Lenovo Yoga Book 9i, Intel Core Ultra 7, 14 cores, sem GPU).

Hyperparams chave: `gamma=0.997` (vs default 0.99 — torna full coverage bonus visível em 20x20), `n_steps=2048` (vs default 512 — rollouts cobrem episódio inteiro), `gae_lambda=0.95`, `clip_range=0.2`, `batch_size=256`, `n_epochs=6`, `n_envs=12` (DummyVecEnv), `torch.set_num_threads=3`.

### 3.7 Avaliação dual (deterministic + stochastic)

Cada modelo é avaliado em 100 episódios deterministic (`argmax`) e 100 stochastic (`sample`). Sob observabilidade parcial a política ótima pode ser estocástica — sampling quebra simetrias do estado conhecido que `argmax` perpetua em loops. Reportar ambos é mais honesto.

## 4. Resultados

100 episódios deterministic + 100 stochastic em cada grid. `max_steps` no teste = `dim²×4` (4.5x mais que o número de células livres — teto generoso pra agente bem-formado).

### 4.1 Grid 5x5 (3 obstáculos)

| Métrica | Baseline | Deterministic | **Stochastic** |
|---------|----------|---------------|----------------|
| Full Coverage Rate | 69-81% | **97.0%** | **100.0%** ✅ |
| Avg Coverage | — | 98.5% | 100.0% |
| Avg Steps | — | 26 / 100 | 24 / 100 |

### 4.2 Grid 10x10 (12 obstáculos)

| Métrica | Baseline | Deterministic | **Stochastic** |
|---------|----------|---------------|----------------|
| Full Coverage Rate | 59-70% | 68.0% | **97.0%** ✅ |
| Avg Coverage | — | 90.2% | 99.7% |
| Avg Steps | — | 196 / 400 | 117 / 400 |

### 4.3 Grid 20x20 (48 obstáculos) — bônus

| Métrica | Deterministic | **Stochastic** |
|---------|---------------|----------------|
| Full Coverage Rate | 6.0% | **97.0%** ✅ |
| Avg Coverage | 67.5% | 99.0% |
| Avg Steps | 1529 / 1600 | 581 / 1600 |

**Síntese:** em modo stochastic o agente atinge **100% / 97% / 97% full coverage rate** em 5x5 / 10x10 / 20x20. Avg coverage stochastic é **100% / 99.7% / 99.0%**. Os 3% de "fail" no 10x10/20x20 ainda mostram avg coverage 99% — o agente fecha quase tudo em quase todos os episódios. Atende com folga "cobertura próxima de 100%" pelo critério da rubric.

Curvas de treino (TensorBoard em `log/`):
- `entropy_loss` final: 5x5 -0.143 / 10x10 ~-0.4 / 20x20 -0.443 (política decisiva mas não saturada)
- `explained_variance` final: 0.985 / 0.97 / 0.99 (critic confiante)
- `value_loss` final: 0.92 / ~7 / 23 (escala saudável com size, não artificialmente baixo como em colapso)
- `clip_fraction`: 0.05-0.10 ao final (atualizações modestas, política convergiu)

## 5. Análise — iteração v1 → v3.7 e o que funcionou

### 5.1 O que travou nas iterações intermediárias

| Versão | Problema | Lição |
|--------|----------|-------|
| v1 (RecurrentPPO+LSTM) | Curriculum 5x5→10x10 colapsou (`ep_rew_mean` +25 → -208) | LSTM em CPU = lento + frágil; preferir memória explícita |
| v3 (mask global) | `obstacle_mask` global revelava obstáculos não vistos | Violava partial obs; voltei pra sensor-only |
| v3.3 (reward hacking) | Bônus 90/95/98% + trunc penalty fez agente "evitar fechar" pra evitar penalty | Reward shaping não-potencial pode mudar política ótima |
| v3.5 (visited_map global) | `value_loss=1, EV=0.999` — *parecia* perfeito; 10x10 deu 9% | Critic confiante ≠ policy boa. Atalho do critic colapsa policy |
| v3.6 (minimalist) | Pipeline + 3 polishes travaram em 67% no 10x10 | Polish ataca decisão (entropy); problema era informação |

### 5.2 O insight final (v3.7)

Conversa no WhatsApp da turma com o colega Rodrigo Medeiros (que tirou sucesso): "Ignorar layouts inalcançáveis e adicionar mais informação pro agente — distância pra fronteira". E Pedro Civita confirmou em aula com o professor que o filtro de solvability é permitido na geração do ambiente.

Diagnóstico final do v3.6: o agente conseguia avg 99.3% no 10x10 mas full=67% porque (1) ~30% dos layouts eram impossíveis de fechar e (2) com 1 célula faltando, ratio direcional ≈ 1/50 = 2% — sinal fraco demais pra política comprometer. **Polish bateu no teto** porque atacava decisão (entropy), não informação.

v3.7 ataca a causa real:
1. **Compass `(dx, dy, dist)`**: vetor unitário pra célula não-visitada mais próxima (excluindo obstáculos conhecidos). Quando sobra 1 célula 8 longe: compass = (0.8, 0, 0.4) — sinal **40× mais forte** que o ratio.
2. **Solvability filter**: BFS de alcançabilidade no `reset()`, regenera até layout válido. Elimina ~30% dos 10x10 e ~80% dos 20x20 que tinham pockets cercadas.
3. **ent_coef=0.01 desde 10x10**: v3.6 usava 0.05 no curriculum pra "amaciar transfer", mas isso travava a política em alta entropia. v3.7 usa 0.01 desde o começo — política comprime decisivamente sem perder estabilidade.

### 5.3 Sinais de saúde (v3.7 vs v3.6)

| Sinal | v3.6 5x5 final | **v3.7 5x5 final** |
|-------|---------------|---------------|
| `entropy_loss` | -0.75 | **-0.143** (5× mais decisivo) |
| FPS treino | 1257 | **1694** (35% mais rápido) |
| Stoch Full | 92% | **100%** |

A drop em entropy_loss é a smoking gun: com compass + solvability filter, a ação ótima vira óbvia em todo estado, e a política colapsa pra "siga o compass". Isso explica porque o **deterministic** mode também pula de 21% (v3.6 5x5) pra 97% — o `argmax` agora é confiável, não preso em loops por uniformidade.

### 5.4 Limitações restantes

- Modelo é treinado num pipeline curriculum 5x5→10x10→20x20 — não foi testado generalização zero-shot pra outros tamanhos.
- O filtro de solvability rejeita layouts impossíveis na avaliação — métrica final é "full coverage condicional a layout válido", não "full coverage incondicional". Decisão documentada e aprovada pelo professor.
- 20x20 deterministic = 6%. O `argmax` ainda cai em loops específicos no grid grande; **stochastic = 97%** é a métrica primária.

## 6. Métodos avaliados e não escolhidos

| Método | Por que não usei |
|--------|------------------|
| RecurrentPPO+LSTM (v1) | Tentei na primeira iteração; LSTM em CPU é 5-10× mais lento que feedforward; instável em curriculum |
| DQN com action masking custom | Action masking não nativo no DQN do SB3; risco de não convergir; sem ganho esperado vs MaskablePPO |
| Maskable Recurrent PPO | sb3-contrib não tem essa classe; implementação custom complexa em prazo curto |
| Frame stacking (VecFrameStack) | Considerado como fallback; v3.7 fechou 90%+ sem precisar — adiar |
| Aumentar capacity da rede | Não muda fundamentos; provavelmente não resolveria gap de informação |
| Curriculum granular 5→7→10→13→17→20 | Mais stages = mais tempo; v3.7 com 3 stages bastou |
| Reward intrinsic motivation (RND, count-based) | Implementação complexa; resultados em v3.7 dispensam |
| Hierarchical RL / MAML / Rainbow DQN | Custo/risco proibitivo perto do deadline |
| Imitation learning + BFS teacher | Borderline com restrição "não usar BFS no policy"; v3.7 mostra que não é necessário |

## 7. Como reproduzir

```powershell
# Setup
git clone https://github.com/pedrocivita/gym_custom_env-pedrotpc
cd gym_custom_env-pedrotpc
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Treinar pipeline v3.7 (~70 min em CPU 14 cores)
.\venv\Scripts\python.exe train_v37_pipeline.py --skip-test

# Avaliar (100 det + 100 stoch episódios em cada size, ~5 min)
.\venv\Scripts\python.exe train_grid_world_cpp.py test 5 3
.\venv\Scripts\python.exe train_grid_world_cpp.py test 10 12
.\venv\Scripts\python.exe train_grid_world_cpp.py test 20 48
```

**Modelos finais salvos em `data/`:**
- `maskppo_cpp_5_3_200_<timestamp>.zip` — 5x5 scratch
- `maskppo_cpp_10_12_600_<timestamp>_curr.zip` — 10x10 curriculum
- `maskppo_cpp_20_48_2000_<timestamp>_curr.zip` — 20x20 curriculum

`test_mode` pega automaticamente o modelo mais recente que casa com `(size, obstacles)`.

**Logs TensorBoard em `log/`:**
```powershell
.\venv\Scripts\tensorboard.exe --logdir log
```

---

**Histórico completo das iterações** (v1 → v3.7) está em `EXPERIMENT_LOG.md`.
