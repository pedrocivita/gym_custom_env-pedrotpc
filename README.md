# Coverage Path Planning com Observabilidade Parcial — APS de RL

**Autor:** Pedro Civita
**Disciplina:** Reinforcement Learning, 10º semestre — Insper
**Entrega:** APS de RL, prazo 2026-05-08

> **Relatório completo:** [`report_cpp.md`](report_cpp.md)
> **Histórico de iterações (v1 → v3.7):** [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)

---

## Resultado final (v3.7)

| Grid | Stoch Full | Stoch Avg | Det Full | Det Avg |
|------|-----------:|----------:|---------:|--------:|
| **5×5** | **100.0%** ✅ | 100.0% | 97.0% | 98.5% |
| **10×10** | **97.0%** ✅ | 99.7% | 68.0% | 90.2% |
| **20×20** (bônus) | **97.0%** ✅ | 99.0% | 6.0% | 67.5% |

Avaliação em 100 episódios deterministic + 100 stochastic. Sob observabilidade parcial a política ótima pode ser estocástica — a coluna **Stochastic** é a leitura primária. Pipeline completo treina em **1h06m** em CPU (Intel Core Ultra 7, 14 cores, sem GPU).

## Sobre o projeto

Fork do [`fbarth/gym_custom_env`](https://github.com/fbarth/gym_custom_env) com um ambiente Gymnasium para **Coverage Path Planning (CPP)** sob **observabilidade parcial**: agente precisa visitar todas as células livres do grid sem acesso ao mapa global, descobrindo obstáculos via sensor local 5x5.

Baseline (PPO + MultiInputPolicy): ~69-81% full em 5x5, ~59-70% em 10x10, falha em 20x20. Meta da APS: cobertura próxima de 100% em 5x5 e 10x10 (nota 10), idealmente em 20x20 (nota 11 bônus).

## Estratégia (v3.7)

Algoritmo: **MaskablePPO** (sb3-contrib) + curriculum learning 5x5 → 10x10 → 20x20.

**Observação (10 floats + 5x5 sensor, partial-obs respeitada):**
- `agent` (10): pose + coverage + 4 razões direcionais + **nearest-unvisited compass `(dx, dy, dist)`**
- `neighbors` (5×5): sensor local (0=free / 1=obstáculo / 2=visited)

**3 mudanças críticas vs versões anteriores:**
1. **Compass de não-visitado** — vetor para a célula mais próxima que o agente não visitou e não conhece como obstáculo. Resolve o gap de informação no endgame (razões direcionais decaem para 1/N quando sobra 1 célula).
2. **Solvability filter no `reset()`** — descarta layouts onde células livres ficam encerradas por obstáculos (ato de geração do ambiente, aprovado pelo professor). Sem o filtro, ~30% dos 10x10 e ~80% dos 20x20 eram impossíveis de fechar 100%.
3. **`ent_coef=0.01` desde 10x10** — versões anteriores usavam 0.05 no curriculum pra "amaciar transfer" mas isso travava a política em alta entropia. 0.01 mantém política decisiva sem prejudicar adaptação.

Reward: `-0.05` step base, `+1.0` célula nova, `-0.25` revisita, `-0.5` stuck, milestones `+2.0` em 25/50/75%, `+10×(size/5)` full coverage. Potential-based shaping (`Φ = 10·coverage_ratio`, `γ=0.997`) dá gradiente denso (Ng et al. 1999, reward-invariant).

Hyperparams: `gamma=0.997`, `n_steps=2048`, `gae_lambda=0.95`, `clip_range=0.2`, `batch_size=256`, `n_epochs=6`, `n_envs=12`.

## Estrutura

```
gym_custom_env-pedrotpc/
├── gymnasium_env/grid_world_cpp.py     # Env v3.7 (compass + solvability filter)
├── utils/feature_extractor.py          # CNN(neighbors) + MLP(agent vec) → 128
├── train_grid_world_cpp.py             # train / curriculum / test / run modes
├── train_v37_pipeline.py               # Pipeline 5x5 → 10x10 → 20x20 (1h06m)
├── train_curriculum_pipeline.py        # Pipeline v3.6 (legado)
├── train_isolated_pipeline.py          # Treino isolado por size (v3.5 legado)
├── continue_pipeline.py                # Continue-training de checkpoint
├── polish_pipeline.py                  # Polish ent_coef (v3.6 legado)
├── finalize.py                         # Roda testes + atualiza report + commita
├── report_cpp.md                       # Relatório principal
├── EXPERIMENT_LOG.md                   # Histórico v1 → v3.7
├── data/                               # Modelos treinados (versionados)
└── log/                                # TensorBoard / CSV (gitignored)
```

## Reproduzir

```powershell
# 1. Setup
git clone https://github.com/pedrocivita/gym_custom_env-pedrotpc.git
cd gym_custom_env-pedrotpc
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Treinar pipeline v3.7 (~1h06m em CPU 14 cores)
.\venv\Scripts\python.exe train_v37_pipeline.py --skip-test

# 3. Avaliar (100 det + 100 stoch episódios em cada size, ~5 min)
.\venv\Scripts\python.exe train_grid_world_cpp.py test 5 3
.\venv\Scripts\python.exe train_grid_world_cpp.py test 10 12
.\venv\Scripts\python.exe train_grid_world_cpp.py test 20 48

# 4. Visualizar um episódio interativo (pygame)
.\venv\Scripts\python.exe train_grid_world_cpp.py run 10 12
```

## Comandos individuais

```powershell
# Treinar um size do zero (dim, n_obstacles, max_steps, total_timesteps)
python train_grid_world_cpp.py train 10 12 600 3000000

# Curriculum (transfer de um modelo base)
python train_grid_world_cpp.py curriculum 10 12 600 3000000 data/.../base.zip

# Testar (modo dual: deterministic + stochastic, 100 ep cada)
python train_grid_world_cpp.py test 10 12

# Rodar um episódio com renderização pygame
python train_grid_world_cpp.py run 10 12

# TensorBoard
.\venv\Scripts\tensorboard.exe --logdir log
```

## Recovery a partir de checkpoint

Cada treino salva checkpoints em `data/<model_name>_checkpoints/ckpt_<step>_steps.zip` a cada ~50k env steps. Em caso de crash, qualquer checkpoint é um modelo completo testável.

## Modelos finais (em `data/`)

```
maskppo_cpp_5_3_200_<timestamp>.zip            # 5x5 scratch v3.7
maskppo_cpp_10_12_600_<timestamp>_curr.zip     # 10x10 curriculum v3.7
maskppo_cpp_20_48_2000_<timestamp>_curr.zip    # 20x20 curriculum v3.7
```

`test_mode` pega automaticamente o modelo mais recente que casa com `(size, obstacles)`.

---

## Projeto base

Fork de [`fbarth/gym_custom_env`](https://github.com/fbarth/gym_custom_env). Os ambientes anteriores (GridWorld v0, 3D, obstacles) e seus scripts (`run_grid_world_v0.py`, `train_grid_world_obstacles.py`, etc.) foram preservados.
