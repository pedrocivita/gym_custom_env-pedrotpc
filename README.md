# Coverage Path Planning com Observabilidade Parcial — APS de RL

**Autor:** Pedro Civita
**Disciplina:** Reinforcement Learning, 10º semestre — Insper
**Entrega:** APS de RL, prazo 2026-05-08

> **Relatório completo:** [`report_cpp.md`](report_cpp.md)

---

## Sobre o repositório

Este repositório é um fork do projeto base [`fbarth/gym_custom_env`](https://github.com/fbarth/gym_custom_env), que provê um ambiente Gymnasium customizado para o problema de **Coverage Path Planning (CPP)**. O agente precisa visitar todas as células livres de um grid evitando obstáculos, sob **observabilidade parcial** (não pode acessar o mapa completo do ambiente).

A baseline (PPO + MultiInputPolicy) atinge ~75-81% de full coverage em 5x5 e ~59-70% em 10x10. A meta da APS é chegar próximo de 100% em 5x5 e 10x10 (nota 10) e idealmente em 20x20 (nota 11, bônus).

## Estratégia adotada (v3.1)

A solução final substitui PPO + MLP por **MaskablePPO** com uma observação que combina visão local e memória explícita do agente:

- `agent` (7 floats): pose normalizada, coverage ratio, e 4 razões direcionais de células ainda não visitadas (right/up/left/down) calculadas só a partir do que o agente já visitou.
- `neighbors` (5×5): janela local centrada no agente — única forma de descobrir obstáculos, junto com a colisão (`stuck` penalty).
- `visited_map` (size×size): mapa binário **gerado pelo próprio agente** ao se mover. Análogo a um mapa de ocupação construído online via SLAM. Não revela obstáculos não-explorados — preserva observabilidade parcial.

**Por que MaskablePPO + visit-map em vez de RecurrentPPO+LSTM?**
- A "memória" passa a ser explícita no input em vez de implícita no estado oculto da LSTM. Treina mais rápido em CPU e é robusto contra colapso de entropia.
- `MaskablePPO` permite mascarar movimentos contra a fronteira do grid (geometria conhecida) sem violar partial observability — obstáculos continuam sendo descobertos por colisão.
- Resultado: **~6× speedup** vs. RecurrentPPO no CPU usado, mantendo qualidade.

Detalhamento completo (incluindo o caminho v1 → v3 → v3.1 e por que cada decisão) está em [`report_cpp.md`](report_cpp.md).

## Estrutura

```
gym_custom_env-pedrotpc/
├── gymnasium_env/grid_world_cpp.py    # Environment v3.1 (partial observability)
├── utils/feature_extractor.py         # CNN dual-path (neighbors + visited_map) + agent MLP
├── train_grid_world_cpp.py            # train / test / run modes (MaskablePPO)
├── train_local_pipeline.py            # Encadeia 5x5 → 10x10 → 20x20 com checkpoints
├── finalize.py                        # Roda testes em todos os modelos, preenche report, faz commit
├── report_cpp.md                      # Relatório principal — leia primeiro
├── data/                              # Modelos treinados + checkpoints (gitignored)
└── log/                               # TensorBoard / CSV (gitignored)
```

## Reproduzir do zero

```powershell
# 1. Setup
git clone https://github.com/pedrocivita/gym_custom_env-pedrotpc.git
cd gym_custom_env-pedrotpc
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Treinar todos os tamanhos em sequência (~3h41m em CPU 14 cores)
python train_local_pipeline.py 5 10 20 --skip-test

# 3. Avaliar (100 ep deterministic + 100 stochastic em cada tamanho)
python train_grid_world_cpp.py test 5 3
python train_grid_world_cpp.py test 10 12
python train_grid_world_cpp.py test 20 48

# 4. Visualizar um episódio interativo (pygame)
python train_grid_world_cpp.py run 10 12

# 5. Atualizar o report com os números reais e commitar
python finalize.py
```

## Comandos individuais (referência)

```powershell
# Treinar um único tamanho (dim, n_obstacles, max_steps, total_timesteps)
python train_grid_world_cpp.py train 10 12 600 2000000

# Testar (modo dual: deterministic + stochastic, 100 ep cada)
python train_grid_world_cpp.py test 10 12

# Rodar um único episódio com renderização
python train_grid_world_cpp.py run 10 12

# TensorBoard (gráficos de ep_rew_mean, ep_len_mean, entropy_loss, etc.)
.\venv\Scripts\tensorboard.exe --logdir log
```

## Recovery a partir de checkpoint

Cada treino salva checkpoints em `data/<model_name>_checkpoints/ckpt_<step>_steps.zip` a cada ~50k env steps. Em caso de crash:

```powershell
python -c "from sb3_contrib import MaskablePPO; m = MaskablePPO.load('data/.../ckpt_1500000_steps.zip'); print('loaded')"
```

Esses checkpoints são modelos completos — testáveis com `train_grid_world_cpp.py test` passando o caminho explícito como 4º argumento.

## Resultados resumidos

> Avaliação em 100 episódios deterministic + 100 stochastic. Veja [`report_cpp.md`](report_cpp.md) Seção 4 para tabelas completas e análise.

| Grid | Avg Coverage (stoch) | Full Coverage (stoch) | Avg Coverage (det) | Full Coverage (det) |
|------|---------------------:|----------------------:|-------------------:|--------------------:|
| 5×5 | **99.0%** | 87.0% | 85.8% | 44.0% |
| 10×10 | **92.9%** | 8.0% | 48.4% | 0.0% |
| 20×20 | **93.1%** | 0.0% | 14.2% | 0.0% |

A política sob observabilidade parcial é fundamentalmente estocástica — a coluna **Stochastic** é a leitura primária. Avg coverage 99% / 93% / 93% atende ao critério "cobertura próxima de 100%" pela métrica de cobertura média; full coverage (atingir exatamente 100% das células livres) fica abaixo do alvo em 10×10/20×20, com hipóteses RL-fundamentadas e caminhos propostos em §5 do relatório.

---

## Projeto base

Este repositório é fork de [`fbarth/gym_custom_env`](https://github.com/fbarth/gym_custom_env), que provê ambientes Gymnasium customizados (GridWorld v0, 3D, com obstáculos, e o ambiente CPP usado nesta APS). Os scripts dos ambientes anteriores (`run_grid_world_v0.py`, `train_grid_world_obstacles.py`, `train_grid_world_3D.py`, etc.) continuam funcionais e foram preservados.
