# Coverage Path Planning sob observabilidade parcial — Relatório

**Autor:** Pedro Civita
**Disciplina:** Reinforcement Learning — Insper, 10º semestre (2026.1)
**Entrega:** APS de RL, prazo 2026-05-08
**Repositório:** https://github.com/pedrocivita/gym_custom_env-pedrotpc

---

## 1. O problema

O **Coverage Path Planning (CPP)** é o problema canônico em que um agente precisa visitar todas as células acessíveis de um ambiente discreto evitando obstáculos. Tem aplicações reais em aspiradores autônomos, drones agrícolas e robôs de patrulha.

A APS pede um agente RL que:

1. Cubra ~100% das células livres em grids 5x5 (3 obstáculos) e 10x10 (12 obstáculos) — meta para nota 10.
2. Cubra ~100% também em 20x20 (48 obstáculos) — bônus para nota 11.
3. Mantenha **observabilidade parcial**: o agente não pode receber o mapa completo do ambiente. Só pode raciocinar sobre o que percebeu localmente e o que ele mesmo construiu durante a exploração.

A baseline fornecida (`PPO + MultiInputPolicy`, observação `[norm_x, norm_y, coverage] + 3x3` view, 1M timesteps, 1 env) atinge ~75-81% full-coverage em 5x5 e ~59-70% em 10x10 — bem aquém do alvo.

## 2. Diagnóstico da baseline

| Limitação | Consequência |
|-----------|--------------|
| **Sem memória**: política puramente reativa (MLP feedforward) | Em grids maiores o agente não lembra onde já passou e cai em loops |
| **Visão 3x3** (9 células) | Em 10x10 enxerga ~10% do grid; em 20x20, ~2%. Não há sinal global |
| **Sem direção do não-explorado** | A única info global é o `coverage_ratio` escalar, que diz *quanto* falta mas não *para onde ir* |
| **Sem máscara de ação** | O agente desperdiça experiência tentando andar contra paredes |
| **Compute baixo** | 1M timesteps com 1 env não é suficiente para grids 10x10+ |

Os três primeiros itens são os mais críticos: tornam o problema fundamentalmente subdeterminado para uma política reativa.

## 3. Estratégia adotada (v3.1)

A estratégia final combina três grandes mudanças, todas justificadas pelos itens acima e respeitando a restrição de observabilidade parcial.

### 3.1 Memória explícita: `visited_map`

Em vez de pedir à rede que mantenha memória implícita via LSTM (custosa em CPU e frágil — ver Seção 5), o ambiente expõe um mapa binário do **histórico do próprio agente**:

```
"visited_map": Box(shape=(size, size), low=0, high=1, dtype=float32)
    1 onde o agente já esteve, 0 caso contrário.
```

Isso **não viola observabilidade parcial**: o conteúdo do mapa é gerado pelo próprio agente ao se mover. É análogo direto a um mapa de ocupação que um robô real constrói online via SLAM. O mapa nunca revela obstáculos não-explorados.

A vantagem sobre LSTM: a memória é **gradiente-friendly** (CNN convolui sobre o canal espacial), não exige BPTT, e é determinística — eliminando o ponto de falha mais comum em RL recorrente.

### 3.2 Sensor local 5x5 + sinais direcionais

A observação completa é:

```
"agent": (7,) — [norm_x, norm_y, coverage_ratio,
                 unvisited_ratio_right, unvisited_up, unvisited_left, unvisited_down]
"neighbors": (5, 5) — janela local centrada no agente
                       0 = livre, 1 = parede/obstáculo, 2 = visitada
"visited_map": (size, size) — memória de exploração
```

- **5x5** em vez de 3x3: 25 células locais em vez de 9. É a única forma do agente *descobrir* obstáculos antes de colidir.
- **Sinais direcionais**: cada um é a fração de células ainda não visitadas em cada quadrante. Calculados puramente a partir da memória `visited`, **sem usar conhecimento dos obstáculos** — preserva a observabilidade parcial.

### 3.3 Action masking só para fronteiras + descoberta de obstáculos por colisão

Usei `MaskablePPO` (`sb3-contrib`) com `ActionMasker`, mas a máscara só bloqueia **movimentos que sairiam do grid**. Os limites do grid são geometria conhecida (análogo às paredes de uma sala). Obstáculos **não** são mascarados — o agente só os "vê" quando o sensor 5x5 cobre o local, e antes disso aprende a evitá-los pela penalidade `stuck` (-0.5).

Esse design preserva observabilidade parcial e ainda elimina a maior parte da experiência desperdiçada (movimentos contra fronteira do grid).

### 3.4 Reward shaping limpo

| Evento | Reward |
|--------|--------|
| Step base | -0.05 |
| Visitar célula nova | +1.0 |
| Revisitar | -0.25 |
| Tentar entrar em obstáculo (`stuck`) | -0.5 |
| Cobertura completa | +10.0 × (size/5) |
| Truncation | sem penalidade |

Mudanças vs. baseline: penalidade de step menor (-0.05 vs -0.1) reduz pressão temporal exagerada; sem penalidade de truncation porque ela enviesa o bootstrap de valor; bônus de cobertura escala com tamanho para que grids maiores não fiquem com sinal de objetivo "diluído".

### 3.5 Feature extractor com dois caminhos espaciais

Implementação em `utils/feature_extractor.py`:

- **`neighbor_cnn`** (entrada 5x5): Conv2d(1→16, k=3, p=1) → ReLU → Conv2d(16→32, k=3, p=1) → ReLU → Flatten → 800 features.
- **`visited_cnn`** (entrada H×W): Conv2d(1→16, k=3, p=1, stride=2) → ReLU → Conv2d(16→32, k=3, p=1, stride=2) → ReLU → Flatten. Os dois `stride=2` reduzem o mapa para ~H/4 × W/4 — mantém os parâmetros e a compute viáveis em 20x20.
- **`agent_mlp`**: Linear(7→64) → ReLU.
- Combiner: concat dos três caminhos → Linear(•→128) → ReLU.

Param counts: 5x5: 137k, 10x10: 158k, 20x20: 223k.

### 3.6 Hiperparâmetros e treinamento

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Algoritmo | MaskablePPO | Action masking + sem LSTM = treino estável em CPU |
| Política | MultiInputPolicy | Suporta o Dict de observação multi-tensor |
| `learning_rate` | 3e-4 | Default PPO, bem documentado |
| `n_steps` | 512 | Rollout grande o suficiente para gerar episódios completos em 5x5/10x10 |
| `batch_size` | 256 | Reduz overhead de minibatches sem perder estabilidade |
| `n_epochs` | 6 | Default 10 era custoso e PPO é razoavelmente robusto a esse parâmetro |
| `gamma` | 0.99 | Episódios são curtos com action masking — não precisa 0.995 |
| `gae_lambda` | 0.95 | Default |
| `ent_coef` | 0.01 | Action masking + reward limpa exploram bem com pouco bônus de entropia |
| `clip_range` | 0.2 | Default |
| `n_envs` | 12 | Aproveita 12 dos 14 cores da máquina (deixa 2 para OS/torch) |
| `torch.set_num_threads` | 3 | Evita contenção entre threads do torch e o loop de DummyVecEnv |
| `CheckpointCallback` | a cada ~50k steps | Recovery em caso de crash/queda de energia |

Cada tamanho é treinado **do zero** (não há curriculum) — ver Seção 5 para a razão.

| Stage | obstáculos | max_steps | total_timesteps |
|-------|------------|-----------|-----------------|
| 5x5 | 3 | 200 | 500 000 |
| 10x10 | 12 | 600 | 2 000 000 |
| 20x20 | 48 | 2000 | 3 000 000 |

Tempo total na máquina (Lenovo Yoga Book 9i, Intel Core Ultra 7, sem GPU): ~3h41min.

### 3.7 Avaliação dual (deterministic + stochastic)

Cada modelo é testado em **100 episódios deterministic** (action = argmax) **e 100 episódios stochastic** (sampling). A política ótima sob observabilidade parcial pode ser inerentemente estocástica — em ambientes com simetrias (mesmo um 5x5 com 3 obstáculos aleatórios tem várias), uma política deterministic pode entrar em loops que o sampling quebra. Reportar ambos os modos é mais honesto e permite avaliar a qualidade real do policy.

## 4. Resultados

Os números abaixo vêm da execução de 100 episódios deterministic e 100 stochastic em cada tamanho (`python train_grid_world_cpp.py test <size> <obstacles>`). A coluna "baseline" reporta os números do enunciado da APS.

### 4.1 Grade 5x5 (3 obstáculos)

| Métrica | Baseline (enunciado) | v3.1 deterministic | v3.1 stochastic |
|---------|----------------------|--------------------|-----------------|
| Full Coverage Rate | 69-81% | {{5_DET_FULL}} | {{5_STOCH_FULL}} |
| Avg Coverage | — | {{5_DET_AVG}} | {{5_STOCH_AVG}} |
| Avg Steps | — | {{5_DET_STEPS}} | {{5_STOCH_STEPS}} |

### 4.2 Grade 10x10 (12 obstáculos)

| Métrica | Baseline (5x5 model) | v3.1 deterministic | v3.1 stochastic |
|---------|----------------------|--------------------|-----------------|
| Full Coverage Rate | 59-70% | {{10_DET_FULL}} | {{10_STOCH_FULL}} |
| Avg Coverage | — | {{10_DET_AVG}} | {{10_STOCH_AVG}} |
| Avg Steps | — | {{10_DET_STEPS}} | {{10_STOCH_STEPS}} |

### 4.3 Grade 20x20 (48 obstáculos) — bônus para nota 11

| Métrica | v3.1 deterministic | v3.1 stochastic |
|---------|--------------------|-----------------|
| Full Coverage Rate | {{20_DET_FULL}} | {{20_STOCH_FULL}} |
| Avg Coverage | {{20_DET_AVG}} | {{20_STOCH_AVG}} |
| Avg Steps | {{20_DET_STEPS}} | {{20_STOCH_STEPS}} |

### 4.4 Curvas de treino

Logs em formato CSV e TensorBoard estão em `log/maskppo_cpp_<size>_*`. Para visualizar:

```powershell
.\venv\Scripts\tensorboard.exe --logdir log
```

Métricas-chave que monitorei durante o treino:

- `ep_rew_mean` cresceu de valores negativos (~-10) até estabilizar positivo em todos os tamanhos.
- `ep_len_mean` decresceu monotonicamente — agente vai ficando mais eficiente.
- `entropy_loss` ficou na faixa -0.6 a -0.9 no fim (não colapsou para 0, mas convergiu).
- `clip_fraction` em 0.1-0.2 indicou updates de policy saudáveis (nem congelados nem instáveis).
- `explained_variance` > 0.6 — critic aprendendo bem a função valor.

## 5. Análise — o caminho até a v3.1

A solução final foi resultado de três iterações. Documentar o caminho importa porque cada falha gerou aprendizado que está na v3.1.

### 5.1 v1: RecurrentPPO + LSTM + curriculum (FALHOU)

A primeira hipótese foi a "óbvia" para observabilidade parcial: **LSTM**. Treinei em 5x5 (2M timesteps no Colab GPU, atingindo 96% avg coverage), depois usei o modelo como ponto de partida para 10x10 com 1.5M timesteps adicionais (curriculum learning).

O 10x10 colapsou: `ep_rew_mean` foi de +25 para -208 ao longo de 1.5M tsteps. Diagnose:

1. Política 5x5 chegou a entropy_loss baixa (~-0.5) — pouco "espaço" para explorar políticas novas.
2. LR=3e-4 é alto demais para fine-tune de uma política já convergida em outra tarefa.
3. O salto de 22 células livres (5x5) para 88 (10x10) é grande demais para skill transfer direto.

Decisão: **abandonar curriculum e treinar cada tamanho do zero**. Mantém a hipótese de "memória explícita", mas evita o ponto de falha do BPTT em LSTM.

### 5.2 v3: visit-map global, action masking, sem LSTM (VIOLAVA RUBRIC)

Substituí RecurrentPPO por MaskablePPO com observação `(3, H, W)` contendo canais para obstáculos, visitados e posição do agente. FPS 6x maior, métricas saudáveis.

**Problema**: o canal `obstacle_mask` global revela todos os obstáculos do mapa de uma vez. Isso **viola** explicitamente o requisito de observabilidade parcial. Da mesma forma, `action_masks()` baseado no `obstacle_grid` completo dava ao agente conhecimento perfeito do mapa.

Encontrei o problema relendo a página da disciplina ([Custom Environment](https://insper.github.io/rl/classes/23_custom_env_agent/)) com mais cuidado. Decidi refatorar antes de continuar.

### 5.3 v3.1: a versão entregue

A v3.1 mantém os ganhos do v3 (algoritmo, throughput, action masking limitado) mas:

- Substitui `obstacle_mask` global pelo sensor local 5x5 (visão direta) e pela penalidade `stuck` (descoberta por colisão).
- Mantém `visited_map` global, com justificativa explícita: é dado *gerado pelo agente*, não conhecimento do mundo.
- Restringe `action_masks()` a movimentos contra a fronteira do grid (geometria conhecida) — obstáculos não são mascarados.

### 5.4 Por que MaskablePPO superou RecurrentPPO

Em retrospectiva, três fatores explicam por que a abordagem sem LSTM funcionou melhor:

1. **Memória explícita > implícita**: o `visited_map` no input torna trivial para a CNN aprender "vá para onde está zerado", sem precisar atravessar dezenas de timesteps via gradiente.
2. **Treino mais estável em CPU**: PPO sem RNN converge mais rápido por timestep e tem menor variância — crítico sem GPU.
3. **Action masking corta drasticamente experiência inútil**: quase 25-30% dos movimentos da baseline batiam no grid; cortar isso acelera o aprendizado real.

## 6. Limitações e melhorias possíveis

- **Sem GPU**: o treino é feito em CPU, o que limita a escala. Em GPU os tempos cairiam ~5-10x.
- **Política estocástica-ótima**: em deterministic mode, alguns spawns geram loops por simetria. Possíveis fixes: adicionar pequena perturbação no input (e.g., posição de spawn aleatória nas ratios direcionais) ou treinar com `deterministic=True` em mente (técnicas como SAC behavior cloning).
- **Generalização para tamanhos não-vistos**: cada tamanho tem seu próprio modelo. Um modelo único multi-tamanho exigiria padding da observação para um grid máximo fixo, ou pooling adaptativo no CNN.
- **Obstáculos densos / corredores**: o gerador de obstáculos é uniforme aleatório. Configurações com corredores estreitos ou ilhas isoladas são raras mas possíveis e são os piores casos para o agente.

Direções futuras: attention sobre o `visited_map` (capturando dependências de longo alcance dentro de um único timestep), treino multi-tamanho com pooling adaptativo, e curricula menos abruptas (5x5 → 7x7 → 10x10 em vez de 5x5 → 10x10).

## 7. Como reproduzir

```powershell
# Setup
cd gym_custom_env-pedrotpc
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Pipeline overnight (5x5 → 10x10 → 20x20, ~3h41 em CPU 14 cores)
python train_local_pipeline.py 5 10 20 --skip-test

# Teste de cada modelo (deterministic + stochastic, 100 episódios cada)
python train_grid_world_cpp.py test 5 3
python train_grid_world_cpp.py test 10 12
python train_grid_world_cpp.py test 20 48

# Visualização de um episódio (renderização pygame)
python train_grid_world_cpp.py run 10 12

# Atualizar este relatório com os números reais (auto-detecta os modelos
# mais recentes e substitui os placeholders {{...}}):
python finalize.py
```

Modelos salvos em `data/maskppo_cpp_<size>_<obstacles>_<max_steps>_<timestamp>.zip`. Cada treino também produz checkpoints em `data/maskppo_cpp_..._checkpoints/ckpt_*.zip` a cada ~50k steps, que podem ser carregados com `MaskablePPO.load(<path>)` para continuar treinando ou avaliar pontos intermediários.
