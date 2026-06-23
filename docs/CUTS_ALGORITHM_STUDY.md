# CUTS_ALGORITHM_STUDY — usar cada cut com sentido e precisão

Estudo (NÃO implementação) de como reformular a escolha de camera/directed cuts para
que **cada cut tenha um motivo musical explícito e caia no instante certo**, em vez da
rotação de pools atual. Ler com `docs/CUTS_REFERENCE.md` ao lado.

---

## 1. Diagnóstico do sistema atual (`build_camera`)

Modelo = **rotação de pools enviesada por energia**:
- `SECTION_CAMERA[kind]` → pool de framings `coop_*` (a "cama", ~87%).
- `_SECTION_DIRECTED[kind]` = (pool calma, pool energética) → 1 directed injetado (~13%).
- Seleção = ciclar a pool, evitar repetição (`recent_coop`/`recent_directed`), `_guard_directed`.
- Injetores especiais: `dircut_at_start` (subida p/ high), full-band nas entradas de impacto,
  pools de solo, `D_BRE_Jump` no fim do BRE.
- Precisão = `_snap_to_music` (acento estrutural ±1 beat, senão beat mais próximo).

**Forças (manter):** bate as estatísticas oficiais (87/13, ~14 directed distintos/música,
top-cut ~11–20%); guards sólidos (presença, `_NP`, duos, sing-along precisa de voz real);
anti-recência; filtro de instrumentos ausentes.

**Limitação central (o que o utilizador quer resolver):** um cut entra porque *"o ciclo
calhou nele"*, não porque *aconteceu algo na música ali*. A precisão é um snap genérico,
não o hit real (pontapé, nota final do BRE, pico da frase vocal, kick).

**Auditoria de cobertura (cuts mortos hoje):**
- Nunca emitidos: `D_BRE` (só sai `D_BRE_Jump`), `D_Gtr_Cam_PR`, `D_Vox_Cam_PR`,
  `D_Stagedive`, `D_Crowdsurf`.
- Só reativos (via despromoção do guard, nunca por mérito): `D_*_NP` (Gtr/Bass/Drums/Keys/Vox).

---

## 2. Modelo proposto: **camada de eventos (triggers) sobre a cama de framing**

Manter a **cama de framing** `coop_*` como filler do pacing 2–4 s (está correta e é o que
fazem os oficiais). Substituir a *seleção de directed* por um **motor de eventos**: os
directed cuts passam a ser **respostas a eventos musicais detetados**, cada tipo de evento
"dono" de um pequeno conjunto de cuts apropriados, colocados no **tick exato do hit**.

```
Layer A  (cama)      framings coop_* paced  ──┐
                                              ├──►  merge  ──►  VENUE track
Layer B  (eventos)   directed nos hits reais ─┘   (B sobrepõe-se ao filler perto,
                                                   respeitando spacing + budgets)
```

### 2.1 Sinais disponíveis (vocabulário de input)
Já temos: onsets por instrumento (`inst_onsets`), `_vocal_real`, acentos estruturais,
**sub-spans de energia com tier** (calm/mid/high já com heaviness-gate), `kind`/`name` da
secção, `bre_spans`, secções de solo.

**A derivar (helpers novos, puros, testáveis):**
- `downtime_spans(inst)` — pausas ≥ N compassos de um instrumento (para `_NP`/crowd).
- `vocal_phrases()` — agrupar `_vocal_real` em frases; pico/última nota de cada (para vox).
- `featured_instrument(win)` — quem carrega a música numa janela (parte mais densa/única).
- `rise_boundaries()` — fronteiras onde a energia sobe (climaxes de build; `dircut_at_start`).
- `kick_hits()` — onsets de bombo (para `D_Drums_KD` no kick certo).

### 2.2 Tabela "evento musical → cut" (o coração: sentido + precisão)

| # | Evento detetado | Cut(s) candidatos | Hit (tick exato) | Estado hoje |
|---|---|---|---|---|
| 1 | Entrada de secção com **subida** de energia p/ high | D_All_Yeah (heavy+voz) / D_All_LT / D_All_Cam | downbeat da entrada | existe |
| 2 | **Última nota do BRE** | D_BRE_Jump (banda) ⇄ D_BRE (smash da guitarra) p/ variar | onset final do BRE | só Jump |
| 3 | **Solo** do instrumento X | D_X_Cam_PR (pre-roll a entrar no solo) → D_X_CLS (técnico) → D_X / D_X_Cam_PT (floreio) | entrada do solo + acentos internos | parcial (sem Cam_PR) |
| 4 | **Downtime** de X (≥2 compassos parado) | D_X_NP (idle) / D_Crowd_Gtr / D_Crowd_Bass / D_Drums_Point (se a voz carrega) | meio do downtime | só reativo |
| 5 | **Pico de frase vocal** (nota alta/sustida) | D_Vox_CLS (dramático) / D_Vox_Cam_PT (excitante) | pico/última nota da frase | em pool, não ancorado |
| 6 | **Dois instrumentos em interação** (riff em unísono / call-response) | D_Duo_GB/KB/KG (instr) · D_Duo_Gtr/Bass/KV (c/ voz) · D_Duo_Drums (baterista canta) | acento partilhado | em pool, não ancorado |
| 7 | **Refrão sing-along** (voz real + high) | D_Crowd / D_All_Yeah / D_Crowd_Bass/Gtr | downbeat/hook do refrão | existe |
| 8 | **Gap vocal longo** (≥16 compassos sem voz, com retorno depois) | D_Stagedive / D_Crowdsurf + **cut-away forçado** a seguir | início do gap | **morto hoje** |
| 9 | **Build/breakdown com bombo** | D_Drums_KD no kick · D_Drums_Point | onset do kick | não ancorado |
| 10| **Passagem rápida/técnica** (onset-gap pequeno) | D_Gtr_CLS / D_Bass_CLS (braço) | acento da passagem | existe |
| 11| **Entrada de impacto** (intro/chorus/drop/breakdown/outro) | D_All_Cam / D_All_LT (pan) | downbeat da entrada | existe |

Framing com precisão também (Layer A): o single-near/closeup segue o **instrumento em
destaque** na janela (quem toca de facto), não só rotação cega — mantém variedade mas
aponta a câmara a quem carrega a música (eventos #3/#4/#6 informam isto).

### 2.3 Algoritmo de seleção / merge
1. Pré-computar **lista de eventos** ordenada por tick: `(tick, cut, prioridade, dramatic?)`.
2. Pré-computar **slots de framing** (pacing como hoje).
3. Percorrer a timeline; em cada slot, se houver evento dentro da janela → emitir o cut do
   evento (mais específico/maior prioridade que passe os guards, sem violar min-gap /
   throttle de dramatic / budget); senão → filler de framing.
4. **Budgets/guardrails** (manter as estatísticas oficiais como invariantes):
   - directed ≤ ~13 % do total; full-band dramatic ≤ ~3–4/música; stagedive/crowdsurf ≤ 1;
   - anti-recência nas duas camadas; throttle dramático ≥ 8 beats.
5. Guards atuais ficam (presença, `_NP`, duos, sing real-voz) — agora quase sempre já
   satisfeitos por construção, porque os eventos saem dos mesmos sinais.

---

## 3. Porque é melhor (sentido + precisão)
- Cada cut tem uma **razão explícita** para existir naquele tick (o evento), não "o ciclo calhou".
- **Precisão**: o hit cai no momento musical real (kick, pico de frase, nota final do BRE,
  entrada do solo, downbeat da subida) em vez do snap genérico.
- **Cobertura**: dá casa aos 10 cuts mortos (Cam_PR, `_NP` proativo, Stagedive, Crowdsurf, D_BRE)
  pelos seus triggers próprios — sem os "forçar".
- **Sem regressão estatística**: a cama de framing fica igual e o directed continua com budget.

---

## 4. Riscos / mitigação
- Mais extração de sinal (downtime, frases vocais, kick, featured) → helpers puros + um
  diagnóstico dev que imprime a timeline de eventos por música (estilo `dev/subspan_moods.py`).
- Não regredir o 87/13 + variedade → manter como **testes/guardrails** comparados aos 20 oficiais.
- Stagedive/crowdsurf são frágeis (regra das 16 compassos) → só na janela #8 com cut-away forçado.

---

## 5. Implementação faseada (proposta — cada fase validada antes da seguinte)
- **Fase 0** — helpers de deteção de eventos + diagnóstico dev (imprime a timeline de eventos).
  **Zero** mudança de comportamento. Validar que os eventos fazem sentido em Broken Mirror/Elegy/etc.
- **Fase 1** — passar os triggers que **já funcionam** (BRE, dircut_at_start, solo, full-band)
  pelo motor novo: refactor equivalente (mesma saída), para assentar a arquitetura.
- **Fase 2** — adicionar os eventos novos um a um (downtime→`_NP`/crowd; frase vocal→vox;
  interação→duo; kick→KD), validando estatísticas a cada passo.
- **Fase 3** — ligar Stagedive/Crowdsurf na janela de gap vocal longo, com cut-away forçado.
- **Fase 4** — enviesar o framing para o instrumento em destaque (precisão na cama).

> Cada fase é um commit isolado e reversível. Em qualquer ponto comparamos com os 20 venues
> oficiais (split, distintos/música, top-cut share) para garantir que não regredimos o "feel".

---

## 6. Decisões (alinhadas com o utilizador)
- **Abordagem:** motor de eventos completo, pelas 5 fases.
- **Stagedive/Crowdsurf:** ATIVAR, só na janela de gap vocal (≥16 compassos sem voz +
  retorno), com cut-away forçado a seguir; ≤1×/música.
- **Validação:** diagnóstico dev (timeline de eventos por música) + stats; os 20 oficiais
  servem de **referência**, não de portão rígido.

### Estado de implementação
- [x] Fase 0 — helpers de deteção + diagnóstico dev (`downcharter/cut_events.py` +
      `dev/cut_events_timeline.py`). Zero mudança de comportamento. Validado em
      Broken Mirror (20 eventos) e Elegy (21). Aprendizagens: excluir a voz do downtime
      (descanso entre frases é normal); BRE vem só da track de guitarra.
- [x] Fase 1 — `build_camera` reescrita em 2 passos: (1) cama de framing `coop_*`,
      (2) overlay de directed a partir de `detect_events` nos ticks de hit, com merge
      (directed ganha ao filler perto), throttle de full-band (≥32 beats), 1 stagedive/
      música, anti-recência e guards. Stats BM 95/4→**90/9**, top-cut 18%→12%, cuts
      antes mortos (`D_Drums_NP`) vivos. Harness `dev/cut_stats.py`. Pools antigas
      (`_SECTION_DIRECTED`/`_ALLBAND_*`/`_SOLO_DIRECTED`/`_fresh_directed`) ficaram
      mortas — remover na limpeza final.
- [x] Fase 2 — `detect_features` (rotação de duos/closeups/vocal por secção → nenhum
      directed domina, máx 2-3/tipo); vocal_peak robustecido e ativo (D_Vox_CLS volta);
      `detect_technical` desligado (duplicava CLS). Falta: kick→KD preciso (sem kick
      isolado hoje); subir directed share ~6-7%→~13% se quisermos mais presença.
- [ ] Fase 3 — Stagedive/Crowdsurf
- [ ] Fase 4 — framing segue o instrumento em destaque
</content>
