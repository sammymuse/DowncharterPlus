# CUTS_REFERENCE — Camera cuts & Directed cuts (Customs Book pp. 683–724)

Lista completa de **cada** camera cut standard (`[coop_*]`) e directed cut
(`[directed_*]`) com a descrição do que faz em jogo, o text event correspondente,
e notas de prioridade/timing. Fonte: Customs Book v1, cap. 20 "Venue Authoring".

Legenda:
- **`*` = predictable / "angle-only"**: cuts com início e fim previsíveis (sem pre/post-roll
  imprevisível) — bons para sincronizar. Os marcados como *angle-only* **não têm animação
  única**, só um ângulo de câmara único (D_All_LT, D_Drums_LT, D_Bass_CLS, D_Gtr_CLS,
  D_Drums_KD, D_Crowd).
- Posições de câmara: **Behind** (atrás) · **Far** (longe) · **Near** (perto) ·
  **Closeup** (Hand = mãos/braço, Head = cara, Closeup = vocalista).

---

## 1. CAMERA CUTS STANDARD (`[coop_*]`)

Empilháveis ("stacked"). A câmara escolhe o cut que melhor corresponde aos membros
presentes no palco. Listados **do mais genérico para o mais específico** (= ordem de
prioridade; o mais específico ganha).

### 1.1 Four character (mais genérico — fallback)
> Se nenhum outro cut faz match, usa-se um destes. Também pode entrar um shot de 3.

| Cut | Text event | Descrição |
|---|---|---|
| All_Behind | `[coop_all_behind]` | Banda toda, vista de trás |
| All_Far | `[coop_all_far]` | Banda toda, plano afastado |
| All_Near | `[coop_all_near]` | Banda toda, plano aproximado |

### 1.2 Three characters (sem drums) — baixa prioridade, muito geral
| Cut | Text event | Descrição |
|---|---|---|
| Front_Behind | `[coop_front_behind]` | Guitarra+baixo+(teclas)+voz, de trás |
| Front_Near | `[coop_front_near]` | Idem, plano aproximado |

### 1.3 One character — standard
> Drums e vocals têm prioridade **mais baixa** (estão sempre presentes → genéricos).

| Cut | Text event | Cut | Text event |
|---|---|---|---|
| D_Behind | `[coop_d_behind]` | D_Near | `[coop_d_near]` |
| V_Behind | `[coop_v_behind]` | V_Near | `[coop_v_near]` |
| B_Behind | `[coop_b_behind]` | B_Near | `[coop_b_near]` |
| G_Behind | `[coop_g_behind]` | G_Near | `[coop_g_near]` |
| K_Behind | `[coop_k_behind]` | K_Near | `[coop_k_near]` |

(D=drums, V=vocals, B=bass, G=guitar, K=keys)

### 1.4 One character — closeup
| Cut | Text event | Descrição |
|---|---|---|
| D_Hand | `[coop_d_closeup_hand]` | Mãos/braços do baterista |
| D_Head | `[coop_d_closeup_head]` | Cara do baterista |
| V_Closeup | `[coop_v_closeup]` | Close-up do vocalista |
| B_Hand | `[coop_b_closeup_hand]` | Mãos do baixista |
| B_Head | `[coop_b_closeup_head]` | Cara do baixista |
| G_Hand | `[coop_g_closeup_hand]` | Mãos do guitarrista |
| G_Head | `[coop_g_closeup_head]` | Cara do guitarrista |
| K_Hand | `[coop_k_closeup_hand]` | Mãos do tecladista |
| K_Head | `[coop_k_closeup_head]` | Cara do tecladista |

> Single keys shot tem prioridade **acima** dos shots de 2 que incluem teclas.

### 1.5 Two character (mais específico dos standard)
> Combinações de 2 ganham aos single-char. Drums e vocals continuam baixa prioridade.
> **BK_Near tem prioridade > GV_Near** (exemplo do livro).

| Cut | Text event | Cut | Text event |
|---|---|---|---|
| DV_Near | `[coop_dv_near]` | BD_Near | `[coop_bd_near]` |
| DG_Near | `[coop_dg_near]` | BV_Behind | `[coop_bv_behind]` |
| BV_Near | `[coop_bv_near]` | GV_Behind | `[coop_gv_behind]` |
| GV_Near | `[coop_gv_near]` | KV_Behind | `[coop_kv_behind]` |
| KV_Near | `[coop_kv_near]` | BG_Behind | `[coop_bg_behind]` |
| BG_Near | `[coop_bg_near]` | BK_Behind | `[coop_bk_behind]` |
| BK_Near | `[coop_bk_near]` | GK_Behind | `[coop_gk_behind]` |
| GK_Near | `[coop_gk_near]` | | |

### 1.6 RANDOM
Nota RANDOM (topo da LIGHTING track) → gera um cut **standard** aleatório (nunca
directed). Nunca repete o cut anterior/seguinte. Em stacking com outros cuts, escolhe
um random de prioridade inferior ao outro stacked.

---

## 2. DIRECTED CUTS (`[directed_*]`)

Câmara dirigida com animação OU ângulo único. Sempre **mais específicos** que standard.
Têm **hit** (instante da ação), pre-roll e post-roll variáveis — coloca o text event no
hit. Em themes o hit fica no início da practice section.

### 2.1 Full band / crowd (dramáticos — usar com moderação)
| Cut | Text event | Descrição |
|---|---|---|
| D_All | `[directed_all]` | Banda interage: saltam, dão pontapé à câmara/público, caem de joelhos. Bom para partes excitantes com sing-along. |
| D_All_Cam | `[directed_all_cam]` | Mais longo. A banda interage com a câmara e dá tudo em conjunto. |
| D_All_LT* | `[directed_all_lt]` | Plano longo e panorâmico da banda toda; mais dinâmico que All_Near/All_Far. **Predictable / angle-only.** |
| D_All_Yeah | `[directed_all_yeah]` | Muito dramático em slow-mo; vocalista aponta ao ar, pan pela banda em câmara lenta. Em arenas o guitarrista salta e desliza de joelhos (NÃO acontece se houver tecladista). Difícil de cronometrar; excelente com BONUSFX em glam/grandes bandas. |
| D_BRE | `[directed_bre]` | Big Rock Ending: guitarrista atira a guitarra ao chão, salta sobre ela e esmaga-a. Colocar na última nota do BRE. Hit = guitarrista a cair para trás. |
| D_BRE_Jump | `[directed_brej]` | Outro cut de BRE: personagens saltam/pisam/martelam a bateria na última nota; às vezes a banda atira-se ao chão antes do hit final. Usado na maioria dos BREs. |
| D_Crowd* | `[directed_crowd]` | Plano amplo do palco que inclui o público; em arenas, plano individual com membros do público. **Predictable / angle-only.** Existe mesmo em músicas sem BRE. |

### 2.2 Single character — a tocar / em ação
| Cut | Text event | Descrição |
|---|---|---|
| D_Drums | `[directed_drums]` | Baterista bate nos pratos; às vezes rodopia as baquetas primeiro. |
| D_Drums_LT* | `[directed_drums_lt]` | Plano dinâmico e panorâmico do baterista; solos/partes interessantes. **Predictable / angle-only.** |
| D_Drums_KD* | `[directed_drums_kd]` | Close-up do pedal de bombo. **Predictable / angle-only.** |
| D_Drums_Point | `[directed_drums_pnt]` | Baterista aponta à câmara com as baquetas. (Por vezes não está a tocar — usar com cuidado.) |
| D_Bass | `[directed_bass]` | Baixista em ação, como D_Gtr mas menos teatral; não desliza de joelhos. |
| D_Bass_Cam | `[directed_bass_cam]` | Baixista mostra-se para a câmara, mais relaxado que o guitarrista; bate na câmara com o instrumento; muitas animações de género. **Não tem _pr/_pt.** Usável em qualquer ponto (não ligado a hit). |
| D_Bass_CLS* | `[directed_bass_cls]` | Close-up do braço do baixo. Partes técnicas/solos. **Predictable / angle-only.** |
| D_Gtr | `[directed_guitar]` | Guitarrista, grande variedade: desliza de joelhos, pontapé ao ar/por cima da câmara, bate na câmara com a guitarra. |
| D_Gtr_Cam_PR | `[directed_guitar_cam_pr]` | Guitarrista exibe-se / interage com o público; **pre-roll longo**. Solos/partes técnicas. |
| D_Gtr_Cam_PT | `[directed_guitar_cam_pt]` | Como _PR mas **post-roll longo, pre-roll curto**; mais animações caem no hit. |
| D_Gtr_CLS* | `[directed_guitar_cls]` | Close-up do braço da guitarra. Partes técnicas/solos. **Predictable / angle-only.** |
| D_Keys | `[directed_keys]` | Tecladista salta ao ar ou bate com as mãos no teclado. |
| D_Keys_Cam | `[directed_keys_cam]` | Tecladista dá tudo e dança. |
| D_Vocals | `[directed_vocals]` | Vocalista, grande variedade: aponta ao ar/ao público, pontapé ao ar, pontapé à câmara; animações de género. |
| D_Vox_CLS | `[directed_vocals_cls]` | Close-up dramático: segura o micro com as duas mãos, abana a cabeça, cai ao chão/joelhos. Notas altas/gritadas. |
| D_Vox_Cam_PR | `[directed_vocals_cam_pr]` | Como _PT mas **pre-roll longo, post-roll curto**; mais animações caem no hit. Momentos vocais excitantes. |
| D_Vox_Cam_PT | `[directed_vocals_cam_pt]` | Grande gama de animações para momentos vocais excitantes; interage com público/câmara, "rock the mic". |

### 2.3 Single character — NÃO a tocar (usar em estado `[idle]`)
| Cut | Text event | Descrição |
|---|---|---|
| D_Drums_NP | `[directed_drums_np]` | Baterista em idle: rodopia/truques com baquetas, gesticula à câmara. |
| D_Bass_NP | `[directed_bass_np]` | Baixista em idle (semelhante a D_Gtr_NP). |
| D_Gtr_NP | `[directed_guitar_np]` | Guitarrista em idle: pontapé ao ar (não à câmara), tira a mão da guitarra; muitas de género. |
| D_Keys_NP | `[directed_keys_np]` | Tecladista balança para trás e para a frente (pouquíssimas animações). |
| D_Vox_NP | `[directed_vocals_np]` | Vocalista em idle: pontapé ao ar, salta, segura o micro mas não à cara. |

### 2.4 Vocalist stage dive / crowd surf (CUIDADO)
| Cut | Text event | Descrição |
|---|---|---|
| D_Stagedive | `[directed_stagedive]` | Vocalista corre para fora do palco e salta para o público. Cortar **assim que** ele salta. Sem lip-sync (boca não mexe). |
| D_Crowdsurf | `[directed_crowdsurf]` | Como stagedive mas o vocalista faz crowd-surf; às vezes só crowd-surf sem saltar. **Cortar para outro plano a seguir** e deixar **≥16 compassos** antes de o vocalista voltar a cantar/tocar tamborim/cowbell. |

### 2.5 Two character / sing-along
> Usar com as notas de sing-along da VENUE track (notas 85/86/87 → lip-sync).

| Cut | Text event | Descrição |
|---|---|---|
| D_Duo_Drums | `[directed_duo_drums]` | Baterista vira-se/interage com a câmara. **Só quando o baterista canta** (fica estranho se não cantar). Não inclui outros membros. |
| D_Duo_Gtr | `[directed_duo_guitar]` | Guitarrista e vocalista interagem: jam juntos, encostam-se, guitarrista canta para o micro do vocalista. |
| D_Duo_Bass | `[directed_duo_bass]` | Igual a D_Duo_Gtr mas baixista + vocalista. |
| D_Duo_GB | `[directed_duo_gb]` | Guitarrista e baixista a tocar juntos. Bom para momentos envolventes. |
| D_Duo_KB | `[directed_duo_kb]` | Baixista e tecladista a dar tudo juntos. |
| D_Duo_KG | `[directed_duo_kg]` | Guitarrista e tecladista a dar tudo juntos. |
| D_Duo_KV | `[directed_duo_kv]` | Vocalista e tecladista a dar tudo juntos. |
| D_Crowd_Gtr | `[directed_crowd_g]` | Guitarrista interage com o público (high-fives, exibe-se). |
| D_Crowd_Bass | `[directed_crowd_b]` | Igual a D_Crowd_Gtr mas para o baixista. |

---

## 3. Prioridade dos directed cuts (genérico → específico)

Do menos para o mais específico (o mais específico ganha quando empilhados):

1. **Full band/crowd:** D_All → D_All_Cam → D_All_Yeah → D_All_LT* → D_BRE → D_BRE_Jump → D_Crowd*
2. **Single (drums/vox primeiro, mais genéricos):** D_Drums → D_Drums_Point → D_Drums_NP → D_Drums_LT* → D_Drums_KD* → D_Vocals → D_Vox_NP → D_Vox_CLS → D_Vox_Cam_PR → D_Vox_Cam_PT → D_Stagedive → D_Crowdsurf
3. **Single (bass/gtr/keys, mais específicos):** D_Bass → D_Crowd_Bass → D_Bass_NP → D_Bass_Cam → D_Bass_CLS* → D_Gtr → D_Crowd_Gtr → D_Gtr_NP → D_Gtr_CLS* → D_Gtr_Cam_PR → D_Gtr_Cam_PT → D_Keys → D_Keys_Cam → D_Keys_NP
4. **Two character (mais específicos de todos):** D_Duo_Drums → D_Duo_Gtr → D_Duo_Bass → D_Duo_KV → D_Duo_GB → D_Duo_KB → D_Duo_KG

---

## 4. Notas de timing / bom gosto (relevante para o gerador)

- **Pacing:** um cut novo a cada **2–4 s** (deixa estabelecer o shot; >4 s e a câmara "foge"
  do foco). Refletir o feel: ~1.5 s/cut em punk rápido, até ~12 s em baladas.
- **≥ 1/12 dos cuts** devem ser directed (já vêm no pool dos random).
- **2–3 special moments por minuto**; encher à volta com filler shots.
- Sincronizar cuts/strobe com a bateria e os acentos.
- Full-band com **moderação** — abusar tira o impacto.
- `_LT`/`_CLS`/`_KD`/`D_Crowd` (predictable) são os melhores para sincronizar com precisão.
</content>
</invoke>
