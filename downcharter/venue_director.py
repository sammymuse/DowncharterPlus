"""venue_director.py — o "director pass" da venue: decisões song-level tomadas
UMA vez por música e partilhadas por todos os sistemas (lighting/pp/câmara/pyro),
para que o resultado pareça intencional (authored) e não sorteado.

Calibrado contra os 100 charts oficiais (dev/venue_repetition_study.py,
dev/venue_sync_study.py, dev/venue_arc_study.py):

* REPETIÇÃO — secções repetidas reutilizam o look: lighting Jaccard same-group
  0.53 vs 0.18 cross (2.9×); 1.º preset igual 57% vs 13% (4.3×). PP idem (2.9×)
  e com look dominante real (top-2 filtros = 56% dos eventos da música).
  Câmara NÃO repete sequências (1.3×) — só herda o instrumento em destaque.
* SINCRONIZAÇÃO — nas fronteiras de secção, luz/pp/câmara mudam NO downbeat
  (offset mediano 0.00 beats); luz+câmara juntas dentro de ¼ beat em 44% das
  fronteiras (25× acima do acaso), as três em 34% → âncora partilhada.
* ARCO — o último chorus tem 1.57× lighting e 1.82× pp do primeiro (CI 95%
  significativos); pyro NÃO sobe (0.75×) e fullband DESCE (0.47×) — o clímax
  oficial é luz/pp, não pirotecnia. `directed_all` cai no 1.º chorus em 32%
  dos oficiais; `directed_all_cam` é de fim de música (posição mediana 0.75).

O RNG é seeded do CONTEÚDO da música → a mesma música gera sempre a mesma
venue (reproduzível); mudar a música muda a seed.
"""
from __future__ import annotations

import random
import re
import zlib
from dataclasses import dataclass, field

# ── Constantes calibradas (dev/arc_stats.json) ────────────────────────────────
CLIMAX_LIGHT_FACTOR = 1.57   # densidade de lighting no último chorus vs 1.º
CLIMAX_PP_FACTOR = 1.82      # densidade de pp no último chorus vs 1.º

# Alvos de sincronização (dev/sync_stats.json) — usados pela validação, e como
# referência: a âncora partilhada coloca o 1.º evento de cada sistema no downbeat.
SYNC_TARGET_QUARTER_BEAT = 0.44   # luz+cam dentro de ¼ beat da fronteira
SYNC_TARGET_TRIPLE = 0.34         # luz+cam+pp dentro de ¼ beat

# Sufixo de repetição nos nomes de secção: "verse_1", "chorus 2a" → base.
_REPEAT_SUFFIX = re.compile(r"[\s_]*\d+[a-z]?$")


def _norm_name(name: str) -> str:
    """Nome normalizado de secção (agrupa repetições: verse_1/verse_2 → verse)."""
    return _REPEAT_SUFFIX.sub("", name.strip().lower()).strip(" _") or "default"


@dataclass
class SectionGroup:
    """Grupo de secções repetidas (mesmo nome normalizado) que partilham o look."""
    key: str
    section_idxs: list[int] = field(default_factory=list)
    light_motif: list[str] | None = None   # gravado pelo build_lighting na 1.ª ocorrência
    pp_hold: str | None = None             # filtro de hold do grupo (o "look" pp)
    featured_inst: str | None = None       # instrumento em destaque (bias de câmara)


@dataclass
class VenueDesign:
    """Decisões song-level, tomadas uma vez em plan_venue e lidas pelos build_*."""
    rng: random.Random
    groups: dict[str, SectionGroup]
    group_of: list[str]                    # idx da secção → key do grupo
    anchors: list[int]                     # 1 tick por secção, snapped a downbeat
    first_chorus_idx: int | None = None
    climax_idx: int | None = None          # última ocorrência do chorus repetido
    intensity: list[float] = field(default_factory=list)  # multiplicador por secção
    dominant_pp: list[str] = field(default_factory=list)  # look pp da música

    def group_at(self, idx: int) -> SectionGroup:
        return self.groups[self.group_of[idx]]

    def is_first_occurrence(self, idx: int) -> bool:
        return self.group_at(idx).section_idxs[0] == idx

    def occurrence_of(self, idx: int) -> int:
        """0-based: quantas ocorrências do grupo vieram antes desta secção."""
        return self.group_at(idx).section_idxs.index(idx)


def _content_seed(sections: list, onsets: list[int], song_end: int,
                  tpb: int) -> int:
    """Seed derivada do conteúdo — estável entre execuções, muda com a música."""
    key = (song_end, tpb, tuple(onsets[:64]),
           tuple((s.start, s.name) for s in sections))
    return zlib.crc32(repr(key).encode("utf-8"))


def plan_venue(sections: list, onsets: list[int],
               inst_onsets: dict[str, list[int]] | None,
               time_sig_map: list, tpb: int, song_end: int) -> VenueDesign:
    """Constrói o VenueDesign: grupos de repetição, âncoras de fronteira,
    arco (1.º chorus / clímax) e RNG seeded. Sem secções → design mínimo no-op."""
    # imports tardios: venue importa venue_director no topo; ao tempo de CHAMADA
    # o módulo venue já está completo (evita o ciclo no load)
    from .venue import _first_downbeat_at_or_after, section_energy

    rng = random.Random(_content_seed(sections, onsets, song_end, tpb))

    # (a) grupos por nome normalizado — espelha as practice sections oficiais
    groups: dict[str, SectionGroup] = {}
    group_of: list[str] = []
    for i, s in enumerate(sections):
        key = _norm_name(s.name)
        g = groups.setdefault(key, SectionGroup(key))
        g.section_idxs.append(i)
        group_of.append(key)

    # (b) âncoras de fronteira: downbeat ≥ s.start, partilhado por luz/pp/câmara
    # (oficiais: offset mediano à fronteira = 0.00 beats nas 3 categorias)
    anchors = [
        _first_downbeat_at_or_after(s.start, time_sig_map, tpb)[0]
        for s in sections
    ]

    # (c) arco: grupo de chorus repetido → 1.ª ocorrência = entrada do arco,
    # última = clímax. Fallback: última secção 'high' depois de 60% da música.
    first_chorus_idx: int | None = None
    climax_idx: int | None = None
    chorus_groups = [
        g for g in groups.values()
        if len(g.section_idxs) >= 2
        and sections[g.section_idxs[0]].kind == "chorus"
    ]
    if chorus_groups:
        main = max(chorus_groups, key=lambda g: len(g.section_idxs))
        first_chorus_idx = main.section_idxs[0]
        climax_idx = main.section_idxs[-1]
    else:
        late_high = [i for i, s in enumerate(sections)
                     if s.start >= 0.6 * song_end
                     and section_energy(s) == "high"]
        if late_high:
            climax_idx = late_high[-1]

    # (d) instrumento em destaque por grupo (share relativa ao total do
    # instrumento na música, agregada sobre TODAS as ocorrências do grupo)
    if inst_onsets:
        import bisect
        totals = {inst: max(1, len(ons)) for inst, ons in inst_onsets.items()
                  if not inst.startswith("_")}
        for g in groups.values():
            best, best_share = None, 0.0
            for inst, tot in totals.items():
                ons = inst_onsets[inst]          # já ordenados pelo caller
                cnt = sum(
                    bisect.bisect_left(ons, sections[i].end)
                    - bisect.bisect_left(ons, sections[i].start)
                    for i in g.section_idxs
                )
                share = cnt / tot
                if share > best_share:
                    best, best_share = inst, share
            g.featured_inst = best

    intensity = [1.0] * len(sections)
    if climax_idx is not None:
        intensity[climax_idx] = CLIMAX_LIGHT_FACTOR

    return VenueDesign(
        rng=rng, groups=groups, group_of=group_of, anchors=anchors,
        first_chorus_idx=first_chorus_idx, climax_idx=climax_idx,
        intensity=intensity,
    )
