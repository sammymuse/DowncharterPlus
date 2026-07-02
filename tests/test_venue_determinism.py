"""Determinismo da venue: a mesma música gera sempre a MESMA venue.

O director pass (downcharter/venue_director.py) semeia o RNG com o conteúdo da
música (zlib.crc32 de secções+onsets), por isso duas chamadas a generate_venue
com o mesmo input têm de produzir eventos byte-idênticos — sem lotaria.
"""
import mido
import pytest

from downcharter.venue import generate_venue, _txt
from downcharter.venue_director import plan_venue, VenueDesign


TPB = 480


def _synthetic_song():
    """Input sintético de 48 compassos com secções repetidas (verse/chorus ×2)."""
    bar = TPB * 4
    events_track = [
        _txt(0, "[section intro]"),
        _txt(4 * bar, "[section verse_1]"),
        _txt(12 * bar, "[section chorus_1]"),
        _txt(20 * bar, "[section verse_2]"),
        _txt(28 * bar, "[section chorus_2]"),
        _txt(36 * bar, "[section outro]"),
    ]
    song_end = 48 * bar
    time_sig_map = [(0, 4, 4)]
    tempo_map = [(0, 500000)]  # 120 BPM

    # Onsets: colcheias na guitarra, semínimas no baixo, batida nos drums
    guitar = [t for t in range(0, song_end, TPB // 2)]
    bass = [t for t in range(0, song_end, TPB)]
    drums = [t for t in range(0, song_end, TPB)]
    inst_onsets = {"guitar": guitar, "bass": bass, "drums": drums}
    onsets = sorted(guitar + bass + drums)
    accents = [t for t in range(0, song_end, TPB * 2)]

    return dict(
        events_track=events_track, bre_spans=[], song_end=song_end,
        tempo_map=tempo_map, time_sig_map=time_sig_map, tpb=TPB,
        accents=accents, onsets=onsets, drum_onsets=drums,
        inst_onsets=inst_onsets,
    )


def _fingerprint(events):
    """Sequência comparável: (tick, texto | nota/velocity) de cada evento."""
    out = []
    for ev in sorted(events, key=lambda e: (e.abs_tick, str(e.msg))):
        txt = getattr(ev.msg, "text", None)
        if txt is not None:
            out.append((ev.abs_tick, txt))
        else:
            out.append((ev.abs_tick, ev.msg.type,
                        getattr(ev.msg, "note", None),
                        getattr(ev.msg, "velocity", None)))
    return out


def test_generate_venue_is_deterministic():
    kwargs = _synthetic_song()
    a = generate_venue(**kwargs)
    b = generate_venue(**kwargs)
    assert _fingerprint(a) == _fingerprint(b)


def test_seed_changes_with_content():
    kwargs = _synthetic_song()
    a = generate_venue(**kwargs)

    # Música diferente (onsets deslocados) → seed diferente → venue diferente
    moved = dict(kwargs)
    moved["onsets"] = [t + TPB // 4 for t in kwargs["onsets"]]
    b = generate_venue(**moved)
    assert _fingerprint(a) != _fingerprint(b)


def test_plan_venue_groups_and_arc():
    kwargs = _synthetic_song()
    from downcharter.venue import resolve_sections
    sections = resolve_sections(kwargs["events_track"], kwargs["song_end"],
                                kwargs["onsets"], kwargs["time_sig_map"], TPB)
    design = plan_venue(sections, kwargs["onsets"], kwargs["inst_onsets"],
                        kwargs["time_sig_map"], TPB, kwargs["song_end"])
    assert isinstance(design, VenueDesign)

    # verse_1 e verse_2 caem no mesmo grupo; chorus_1/chorus_2 idem
    verse_idxs = [i for i, s in enumerate(sections)
                  if s.name.lower().startswith("verse")]
    assert len(verse_idxs) == 2
    assert design.group_of[verse_idxs[0]] == design.group_of[verse_idxs[1]]
    assert design.is_first_occurrence(verse_idxs[0])
    assert not design.is_first_occurrence(verse_idxs[1])

    # arco: 1.º chorus e clímax = última ocorrência do chorus
    chorus_idxs = [i for i, s in enumerate(sections) if s.kind == "chorus"]
    assert design.first_chorus_idx == chorus_idxs[0]
    assert design.climax_idx == chorus_idxs[-1]

    # âncoras: uma por secção, alinhadas a compasso, nunca antes do início
    bar = TPB * 4
    assert len(design.anchors) == len(sections)
    for s, a in zip(sections, design.anchors):
        assert a >= s.start
        assert a % bar == 0


def test_repeated_sections_share_lighting_look():
    """Fase 2.1: verse_2 reabre com o look de verse_1 (motivo por grupo)."""
    kwargs = _synthetic_song()
    events = generate_venue(**kwargs)

    from downcharter.venue import resolve_sections
    import re
    sections = resolve_sections(kwargs["events_track"], kwargs["song_end"],
                                kwargs["onsets"], kwargs["time_sig_map"], TPB)
    light_re = re.compile(r"\[lighting \(([^)]+)\)\]")

    def _presets_in(section):
        out = []
        for ev in events:
            txt = getattr(ev.msg, "text", "") or ""
            m = light_re.search(txt)
            if m and section.start <= ev.abs_tick < section.end:
                out.append(m.group(1))
        return out

    verses = [s for s in sections if s.name.lower().startswith("verse")]
    assert len(verses) == 2
    p1, p2 = _presets_in(verses[0]), _presets_in(verses[1])
    assert p1 and p2
    # replay do motivo: o início de verse_2 repete a sequência de verse_1
    n = min(len(p1), len(p2))
    assert p2[:n] == p1[:n]


def test_design_none_still_works():
    """Sem design (caminho legado), os builders continuam a funcionar."""
    from downcharter.venue import build_lighting, resolve_sections, THEMES
    kwargs = _synthetic_song()
    sections = resolve_sections(kwargs["events_track"], kwargs["song_end"],
                                kwargs["onsets"], kwargs["time_sig_map"], TPB)
    out = build_lighting(sections, THEMES["rock"], TPB,
                         kwargs["time_sig_map"], kwargs["drum_onsets"])
    assert out  # emite eventos sem design
