# Betting Boyz — SofaScore Prediction Bot (NO ODDS)
# Pure prediction mode:
# - Uses SofaScore scheduled events + team recent results to generate confidence-based tips
# - No bookmaker odds
# - Sends 2 predictions (SAFE + VALUE) to WhatsApp via Twilio
# - Two slots supported: morning / afternoon (avoids duplicates via used_picks.json)
#
# Requirements:
#   pip install requests twilio
#
# Run:
#   python betting_boyz_sofascore_bot.py --slot morning
#   python betting_boyz_sofascore_bot.py --slot afternoon
#
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, date, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from twilio.rest import Client

print("🔹 Betting Boyz (SofaScore) starting...")

# ------------------------------
# ENV CONFIG (no secrets in code)
# ------------------------------
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
TO_WHATSAPP_NUMBER = os.getenv("TO_WHATSAPP_NUMBER")
WHATSAPP_CHANNEL_LINK = os.getenv("WHATSAPP_CHANNEL_LINK", "")

USED_PICKS_FILE = os.getenv("USED_PICKS_FILE", "used_picks.json")
LOG_FILE = os.getenv("PICKS_LOG_FILE", "picks_log.json")
DEBUG_LOG_FILE = os.getenv("DEBUG_LOG_FILE", "sofascore_debug.log")

CACHE_DIR = os.getenv("CACHE_DIR", ".cache")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "21600"))  # 6 hours

# Selection windows (local time on runner)
MORNING_CUTOFF_HOUR = int(os.getenv("MORNING_CUTOFF_HOUR", "12"))
LOOKAHEAD_HOURS = int(os.getenv("LOOKAHEAD_HOURS", "36"))  # allow tomorrow morning if today is empty

# Model config
LAST_N_MATCHES = int(os.getenv("LAST_N_MATCHES", "8"))
MIN_CONF_SAFE = float(os.getenv("MIN_CONF_SAFE", "0.62"))
MIN_CONF_VALUE = float(os.getenv("MIN_CONF_VALUE", "0.55"))

# SofaScore base (unofficial)
SOFA_BASE = "https://api.sofascore.com/api/v1"


def _now_local() -> datetime:
    return datetime.now().astimezone()


def append_debug(text: str) -> None:
    ts = _now_local().isoformat(timespec="seconds")
    with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")


def load_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _cache_path(url: str) -> str:
    safe = url.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_").replace("=", "_")
    return os.path.join(CACHE_DIR, f"{safe}.json")


def get_json_cached(url: str) -> Dict[str, Any]:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = _cache_path(url)
    if os.path.exists(cp):
        age = time.time() - os.path.getmtime(cp)
        if age <= CACHE_TTL_SECONDS:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BettingBoyzBot/1.0; +https://github.com/)",
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def parse_sofa_dt(epoch_seconds: Optional[int]) -> Optional[datetime]:
    if not epoch_seconds:
        return None
    try:
        return datetime.fromtimestamp(int(epoch_seconds), tz=timezone.utc).astimezone()
    except Exception:
        return None


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class TeamForm:
    played: int
    wins: int
    draws: int
    losses: int
    gf: int
    ga: int
    over25: int
    btts: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.played if self.played else 0.0

    @property
    def avg_gf(self) -> float:
        return self.gf / self.played if self.played else 0.0

    @property
    def avg_ga(self) -> float:
        return self.ga / self.played if self.played else 0.0

    @property
    def over25_rate(self) -> float:
        return self.over25 / self.played if self.played else 0.0

    @property
    def btts_rate(self) -> float:
        return self.btts / self.played if self.played else 0.0


def fetch_scheduled_events(day: date) -> List[Dict[str, Any]]:
    url = f"{SOFA_BASE}/sport/football/scheduled-events/{day.isoformat()}"
    data = get_json_cached(url)
    return data.get("events", []) or []


def fetch_team_last_events(team_id: int, page: int = 0) -> List[Dict[str, Any]]:
    url = f"{SOFA_BASE}/team/{team_id}/events/last/{page}"
    data = get_json_cached(url)
    return data.get("events", []) or []


def team_form_from_events(team_id: int, n: int) -> TeamForm:
    events = fetch_team_last_events(team_id, page=0)
    finished: List[Dict[str, Any]] = []
    for e in events:
        st = (e.get("status") or {}).get("type")
        if st != "finished":
            continue
        hs = (e.get("homeScore") or {}).get("current")
        as_ = (e.get("awayScore") or {}).get("current")
        if hs is None or as_ is None:
            continue
        finished.append(e)
        if len(finished) >= n:
            break

    wins = draws = losses = gf = ga = over25 = btts = 0
    for e in finished:
        is_home = (e.get("homeTeam") or {}).get("id") == team_id
        hs = int((e.get("homeScore") or {}).get("current") or 0)
        as_ = int((e.get("awayScore") or {}).get("current") or 0)

        gf_i, ga_i = (hs, as_) if is_home else (as_, hs)
        gf += gf_i
        ga += ga_i

        if gf_i > ga_i:
            wins += 1
        elif gf_i == ga_i:
            draws += 1
        else:
            losses += 1

        if (hs + as_) >= 3:
            over25 += 1
        if hs >= 1 and as_ >= 1:
            btts += 1

    return TeamForm(len(finished), wins, draws, losses, gf, ga, over25, btts)


def prob_home_win(home: TeamForm, away: TeamForm) -> float:
    if home.played == 0 or away.played == 0:
        return 0.50
    away_loss_rate = away.losses / away.played
    p = 0.5 * home.win_rate + 0.5 * away_loss_rate
    gd_home = (home.gf - home.ga) / max(1, home.played)
    gd_away = (away.gf - away.ga) / max(1, away.played)
    p += 0.04 * (gd_home - gd_away)
    return clamp(p, 0.05, 0.90)


def prob_over25(home: TeamForm, away: TeamForm) -> float:
    if home.played == 0 or away.played == 0:
        return 0.50
    p = 0.5 * home.over25_rate + 0.5 * away.over25_rate
    avg_total = (home.avg_gf + home.avg_ga + away.avg_gf + away.avg_ga) / 2.0
    p += 0.03 * (avg_total - 2.6)
    return clamp(p, 0.05, 0.90)


def prob_btts(home: TeamForm, away: TeamForm) -> float:
    if home.played == 0 or away.played == 0:
        return 0.50
    p = 0.5 * home.btts_rate + 0.5 * away.btts_rate
    p += 0.02 * ((home.avg_gf + away.avg_gf) - 2.4)
    return clamp(p, 0.05, 0.90)


def confidence_stars(p: float) -> str:
    if p >= 0.72:
        return "⭐⭐⭐⭐⭐"
    if p >= 0.65:
        return "⭐⭐⭐⭐"
    if p >= 0.58:
        return "⭐⭐⭐"
    if p >= 0.52:
        return "⭐⭐"
    return "⭐"


def build_reason_lines(home: TeamForm, away: TeamForm, pick: str) -> List[str]:
    if home.played and away.played:
        lines = [
            f"Form (last {home.played}/{away.played}): Home W{home.wins}-D{home.draws}-L{home.losses} | Away W{away.wins}-D{away.draws}-L{away.losses}",
            f"Goals: Home {home.gf}:{home.ga} | Away {away.gf}:{away.ga}",
        ]
        if "Over 2.5" in pick:
            lines.append(f"Over 2.5 hit-rate: Home {home.over25_rate:.0%} | Away {away.over25_rate:.0%}")
        if "BTTS" in pick:
            lines.append(f"BTTS hit-rate: Home {away.btts_rate:.0%} | Away {away.btts_rate:.0%}")
        return lines
    return ["Not enough recent match data; conservative estimate used."]


@dataclass
class Prediction:
    event_id: int
    kickoff_dt: Optional[datetime]
    kickoff_time: str
    league: str
    home_name: str
    away_name: str
    pick: str
    confidence: float
    stars: str
    reasons: List[str]


def candidate_predictions_for_event(event: Dict[str, Any]) -> List[Prediction]:
    home = event.get("homeTeam") or {}
    away = event.get("awayTeam") or {}
    tournament = (event.get("tournament") or {}).get("name") or ""
    category = ((event.get("tournament") or {}).get("category") or {}).get("name") or ""
    league = f"{category} - {tournament}".strip(" -")

    start_dt = parse_sofa_dt(event.get("startTimestamp"))
    kickoff_time = start_dt.strftime("%H:%M") if start_dt else "TBC"

    home_id = int(home.get("id") or 0)
    away_id = int(away.get("id") or 0)
    if not home_id or not away_id:
        return []

    home_form = team_form_from_events(home_id, LAST_N_MATCHES)
    away_form = team_form_from_events(away_id, LAST_N_MATCHES)

    eid = int(event.get("id") or 0)
    hn = home.get("name") or home.get("shortName") or "Home"
    an = away.get("name") or away.get("shortName") or "Away"

    preds = []
    p_hw = prob_home_win(home_form, away_form)
    preds.append(Prediction(eid, start_dt, kickoff_time, league, hn, an, "Home Win", p_hw, confidence_stars(p_hw),
                            build_reason_lines(home_form, away_form, "Home Win")))
    p_o25 = prob_over25(home_form, away_form)
    preds.append(Prediction(eid, start_dt, kickoff_time, league, hn, an, "Over 2.5 Goals", p_o25, confidence_stars(p_o25),
                            build_reason_lines(home_form, away_form, "Over 2.5")))
    p_btts = prob_btts(home_form, away_form)
    preds.append(Prediction(eid, start_dt, kickoff_time, league, hn, an, "BTTS: Yes", p_btts, confidence_stars(p_btts),
                            build_reason_lines(home_form, away_form, "BTTS")))
    return preds


def build_predictions(events: List[Dict[str, Any]]) -> List[Prediction]:
    all_preds: List[Prediction] = []
    for e in events:
        all_preds.extend(candidate_predictions_for_event(e))
    all_preds.sort(key=lambda p: (p.confidence, -(p.kickoff_dt.timestamp() if p.kickoff_dt else 0)), reverse=True)
    return all_preds


def distinct_picks(preds: List[Prediction], used: List[str]) -> Tuple[Optional[Prediction], Optional[Prediction]]:
    used_set = set(used or [])

    def pick_key(p: Prediction) -> str:
        return f"{p.home_name} vs {p.away_name} | {p.pick}"

    def match_key(p: Prediction) -> str:
        return f"{p.home_name} vs {p.away_name}"

    safe = value = None

    for p in preds:
        if pick_key(p) in used_set:
            continue
        if p.confidence >= MIN_CONF_SAFE:
            safe = p
            break

    for p in preds:
        if pick_key(p) in used_set:
            continue
        if safe and match_key(p) == match_key(safe):
            continue
        if p.confidence >= MIN_CONF_VALUE:
            value = p
            break

    if safe is None:
        for p in preds:
            if pick_key(p) not in used_set:
                safe = p
                break

    if value is None:
        for p in preds:
            if safe and match_key(p) == match_key(safe):
                continue
            if pick_key(p) not in used_set:
                value = p
                break

    return safe, value


def countdown(dt: Optional[datetime]) -> str:
    if not dt:
        return "TBC"
    now = _now_local()
    delta = dt - now
    if delta.total_seconds() <= 0:
        return "Live / Started"
    h, rem = divmod(int(delta.total_seconds()), 3600)
    m = rem // 60
    return f"Starts in {h}h {m}m"


def format_pick(title: str, p: Prediction) -> str:
    reasons = "\n".join([f"• {x}" for x in p.reasons[:3]])
    return (
        f"{title} {p.stars}\n"
        f"{p.home_name} vs {p.away_name}\n"
        f"🏟️ {p.league}\n"
        f"⏰ Kick-off: {p.kickoff_time} — {countdown(p.kickoff_dt)}\n"
        f"Prediction: {p.pick}\n"
        f"Confidence: {p.confidence:.0%}\n"
        f"Reason:\n{reasons}\n"
    )


def format_message(slot: str, safe: Prediction, value: Prediction) -> str:
    today = _now_local().strftime("%A, %d %B %Y")
    msg = f"⚽ BETTING BOYZ — SOFASCORE PREDICTIONS\n📅 {today}\n🕒 Slot: {slot.upper()}\n\n"
    msg += format_pick("🔒 SAFE PICK", safe) + "\n"
    msg += format_pick("🎯 VALUE PICK", value) + "\n"
    if WHATSAPP_CHANNEL_LINK:
        msg += f"👉 Join the Boyz: {WHATSAPP_CHANNEL_LINK}\n"
    msg += "— Betting Boyz"
    return msg


def send_whatsapp(message: str) -> None:
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TO_WHATSAPP_NUMBER):
        raise RuntimeError("Missing Twilio env vars (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TO_WHATSAPP_NUMBER).")
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    msg = client.messages.create(body=message, from_=TWILIO_WHATSAPP_FROM, to=TO_WHATSAPP_NUMBER)
    append_debug(f"WhatsApp message sent! SID: {msg.sid}")


def filter_events_by_slot(events: List[Dict[str, Any]], slot: str, target_day: date) -> List[Dict[str, Any]]:
    now = _now_local()
    out = []
    for e in events:
        dt = parse_sofa_dt(e.get("startTimestamp"))
        if not dt:
            continue
        if dt < now - timedelta(hours=1) or dt > now + timedelta(hours=LOOKAHEAD_HOURS):
            continue
        if dt.date() != target_day:
            continue
        if slot == "morning" and dt.hour <= MORNING_CUTOFF_HOUR:
            out.append(e)
        elif slot == "afternoon" and dt.hour > MORNING_CUTOFF_HOUR:
            out.append(e)
    return sorted(out, key=lambda ev: parse_sofa_dt(ev.get("startTimestamp")) or now)


def self_test() -> None:
    tf = TeamForm(played=8, wins=5, draws=2, losses=1, gf=14, ga=7, over25=5, btts=4)
    tf2 = TeamForm(played=8, wins=2, draws=2, losses=4, gf=8, ga=13, over25=4, btts=5)
    assert 0.0 <= prob_home_win(tf, tf2) <= 1.0
    assert 0.0 <= prob_over25(tf, tf2) <= 1.0
    assert 0.0 <= prob_btts(tf, tf2) <= 1.0
    assert confidence_stars(0.73) == "⭐⭐⭐⭐⭐"
    print("✅ Self-test passed")


def main(args: argparse.Namespace) -> None:
    slot = args.slot.lower().strip()
    if slot not in ("morning", "afternoon"):
        raise ValueError("--slot must be morning or afternoon")

    if args.self_test:
        self_test()
        return

    target_day = date.fromisoformat(args.date) if args.date else _now_local().date()

    events = fetch_scheduled_events(target_day)
    if not events:
        append_debug(f"No events returned for {target_day.isoformat()} — trying next day.")
        target_day = target_day + timedelta(days=1)
        events = fetch_scheduled_events(target_day)

    slot_events = filter_events_by_slot(events, slot, target_day)
    append_debug(f"Events fetched: {len(events)} | Slot events ({slot}): {len(slot_events)}")

    if not slot_events:
        raise RuntimeError(f"No {slot} events found for {target_day.isoformat()} within lookahead window.")

    preds = build_predictions(slot_events)
    append_debug(f"Predictions built: {len(preds)}")

    used = load_json(USED_PICKS_FILE, default=[])
    safe, value = distinct_picks(preds, used)
    if not safe or not value:
        raise RuntimeError("Could not build 2 distinct predictions (SAFE + VALUE).")

    used.append(f"{safe.home_name} vs {safe.away_name} | {safe.pick}")
    used.append(f"{value.home_name} vs {value.away_name} | {value.pick}")
    used = used[-300:]
    save_json(USED_PICKS_FILE, used)

    log = load_json(LOG_FILE, default={})
    key = f"{target_day.isoformat()}_{slot}"
    log[key] = {"safe": safe.__dict__, "value": value.__dict__, "generated_at": _now_local().isoformat()}
    save_json(LOG_FILE, log)

    msg = format_message(slot, safe, value)
    send_whatsapp(msg)
    print("✅ Message sent")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slot", required=True, help="morning or afternoon")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (defaults to today local)")
    p.add_argument("--self-test", action="store_true", help="Run self-test (no network)")
    main(p.parse_args())
