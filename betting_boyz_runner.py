from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import betting_boyz_sofascore_bot as bot

def must_env(keys):
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        print("Missing environment variables:", missing)
        sys.exit(1)

if __name__ == "__main__":
    print("Betting Boyz Runner Started", datetime.now(timezone.utc).isoformat())
    must_env(["TWILIO_ACCOUNT_SID","TWILIO_AUTH_TOKEN","TO_WHATSAPP_NUMBER"])

    slot = (os.getenv("SLOT") or "").strip().lower()
    date_ = (os.getenv("RUN_DATE") or "").strip() or None

    if slot in ("morning","afternoon"):
        bot.main(bot.argparse.Namespace(slot=slot, date=date_, self_test=False))
    else:
        bot.main(bot.argparse.Namespace(slot="morning", date=date_, self_test=False))
        bot.main(bot.argparse.Namespace(slot="afternoon", date=date_, self_test=False))
