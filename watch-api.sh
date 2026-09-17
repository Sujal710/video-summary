#!/usr/bin/env bash
# Follow the newest log and show ONLY outbound calls, tagged by destination:
#   [AI->] [AI<-]      OpenRouter / GLM  (the model)
#   [ALERT->] [ALERT<-] the VMS alert API (a real POST)
#   [ALERT-DRY]        an alert that WOULD have been posted (posting disabled)
#   [AI!!] [ALERT!!]   failures
# grep -F on the tags, because the word "alert" appears inside every description.
cd "$(dirname "$0")"
LOG="${1:-$(ls -t logs/*.log 2>/dev/null | head -1)}"
[ -n "$LOG" ] || { echo "no log files in logs/"; exit 1; }
echo "watching: $LOG"
echo "  [AI->]/[AI<-]        model calls to OpenRouter"
echo "  [ALERT->]/[ALERT<-]  real POST to the VMS alert API"
echo "  [ALERT-DRY]          alert suppressed because ALERT_POST_ENABLED=false"
echo
tail -f "$LOG" | grep --line-buffered -F \
  -e '[AI->]' -e '[AI<-]' -e '[AI!!]' \
  -e '[ALERT->]' -e '[ALERT<-]' -e '[ALERT!!]' -e '[ALERT-DRY]' \
  -e 'alert(s)' -e 'alerts:'
