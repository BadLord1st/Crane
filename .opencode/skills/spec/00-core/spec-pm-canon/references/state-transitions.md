# State transitions

Allowed states: `NEW | IN_PROGRESS | BLOCKED | DONE | DROPPED`

## Transition requirements
NEW -> IN_PROGRESS
- owner set
- plan exists (at least TODO skeleton)

IN_PROGRESS -> BLOCKED
- blocker recorded (pulse or ticket note)
- next action captured

BLOCKED -> IN_PROGRESS
- blocker resolved; pulse event recorded

IN_PROGRESS -> DONE
- all AC checked
- evidence added
- traceability updated (SPEC → AC → tests → code)

Any -> DROPPED
- rationale recorded (pulse)
