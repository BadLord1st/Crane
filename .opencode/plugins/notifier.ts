import type { Plugin } from "@opencode-ai/plugin";
import { readFile } from "node:fs/promises";
import path from "node:path";

type Kind = "done" | "error" | "permission" | "stall" | "stall_resolved" | "status";

async function loadEnvFile(root?: string) {
  if (!root) return;
  const envPath = path.join(root, ".env");
  let content: string;
  try {
    content = await readFile(envPath, "utf-8");
  } catch {
    return;
  }
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const eq = line.indexOf("=");
    if (eq <= 0) continue;
    const key = line.slice(0, eq).trim();
    if (!key || Object.prototype.hasOwnProperty.call(process.env, key)) continue;
    let value = line.slice(eq + 1).trim();
    if ((value.startsWith("\"") && value.endsWith("\"")) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    process.env[key] = value;
  }
}

function isLikelyHash(value?: string) {
  if (!value) return false;
  return /^[a-f0-9]{32,}$/.test(value);
}

function getEnv(name: string, fallback = ""): string {
  return process.env[name] ?? fallback;
}

function numEnv(name: string, fallback: number): number {
  const v = process.env[name];
  if (!v) return fallback;
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function nowMs() {
  return Date.now();
}

function getParentSessionId(payload: any, event: any) {
  return (
    event?.properties?.session?.parentID ??
    event?.session?.parentID ??
    payload?.session?.parentID
  );
}

function isSubagentEvent(payload: any, event: any) {
  const markers = [
    event?.properties?.agent?.mode,
    event?.properties?.agent?.type,
    event?.properties?.agent?.role,
    event?.agent?.mode,
    event?.agent?.type,
    event?.agent?.role,
    payload?.agent?.mode,
    payload?.agent?.type,
    payload?.agent?.role,
    event?.properties?.session?.mode,
    event?.properties?.session?.type,
    event?.session?.mode,
    event?.session?.type,
    payload?.session?.mode,
    payload?.session?.type,
  ];

  if (markers.some((value) => typeof value === "string" && value.toLowerCase() === "subagent")) {
    return true;
  }

  const parentId = getParentSessionId(payload, event);

  if (typeof parentId === "string" && parentId.length > 0) {
    return true;
  }

  if (
    event?.properties?.agent?.isSubagent === true ||
    event?.agent?.isSubagent === true ||
    payload?.agent?.isSubagent === true
  ) {
    return true;
  }

  return false;
}

function hasParentSessionId(payload: any, event: any) {
  const parentId = getParentSessionId(payload, event);

  return typeof parentId === "string" && parentId.length > 0;
}

function isPrimarySessionEvent(payload: any, event: any) {
  if (hasParentSessionId(payload, event)) return false;
  if (
    event?.properties?.agent?.isSubagent === true ||
    event?.agent?.isSubagent === true ||
    payload?.agent?.isSubagent === true
  ) {
    return false;
  }
  return true;
}

export const NotifierPlugin: Plugin = async ({ project, client, directory, worktree }) => {
  await loadEnvFile(directory ?? worktree ?? process.cwd());

  const NOTIFIER_URL = getEnv(
    "OPENCODE_NOTIFIER_URL",
    "http://127.0.0.1:8900/v1/events/opencode",
  );

  const ATTENTION_KEY = getEnv("OPENCODE_NOTIFIER_KEY");

  const STALL_MS = numEnv("OPENCODE_NOTIFIER_STALL_MS", 180_000);

  const STALL_REMIND_MS = numEnv("OPENCODE_NOTIFIER_STALL_REMIND_MS", 900_000);

  const THROTTLE_MS = numEnv("OPENCODE_NOTIFIER_THROTTLE_MS", 30_000);
  const QUESTION_EVENT_TYPES = new Set(
    getEnv("OPENCODE_NOTIFIER_QUESTION_EVENT_TYPES", "session.question,session.prompt")
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );

  let lastActivityAt = nowMs();

  let stalled = false;
  let stallStartedAt: number | null = null;
  let lastStallReminderAt: number | null = null;

  const lastNotifyAt = new Map<string, number>();
  let warnedMissingKey = false;

  const shouldNotify = (dedupKey: string, overrideThrottleMs?: number) => {
    const throttle = overrideThrottleMs ?? THROTTLE_MS;
    const t = lastNotifyAt.get(dedupKey) ?? 0;
    if (nowMs() - t < throttle) return false;
    lastNotifyAt.set(dedupKey, nowMs());
    return true;
  };

  const warnMissingKey = async () => {
    if (warnedMissingKey) return;
    warnedMissingKey = true;
    if (!client?.app?.log) return;
    await client.app.log({
      service: "opencode-notifier-plugin",
      level: "warn",
      message: "Notifier disabled: OPENCODE_NOTIFIER_KEY is not set",
    });
  };

  const ensureKey = async () => {
    if (ATTENTION_KEY) return true;
    await warnMissingKey();
    return false;
  };

  const projectRoot = directory ?? worktree ?? "";
  const rawProjectName = (project as any)?.name;
  const projectName =
    rawProjectName && !isLikelyHash(rawProjectName)
      ? rawProjectName
      : projectRoot
        ? path.basename(projectRoot)
        : "unknown";
  const projectId = (project as any)?.id ?? rawProjectName ?? projectName;

  const post = async (kind: Kind, event: any, extra?: Record<string, unknown>) => {
    if (!(await ensureKey())) return;

    const payload = {
      kind,
      at: new Date().toISOString(),
      project: {
        id: projectId,
        name: projectName,
      },
      directory,
      worktree,
      event,
      extra,
    };

    try {
      const resp = await fetch(NOTIFIER_URL, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-attention-key": ATTENTION_KEY,
        },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        if (client?.app?.log) {
          await client.app.log({
            service: "opencode-notifier-plugin",
            level: "warn",
            message: "Notifier returned non-OK",
            extra: { status: resp.status, kind },
          });
        }
      }
    } catch (e: any) {
      if (client?.app?.log) {
        await client.app.log({
          service: "opencode-notifier-plugin",
          level: "error",
          message: "Notifier request failed",
          extra: { error: String(e?.message ?? e), kind },
        });
      }
    }
  };

  const enterStall = async (idleFor: number) => {
    stalled = true;
    stallStartedAt = nowMs();
    lastStallReminderAt = nowMs();

    await post(
      "stall",
      { type: "stall", properties: { idleMs: idleFor } },
      { idleMs: idleFor, first: true },
    );
  };

  const sendStallReminder = async (idleFor: number) => {
    lastStallReminderAt = nowMs();

    await post(
      "stall",
      { type: "stall", properties: { idleMs: idleFor } },
      { idleMs: idleFor, reminder: true },
    );
  };

  const resolveStall = async (event: any) => {
    const stallDurationMs = stallStartedAt ? nowMs() - stallStartedAt : null;
    stalled = false;
    stallStartedAt = null;
    lastStallReminderAt = null;

    await post("stall_resolved", event, { resolved: true, stallDurationMs });
  };

  const stallTimer = setInterval(async () => {
    const idleFor = nowMs() - lastActivityAt;

    if (!stalled && idleFor >= STALL_MS) {
      if (shouldNotify("stall:first", STALL_MS)) {
        await enterStall(idleFor);
      }
      return;
    }

    if (stalled && STALL_REMIND_MS > 0 && idleFor >= STALL_MS) {
      const last = lastStallReminderAt ?? 0;
      if (nowMs() - last >= STALL_REMIND_MS) {
        if (shouldNotify("stall:reminder", STALL_REMIND_MS)) {
          await sendStallReminder(idleFor);
        }
      }
    }
  }, 30_000);

  const handleEvent = async (payload: any) => {
    const event = payload?.event ?? payload;
    if (!event || !event.type) return;

    if (isSubagentEvent(payload, event)) {
      return;
    }

    const wasStalled = stalled;

    lastActivityAt = nowMs();

    if (event.type === "session.idle") {
      stalled = false;
      stallStartedAt = null;
      lastStallReminderAt = null;

      if (isPrimarySessionEvent(payload, event) && shouldNotify("done")) {
        await post("done", event);
      }
      return;
    }

    if (event.type === "session.error") {
      stalled = false;
      stallStartedAt = null;
      lastStallReminderAt = null;

      if (isPrimarySessionEvent(payload, event) && shouldNotify("error")) {
        await post("error", event);
      }
      return;
    }

    if (wasStalled) {
      if (shouldNotify("stall:resolved", 60_000)) {
        await resolveStall(event);
      } else {
        stalled = false;
        stallStartedAt = null;
        lastStallReminderAt = null;
      }
    }

    if (event.type === "permission.ask" || event.type === "permission.asked") {
      if (shouldNotify("permission:ask")) {
        await post("permission", event, { eventType: event.type });
      }
      return;
    }

    if (QUESTION_EVENT_TYPES.has(event.type)) {
      const key = `question:${event.type}`;
      if (shouldNotify(key)) {
        await post("status", event, { statusType: "question", eventType: event.type });
      }
      return;
    }

    if (event.type === "session.status") {
      const statusType = event?.properties?.status?.type;
      if (statusType && ["waiting", "blocked"].includes(statusType)) {
        const k = `status:${statusType}`;
        if (shouldNotify(k)) {
          await post("status", event, { statusType });
        }
      }
      return;
    }
  };
  return {
    event: handleEvent,

    // @ts-ignore
    dispose: async () => clearInterval(stallTimer),
  };
};

export default NotifierPlugin;
