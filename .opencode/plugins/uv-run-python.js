export const UvRunPythonPlugin = async ({ client }) => {
    const SERVICE = "uv-run-python";
    
    // Allow list
    const WRAP_WITH_UV_RUN = new Set([
        "pytest",
        "pytest-bdd",
        "alembic",
        "ty",
        "pre-commit",
        "ruff",
        "black",
        "mypy",
    ]);

    function escapeRegExp(s) {
        return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    function rewriteCommand(command) {
        if (typeof command !== "string") return command;

        if (/\buv\s+run\b/.test(command)) return command;

        const envPrefixRe =
            "((?:[A-Za-z_][A-Za-z0-9_]*=(?:'[^']*'|\"[^\"]*\"|[^ \\t\\n;&|]+)\\s+)*)";

        const sepRe = "(^|(?:\\s*(?:&&|\\|\\||;|\\n)\\s*))";

        let rewritten = command;

        // 1) python / python3 / python3.12 -> uv run ... python
        {
            const pyRe = new RegExp(
                `${sepRe}${envPrefixRe}(python(?:\\d+(?:\\.\\d+)?)?)\\b(?!-)`,
                "g"
            );

            rewritten = rewritten.replace(pyRe, (full, sep, envPrefix, pyBin) => {
                const m = /^python(\d+(?:\.\d+)?)$/.exec(pyBin);
                if (m) {
                    const ver = m[1];

                    // python3 -> uv run python
                    if (ver === "3") {
                        return `${sep}${envPrefix}uv run python`;
                    }

                    // python3.12 -> uv run --python 3.12 python
                    return `${sep}${envPrefix}uv run --python ${ver} -- python`;
                }

                // python
                return `${sep}${envPrefix}uv run python`;
            });
        }

        // 2) pytest / pytest-bdd -> uv run pytest / uv run pytest-bdd
        {
            const tools = Array.from(WRAP_WITH_UV_RUN).map(escapeRegExp).join("|");
            const toolRe = new RegExp(
                `${sepRe}${envPrefixRe}(${tools})\\b`,
                "g"
            );

            rewritten = rewritten.replace(toolRe, (full, sep, envPrefix, tool) => {
                return `${sep}${envPrefix}uv run ${tool}`;
            });
        }

        return rewritten;
    }

    async function logDebug(original, rewritten) {
        if (!client?.app?.log) return;
        await client.app.log({
            service: SERVICE,
            level: "debug",
            message: "Rewrote command to run via uv",
            extra: { original, rewritten },
        });
    }

    return {
        "tool.execute.before": async (input, output) => {
            if (!input || input.tool !== "bash") return;
            const original = output?.args?.command;
            if (typeof original !== "string") return;

            const rewritten = rewriteCommand(original);
            if (rewritten !== original) {
                output.args.command = rewritten;
                await logDebug(original, rewritten);
            }
        },

        tool: {
            execute: {
                before: async (input, output) => {
                    if (!input || input.tool !== "bash") return;
                    const original = output?.args?.command;
                    if (typeof original !== "string") return;

                    const rewritten = rewriteCommand(original);
                    if (rewritten !== original) {
                        output.args.command = rewritten;
                        await logDebug(original, rewritten);
                    }
                },
            },
        },
    };
};
