'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');
const { Client } = require('ssh2');

const PORT = parseInt(process.env.CONTEXTFORGE_PORT || '4328');
const SERVERS_JSON = process.env.SERVERS_CONFIG || path.join(os.homedir(), '.config', 'machina', 'servers.json');
const BUGCAPTURE_DIR = process.env.BUGCAPTURE_OUTPUT_DIR || path.join(os.homedir(), 'bugcapture-output');

function expandHome(p) {
  if (!p) return p;
  return p.startsWith('~') ? path.join(os.homedir(), p.slice(1)) : p;
}

function readServers() {
  try {
    const raw = fs.readFileSync(SERVERS_JSON, 'utf8');
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : (parsed.connections || []);
  } catch { return []; }
}

function findConnection(id) {
  return readServers().find(c => c.id === id) || null;
}

function sanitizeConnection(c) {
  const { password, ...rest } = c;
  return rest;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', chunk => { data += chunk; });
    req.on('end', () => {
      try { resolve(data ? JSON.parse(data) : {}); }
      catch (e) { reject(e); }
    });
    req.on('error', reject);
  });
}

function json(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Access-Control-Allow-Origin': '*',
    'Content-Length': Buffer.byteLength(body)
  });
  res.end(body);
}

function gitContext(localPath) {
  return new Promise((resolve) => {
    const { execFile } = require('child_process');
    const cwd = expandHome(localPath);
    let results = { status: '', log: '', diff: '' };
    let pending = 3;
    const done = () => { if (--pending === 0) resolve(results); };
    execFile('git', ['-C', cwd, 'status', '--short'], { timeout: 10000 }, (err, stdout) => { results.status = err ? '' : stdout.trim(); done(); });
    execFile('git', ['-C', cwd, 'log', '--oneline', '-5'], { timeout: 10000 }, (err, stdout) => { results.log = err ? '' : stdout.trim(); done(); });
    execFile('git', ['-C', cwd, 'diff', '--stat', 'HEAD'], { timeout: 10000 }, (err, stdout) => { results.diff = err ? '' : stdout.trim(); done(); });
  });
}

function sshExec(conn, command) {
  return new Promise((resolve, reject) => {
    conn.exec(command, (err, stream) => {
      if (err) return reject(err);
      let out = '', errOut = '';
      stream.on('data', d => { out += d; });
      stream.stderr.on('data', d => { errOut += d; });
      stream.on('close', () => resolve(out || errOut));
    });
  });
}

function sshConnect(connection) {
  return new Promise((resolve, reject) => {
    const client = new Client();
    const opts = { host: connection.host, port: connection.port || 22, username: connection.username, readyTimeout: 15000 };
    const auth = connection.auth || 'password';
    if (auth === 'key' && connection.keyFile) {
      try { opts.privateKey = fs.readFileSync(expandHome(connection.keyFile)); }
      catch (e) { return reject(new Error('Cannot read key file: ' + e.message)); }
    } else {
      opts.password = connection.password || '';
    }
    client.on('ready', () => resolve(client));
    client.on('error', reject);
    client.connect(opts);
  });
}

async function scanLocalProject(localPath) {
  const abs = expandHome(localPath);
  const has = f => { try { return fs.existsSync(path.join(abs, f)); } catch { return false; } };
  const read = f => { try { return fs.readFileSync(path.join(abs, f), 'utf8'); } catch { return ''; } };

  let projectType = 'generic';
  let configFileNames = [];
  let suggestedLogPaths = [];

  if (has('configuration.php')) {
    projectType = 'joomla';
    configFileNames = ['configuration.php'];
    const cfg = read('configuration.php');
    const m = cfg.match(/\$log_path\s*=\s*'([^']+)'/);
    if (m) suggestedLogPaths.push(m[1] + '/error.php');
    suggestedLogPaths.push('/var/log/apache2/error.log', '/tmp/joomla_error.log');
  } else if (has('wp-config.php')) {
    projectType = 'wordpress';
    configFileNames = ['wp-config.php'];
    if (read('wp-config.php').includes('WP_DEBUG_LOG'))
      suggestedLogPaths.push(path.join(abs, 'wp-content/debug.log'));
    suggestedLogPaths.push('/var/log/apache2/error.log');
  } else if (has('artisan') && has('composer.json')) {
    projectType = 'laravel';
    configFileNames = ['composer.json'];
    if (has('.env')) configFileNames.push('.env');
    suggestedLogPaths.push(path.join(abs, 'storage/logs/laravel.log'));
  } else if (has('package.json')) {
    projectType = 'nodejs';
    configFileNames = ['package.json'];
    if (has('.env')) configFileNames.push('.env');
    try {
      fs.readdirSync(path.join(abs, 'logs'))
        .filter(f => f.endsWith('.log'))
        .forEach(f => suggestedLogPaths.push(path.join(abs, 'logs', f)));
    } catch {}
  } else if (has('composer.json')) {
    projectType = 'php';
    configFileNames = ['composer.json'];
    if (has('.env')) configFileNames.push('.env');
    suggestedLogPaths.push('/var/log/apache2/error.log');
  } else if (has('index.php') || has('index.html')) {
    projectType = 'web';
    suggestedLogPaths.push('/var/log/apache2/error.log', '/var/log/nginx/error.log');
  }

  if (!suggestedLogPaths.includes('/var/log/apache2/error.log'))
    suggestedLogPaths.push('/var/log/apache2/error.log');

  const configFiles = configFileNames.map(f => {
    try {
      const stat = fs.statSync(path.join(abs, f));
      return { name: f, path: path.join(abs, f), size: stat.size, mtime: stat.mtime.toISOString() };
    } catch { return null; }
  }).filter(Boolean);

  const existingLogs = suggestedLogPaths.filter(p => { try { return fs.existsSync(p); } catch { return false; } });
  const gitInfo = await gitContext(abs).catch(() => ({ status: '', log: '', diff: '' }));

  return {
    ok: true, projectType,
    projectName: path.basename(abs),
    rootPath: abs, configFiles,
    suggestedLogPaths: existingLogs.length ? existingLogs : suggestedLogPaths.slice(0, 4),
    gitInfo, hasGit: !!(gitInfo.log || gitInfo.status),
  };
}

async function scanRemoteProject(conn, remotePath) {
  const client = await sshConnect(conn);
  const cmd = `cd "${remotePath}" 2>/dev/null || exit 1;
TYPE=generic;
[ -f configuration.php ] && TYPE=joomla;
[ -f wp-config.php ] && TYPE=wordpress;
[ -f artisan ] && [ -f composer.json ] && TYPE=laravel;
[ -f package.json ] && [ "$TYPE" = "generic" ] && TYPE=nodejs;
[ -f composer.json ] && [ "$TYPE" = "generic" ] && TYPE=php;
echo "TYPE=$TYPE";
echo "---FILES---";
for f in configuration.php wp-config.php package.json .env composer.json artisan; do [ -f "$f" ] && echo "$f"; done;
echo "---LOGS---";
for p in /var/log/apache2/error.log /var/log/nginx/error.log /var/log/php8.2-fpm.log /var/log/mysql/error.log /tmp/joomla_error.log; do [ -f "$p" ] && echo "$p"; done;
echo "---GIT---";
git log --oneline -5 2>/dev/null || true;
echo "---GITSTATUS---";
git status --short 2>/dev/null || true;`;

  const raw = await sshExec(client, cmd);
  client.end();

  const sections = { type: 'generic', files: [], logs: [], git: '', gitstatus: '' };
  let cur = 'pre';
  raw.split('\n').forEach(line => {
    line = line.trim();
    if (line.startsWith('TYPE=')) { sections.type = line.split('=')[1]; return; }
    if (line === '---FILES---') { cur = 'files'; return; }
    if (line === '---LOGS---') { cur = 'logs'; return; }
    if (line === '---GIT---') { cur = 'git'; return; }
    if (line === '---GITSTATUS---') { cur = 'gitstatus'; return; }
    if (cur === 'files' && line) sections.files.push(line);
    else if (cur === 'logs' && line) sections.logs.push(line);
    else if (cur === 'git' && line) sections.git += line + '\n';
    else if (cur === 'gitstatus' && line) sections.gitstatus += line + '\n';
  });

  const configFiles = sections.files.map(f => ({ name: f, path: path.join(remotePath, f), size: 0, mtime: '' }));
  const gitInfo = { status: sections.gitstatus.trim(), log: sections.git.trim(), diff: '' };
  const icons = { joomla: '🟠', wordpress: '🔵', laravel: '🔴', nodejs: '🟢', php: '🐘', web: '🌐', generic: '📁' };

  return {
    ok: true, projectType: sections.type, icon: icons[sections.type] || '📁',
    projectName: path.basename(remotePath), rootPath: remotePath,
    configFiles, suggestedLogPaths: sections.logs,
    gitInfo, hasGit: !!(sections.git.trim()),
  };
}

function getBugCaptures() {
  try {
    return fs.readdirSync(BUGCAPTURE_DIR)
      .filter(f => f.startsWith('bug_') && f.endsWith('.md'))
      .map(f => {
        const full = path.join(BUGCAPTURE_DIR, f);
        const stat = fs.statSync(full);
        return { file: full, name: f, date: stat.mtime.toISOString(), size: stat.size };
      })
      .sort((a, b) => new Date(b.date) - new Date(a.date))
      .slice(0, 10);
  } catch { return []; }
}

function truncateBugCapture(content) {
  const idx = content.indexOf('data:image');
  if (idx === -1) return content;
  const before = content.lastIndexOf('\n', idx);
  return content.slice(0, before > 0 ? before : idx).trim() + '\n\n[...base64 images omitted...]';
}

async function handleGenerate(body) {
  const {
    title = '', scope = '', connectionId, logPath, logLines = 60,
    localGitPath, localLogPath, localLogLines = 60,
    bugCaptureFile, extraNotes, configFile, browserUrl,
  } = body;

  let gitSection = 'Not available';
  let logSection = 'Not available';
  let bugSection = 'Not available';
  let serverLabel = '—';
  let projectPath = localGitPath || '—';

  if (localGitPath) {
    try {
      const ctx = await gitContext(localGitPath);
      const parts = [];
      if (ctx.status) parts.push('**Status**\n```\n' + ctx.status + '\n```');
      if (ctx.log) parts.push('**Last 5 commits**\n```\n' + ctx.log + '\n```');
      if (ctx.diff) parts.push('**Diff stat HEAD**\n```\n' + ctx.diff + '\n```');
      gitSection = parts.length ? parts.join('\n\n') : 'Clean repository, no changes.';
    } catch (e) {
      gitSection = 'Error: ' + e.message;
    }
  }

  if (connectionId) {
    const conn = findConnection(connectionId);
    if (conn) {
      serverLabel = conn.label || conn.host;
      if (!projectPath || projectPath === '—') projectPath = conn.projectPath || '—';
      try {
        const client = await sshConnect(conn);
        const paths = logPath ? [logPath] : (conn.logPaths || []);
        const logs = [];
        for (const lp of paths) {
          try {
            const out = await sshExec(client, `tail -n ${logLines} ${lp}`);
            logs.push({ path: lp, lines: out.split('\n') });
          } catch (e) {
            logs.push({ path: lp, lines: ['Error: ' + e.message] });
          }
        }
        client.end();
        logSection = logs.length
          ? logs.map(l => `**${l.path}**\n\`\`\`\n${l.lines.join('\n')}\n\`\`\``).join('\n\n')
          : 'No log paths configured for this connection.';
      } catch (e) {
        logSection = 'SSH connection error: ' + e.message;
      }
    } else {
      logSection = 'Connection not found: ' + connectionId;
    }
  }

  let localLogSection = '';
  if (localLogPath) {
    try {
      const { execSync } = require('child_process');
      const lines = execSync(`tail -n ${Number(localLogLines) || 60} "${expandHome(localLogPath)}" 2>/dev/null`, { timeout: 5000 }).toString();
      localLogSection = lines.trim();
    } catch { localLogSection = 'Local log not readable.'; }
  }

  let configSection = '';
  if (configFile) {
    try {
      const raw = fs.readFileSync(expandHome(configFile), 'utf8').slice(0, 3000);
      configSection = '```\n' + raw + (raw.length === 3000 ? '\n[truncated...]' : '') + '\n```';
    } catch { configSection = 'File not readable.'; }
  }

  if (bugCaptureFile) {
    try {
      const raw = fs.readFileSync(bugCaptureFile, 'utf8');
      bugSection = truncateBugCapture(raw.slice(0, 8000));
    } catch (e) {
      bugSection = 'Error reading file: ' + e.message;
    }
  }

  const label = title || scope || 'Work session';
  const scopeMap = { bug: '🐛 Bug fix', feature: '✨ New feature', refactor: '🔄 Refactoring', review: '🔍 Code review' };
  const scopeLabel = scopeMap[scope] || scope || '—';

  let prompt = `# AI Briefing — ${label}\n`;
  prompt += `**Date**: ${new Date().toLocaleString('en-US')}  \n`;
  prompt += `**Scope**: ${scopeLabel}  \n`;
  prompt += `**Server**: ${serverLabel}  \n`;
  prompt += `**Project**: ${projectPath}  \n`;
  if (browserUrl) prompt += `**Browser URL**: ${browserUrl}  \n`;
  prompt += `\n---\n\n`;

  if (gitSection !== 'Not available') prompt += `## Git context\n${gitSection}\n\n---\n\n`;
  if (logSection !== 'Not available') prompt += `## Server error log (last ${logLines} lines)\n${logSection}\n\n---\n\n`;
  if (localLogSection) prompt += `## Local log (${localLogPath})\n\`\`\`\n${localLogSection}\n\`\`\`\n\n---\n\n`;
  if (configSection) prompt += `## Config file (${configFile})\n${configSection}\n\n---\n\n`;
  if (bugSection !== 'Not available') prompt += `## Recorded bug (BugCapture)\n${bugSection}\n\n---\n\n`;
  if (extraNotes) prompt += `## Additional notes\n${extraNotes}\n\n---\n\n`;

  prompt += `## Objective\n`;
  if (scope === 'bug' || !scope) prompt += `Analyze the context above and propose a solution for: **${label}**\n`;
  else if (scope === 'feature') prompt += `Help me implement the following feature in the project context: **${label}**\n`;
  else if (scope === 'refactor') prompt += `Suggest how to refactor the code in the project context: **${label}**\n`;
  else prompt += `Analyze the code and project context for: **${label}**\n`;

  return prompt;
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, { 'Access-Control-Allow-Origin': '*', 'Access-Control-Allow-Methods': 'GET, POST, OPTIONS', 'Access-Control-Allow-Headers': 'Content-Type' });
    return res.end();
  }

  const url = req.url.split('?')[0];

  if (req.method === 'GET' && url === '/health') return json(res, 200, { ok: true });
  if (req.method === 'POST' && url === '/shutdown') { json(res, 200, { ok: true }); setTimeout(() => process.exit(0), 100); return; }
  if (req.method === 'GET' && url === '/connections') return json(res, 200, readServers().map(sanitizeConnection));

  if (req.method === 'POST' && url === '/git-context') {
    let body;
    try { body = await readBody(req); } catch { return json(res, 400, { ok: false, error: 'Invalid JSON' }); }
    if (!body.localPath) return json(res, 400, { ok: false, error: 'localPath required' });
    try { return json(res, 200, { ok: true, ...(await gitContext(body.localPath)) }); }
    catch (e) { return json(res, 500, { ok: false, error: e.message }); }
  }

  if (req.method === 'POST' && url === '/scan-project') {
    let b;
    try { b = await readBody(req); } catch { return json(res, 400, { ok: false, error: 'Invalid JSON' }); }
    try {
      if (b.connectionId && b.remotePath) {
        const conn = findConnection(b.connectionId);
        if (!conn) return json(res, 404, { ok: false, error: 'Server not found' });
        return json(res, 200, await scanRemoteProject(conn, b.remotePath));
      } else if (b.localPath) {
        return json(res, 200, await scanLocalProject(b.localPath));
      } else {
        return json(res, 400, { ok: false, error: 'Provide localPath or connectionId + remotePath' });
      }
    } catch (e) { return json(res, 500, { ok: false, error: e.message }); }
  }

  if (req.method === 'GET' && url === '/bugcaptures') return json(res, 200, getBugCaptures());

  if (req.method === 'POST' && url === '/generate') {
    let body;
    try { body = await readBody(req); } catch { return json(res, 400, { ok: false, error: 'Invalid JSON' }); }
    try { return json(res, 200, { ok: true, prompt: await handleGenerate(body) }); }
    catch (e) { return json(res, 500, { ok: false, error: e.message }); }
  }

  json(res, 404, { ok: false, error: 'Not found' });
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`ContextForge → http://127.0.0.1:${PORT}`);
  console.log(`Servers config: ${SERVERS_JSON}`);
  console.log(`BugCapture dir: ${BUGCAPTURE_DIR}`);
});
