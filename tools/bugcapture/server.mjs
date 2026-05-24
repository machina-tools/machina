import http from 'http';
import { exec } from 'child_process';
import fs from 'fs';
import path from 'path';
import os from 'os';
import { pipeline, env } from '@xenova/transformers';
import { Client as SshClient } from 'ssh2';

const PORT = parseInt(process.env.BUGCAPTURE_PORT || '4327');
const SERVERS_CONFIG = process.env.SERVERS_CONFIG || path.join(os.homedir(), '.config', 'machina', 'servers.json');
const WHISPER_LANGUAGE = process.env.WHISPER_LANGUAGE || 'english';

const SESSIONS_DIR = path.join(os.tmpdir(), 'bugcapture-sessions');
env.cacheDir = process.env.WHISPER_CACHE || path.join(os.homedir(), '.cache', 'huggingface', 'hub');

function readServers() {
  try { return JSON.parse(fs.readFileSync(SERVERS_CONFIG, 'utf8')).connections || []; }
  catch { return []; }
}
function expandHome(p) { return p ? p.replace(/^~/, os.homedir()) : p; }

const loglens = {};

fs.mkdirSync(SESSIONS_DIR, { recursive: true });

let whisperPipeline = null;
let whisperLoading = false;
let whisperReady = false;

async function loadWhisper() {
  if (whisperReady || whisperLoading) return;
  whisperLoading = true;
  try {
    whisperPipeline = await pipeline('automatic-speech-recognition', 'Xenova/whisper-small', { revision: 'main' });
    whisperReady = true;
    console.log('Whisper model ready.');
  } catch (e) {
    console.error('Whisper load error:', e.message);
  }
  whisperLoading = false;
}
loadWhisper();

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Content-Length');
}
function json(res, data, code = 200) {
  cors(res);
  res.writeHead(code, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

function sh(cmd, opts = {}) {
  return new Promise((resolve, reject) => {
    exec(cmd, { maxBuffer: 100 * 1024 * 1024, ...opts }, (err, stdout, stderr) => {
      if (err) reject(new Error(stderr || err.message));
      else resolve(stdout.trim());
    });
  });
}

async function getVideoDuration(inputPath) {
  try {
    const out = await sh(`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${inputPath}"`);
    return parseFloat(out) || 0;
  } catch { return 0; }
}

async function extractFrames(inputPath, sessionDir) {
  const framesDir = path.join(sessionDir, 'frames');
  fs.mkdirSync(framesDir, { recursive: true });
  await sh(`ffmpeg -i "${inputPath}" -vf "fps=1/3" -frames:v 20 -q:v 3 "${framesDir}/frame_%04d.jpg" -y 2>/dev/null`);
  const files = fs.readdirSync(framesDir).filter(f => f.endsWith('.jpg')).sort();
  return files.map((f, i) => {
    const ts = i * 3;
    const b64 = fs.readFileSync(path.join(framesDir, f)).toString('base64');
    return { ts, file: f, b64 };
  });
}

async function extractAudio(inputPath, sessionDir) {
  const wavPath = path.join(sessionDir, 'audio.wav');
  await sh(`ffmpeg -i "${inputPath}" -ar 16000 -ac 1 -f wav "${wavPath}" -y 2>/dev/null`);
  return wavPath;
}

async function transcribeAudio(wavPath) {
  if (!whisperReady) {
    for (let i = 0; i < 60; i++) {
      await new Promise(r => setTimeout(r, 1000));
      if (whisperReady) break;
    }
  }
  if (!whisperReady) return '[Whisper not available]';
  try {
    const buf = fs.readFileSync(wavPath);
    const int16 = new Int16Array(buf.buffer, buf.byteOffset + 44, (buf.byteLength - 44) / 2);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
    const result = await whisperPipeline(float32, {
      language: WHISPER_LANGUAGE,
      task: 'transcribe',
      sampling_rate: 16000,
      return_timestamps: false,
    });
    return (result.text || '').trim();
  } catch (e) {
    console.error('Transcription error:', e.message);
    return '[Transcription error]';
  }
}

function buildLogSection(logData) {
  if (!logData || !logData.lines.length) return '';
  let s = `---\n\n## Server log (captured during recording)\n\n`;
  s += `**Server**: ${logData.serverLabel}  \n**Log**: \`${logData.logPath}\`\n\n\`\`\`\n`;
  s += logData.lines.map(l => l.text).join('\n');
  s += `\n\`\`\`\n\n`;
  return s;
}

function buildMarkdownBase64(sessionId, duration, frames, transcript, logData) {
  const dMin = Math.floor(duration / 60);
  const dSec = Math.round(duration % 60);
  const now = new Date().toLocaleString('en-US');
  let md = `# BugCapture Report\n\n`;
  md += `**Date**: ${now}  \n`;
  md += `**Duration**: ${dMin > 0 ? dMin + 'm ' : ''}${dSec}s  \n`;
  md += `**Session ID**: ${sessionId}\n\n`;
  md += `---\n\n`;
  md += `## Audio Transcript\n\n`;
  md += transcript || '_No audio detected_';
  md += `\n\n`;
  md += buildLogSection(logData);
  md += `---\n\n## Sequential Screenshots\n\n`;
  frames.forEach((f, i) => {
    const min = Math.floor(f.ts / 60);
    const sec = f.ts % 60;
    const label = min > 0 ? `${min}m ${sec}s` : `${sec}s`;
    md += `### Screenshot ${i + 1} — ${label}\n\n`;
    md += `![Screenshot ${i + 1}](data:image/jpeg;base64,${f.b64})\n\n`;
  });
  return md;
}

function buildMarkdownFolder(sessionId, duration, frames, transcript, logData) {
  const dMin = Math.floor(duration / 60);
  const dSec = Math.round(duration % 60);
  const now = new Date().toLocaleString('en-US');
  let md = `# BugCapture Report\n\n`;
  md += `**Date**: ${now}  \n`;
  md += `**Duration**: ${dMin > 0 ? dMin + 'm ' : ''}${dSec}s  \n`;
  md += `**Session ID**: ${sessionId}\n\n`;
  md += `---\n\n`;
  md += `## Audio Transcript\n\n`;
  md += transcript || '_No audio detected_';
  md += `\n\n`;
  md += buildLogSection(logData);
  md += `---\n\n## Sequential Screenshots\n\n`;
  frames.forEach((f, i) => {
    const min = Math.floor(f.ts / 60);
    const sec = f.ts % 60;
    const label = min > 0 ? `${min}m ${sec}s` : `${sec}s`;
    md += `### Screenshot ${i + 1} — ${label}\n\n`;
    md += `![Screenshot ${i + 1}](frames/${f.file})\n\n`;
  });
  return md;
}

async function buildZip(sessionDir, sessionId) {
  const zipPath = path.join(SESSIONS_DIR, `${sessionId}.zip`);
  await sh(`cd "${sessionDir}" && zip -r "${zipPath}" report_folder.md frames/ 2>/dev/null`);
  return zipPath;
}

function sshConnect(conn) {
  return new Promise((resolve, reject) => {
    const c = new SshClient();
    const cfg = { host: conn.host, port: conn.port || 22, username: conn.username, readyTimeout: 8000 };
    if (conn.auth === 'key') {
      try { cfg.privateKey = fs.readFileSync(expandHome(conn.keyFile || '~/.ssh/id_rsa')); }
      catch (e) { return reject(new Error('SSH key not found: ' + expandHome(conn.keyFile))); }
    } else {
      cfg.password = conn.password;
    }
    c.on('ready', () => resolve(c)).on('error', reject).connect(cfg);
  });
}

function startLogTail(loglensId, conn, logPath) {
  return new Promise(async (resolve, reject) => {
    try {
      const client = await sshConnect(conn);
      loglens[loglensId] = { conn: client, lines: [], active: true, serverLabel: conn.label || conn.host, logPath };
      client.exec(`tail -n 0 -f "${logPath}"`, (err, stream) => {
        if (err) { loglens[loglensId].active = false; return reject(err); }
        stream.on('data', data => {
          if (!loglens[loglensId]) return;
          data.toString().split('\n').forEach(line => {
            if (line.trim()) loglens[loglensId].lines.push({ ts: Date.now(), text: line });
          });
        });
        stream.on('close', () => { if (loglens[loglensId]) loglens[loglensId].active = false; });
        resolve({ ok: true });
      });
    } catch (e) { reject(e); }
  });
}

function stopLogTail(loglensId) {
  const state = loglens[loglensId];
  if (!state) return null;
  state.active = false;
  try { state.conn.end(); } catch {}
  const result = { serverLabel: state.serverLabel, logPath: state.logPath, lines: state.lines };
  delete loglens[loglensId];
  return result;
}

const sessions = {};

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  if (req.method === 'OPTIONS') { cors(res); res.writeHead(204); res.end(); return; }

  if (url.pathname === '/health') return json(res, { ok: true, whisperReady });

  if (url.pathname === '/shutdown' && req.method === 'POST') {
    json(res, { ok: true });
    setTimeout(() => process.exit(0), 200);
    return;
  }

  if (url.pathname === '/connections') {
    return json(res, readServers().map(c => ({ ...c, password: c.password ? '***' : '' })));
  }

  if (url.pathname === '/loglens/test' && req.method === 'POST') {
    const b = JSON.parse((await readBody(req)).toString() || '{}');
    const conn = readServers().find(c => c.id === b.connectionId);
    if (!conn) return json(res, { ok: false, error: 'Server not found' }, 404);
    try {
      const t0 = Date.now();
      const client = await sshConnect(conn);
      await new Promise((resolve, reject) => {
        client.exec('echo ok', (err, stream) => {
          if (err) return reject(err);
          stream.on('close', () => { client.end(); resolve(); });
          stream.on('data', () => {});
        });
      });
      return json(res, { ok: true, ms: Date.now() - t0 });
    } catch (e) { return json(res, { ok: false, error: e.message }); }
  }

  if (url.pathname === '/loglens/start' && req.method === 'POST') {
    const b = JSON.parse((await readBody(req)).toString() || '{}');
    const conn = readServers().find(c => c.id === b.connectionId);
    if (!conn) return json(res, { ok: false, error: 'Server not found' }, 404);
    const logPath = b.logPath || (conn.logPaths && conn.logPaths[0]);
    if (!logPath) return json(res, { ok: false, error: 'No log path configured' }, 400);
    const loglensId = b.loglensId || `ll_${Date.now()}`;
    try {
      await startLogTail(loglensId, conn, logPath);
      return json(res, { ok: true, loglensId });
    } catch (e) { return json(res, { ok: false, error: e.message }); }
  }

  const llStopM = url.pathname.match(/^\/loglens\/stop\/(\w+)$/);
  if (llStopM && req.method === 'POST') {
    const result = stopLogTail(llStopM[1]);
    if (!result) return json(res, { ok: false, error: 'LogLens session not found' }, 404);
    return json(res, { ok: true, ...result });
  }

  if (url.pathname === '/process' && req.method === 'POST') {
    try {
      const webmData = await readBody(req);
      if (!webmData.length) return json(res, { ok: false, error: 'Empty payload' }, 400);

      const sessionId = `bug_${Date.now()}`;
      const sessionDir = path.join(SESSIONS_DIR, sessionId);
      fs.mkdirSync(sessionDir, { recursive: true });

      const inputPath = path.join(sessionDir, 'recording.webm');
      fs.writeFileSync(inputPath, webmData);

      console.log(`[${sessionId}] Processing ${(webmData.length / 1024 / 1024).toFixed(1)}MB...`);

      const frames = await extractFrames(inputPath, sessionDir);
      const duration = (await getVideoDuration(inputPath)) || frames.length * 3;
      const wavPath = await extractAudio(inputPath, sessionDir);

      console.log(`[${sessionId}] ${frames.length} frames, ${duration.toFixed(1)}s, transcribing...`);
      const transcript = await transcribeAudio(wavPath);
      console.log(`[${sessionId}] Done.`);

      const loglensId = req.headers['x-loglens-id'];
      const logData = loglensId ? stopLogTail(loglensId) : null;

      const mdFolder = buildMarkdownFolder(sessionId, duration, frames, transcript, logData);
      fs.writeFileSync(path.join(sessionDir, 'report_folder.md'), mdFolder, 'utf8');

      sessions[sessionId] = { dir: sessionDir, duration, frames, transcript, logData };

      return json(res, {
        ok: true, sessionId, duration, frameCount: frames.length, transcript,
        frames: frames.map(f => ({ ts: f.ts, file: f.file, b64: f.b64 })),
        logData: logData ? { serverLabel: logData.serverLabel, logPath: logData.logPath, lineCount: logData.lines.length, lines: logData.lines.map(l => l.text) } : null,
      });
    } catch (e) {
      console.error('Process error:', e.message);
      return json(res, { ok: false, error: e.message }, 500);
    }
  }

  const zipM = url.pathname.match(/^\/zip\/(\w+)$/);
  if (zipM && req.method === 'GET') {
    const sessionId = zipM[1];
    const sess = sessions[sessionId];
    if (!sess) return json(res, { ok: false, error: 'Session not found' }, 404);
    try {
      const zipPath = await buildZip(sess.dir, sessionId);
      const data = fs.readFileSync(zipPath);
      cors(res);
      res.writeHead(200, { 'Content-Type': 'application/zip', 'Content-Disposition': `attachment; filename="${sessionId}.zip"`, 'Content-Length': data.length });
      res.end(data);
    } catch (e) {
      return json(res, { ok: false, error: e.message }, 500);
    }
    return;
  }

  cors(res); res.writeHead(404); res.end('{}');
}).listen(PORT, '127.0.0.1', () => {
  console.log(`BugCapture → http://127.0.0.1:${PORT}`);
  console.log(`Whisper language: ${WHISPER_LANGUAGE}`);
  console.log(`Servers config: ${SERVERS_CONFIG}`);
});
