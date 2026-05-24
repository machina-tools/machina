'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');
const os = require('os');

const PORT = parseInt(process.env.LEARNBOARD_PORT || '4331');
const LEARNING_FILE = process.env.LEARNING_FILE || path.join(process.cwd(), 'LEARNING.md');
const BACKUP_DIR = process.env.LEARNBOARD_BACKUP_DIR || path.join(path.dirname(LEARNING_FILE), '.learnboard-backups');

fs.mkdirSync(BACKUP_DIR, { recursive: true });

// ─── MARKDOWN TABLE PARSER ─────────────────────────────────────────────────────

function parseTableBlock(lines) {
  if (lines.length < 3) return { headers: [], rows: [] };
  const headers = lines[0].split('|').slice(1, -1).map(c => c.trim());
  const rows = lines.slice(2).map(line => line.split('|').slice(1, -1).map(c => c.trim()));
  return { headers, rows };
}

function extractSectionTable(content, sectionMarker) {
  const sIdx = content.indexOf(sectionMarker);
  if (sIdx === -1) return null;

  const fromSec = content.slice(sIdx + sectionMarker.length);
  const allLines = fromSec.split('\n');
  const tableLines = [];
  let tableStartOffset = 0;
  let inTable = false;
  let charCount = sectionMarker.length;

  for (let i = 0; i < allLines.length; i++) {
    const line = allLines[i];
    const trimmed = line.trim();
    if (trimmed.startsWith('|')) {
      if (!inTable) { tableStartOffset = charCount; inTable = true; }
      tableLines.push(line);
    } else if (inTable) {
      break;
    } else if (trimmed.startsWith('##') && i > 0) {
      break;
    }
    charCount += line.length + 1;
  }

  if (tableLines.length < 3) return null;
  const { headers, rows } = parseTableBlock(tableLines);
  const absStart = sIdx + tableStartOffset;
  const absEnd = absStart + tableLines.join('\n').length;
  return { headers, rows, absStart, absEnd };
}

function serializeTable(headers, rows) {
  const widths = headers.map((h, i) => Math.max(h.length, 3, ...rows.map(r => (r[i] || '').length)));
  const pad = (s, w) => String(s == null ? '' : s).padEnd(w);
  const header = '| ' + headers.map((h, i) => pad(h, widths[i])).join(' | ') + ' |';
  const sep = '|' + widths.map(w => '-'.repeat(w + 2)).join('|') + '|';
  const data = rows.map(row => '| ' + headers.map((_, i) => pad(row[i] || '', widths[i])).join(' | ') + ' |');
  return [header, sep, ...data].join('\n');
}

// ─── SECTIONS ─────────────────────────────────────────────────────────────────
// Markers are read from env vars so users can customize section names.
// Defaults match the LEARNING.md template included with this repo.

const SECTIONS = {
  requests: {
    marker: process.env.SECTION_REQUESTS || '### User Requests',
    label: 'User Requests',
    icon: '📋',
  },
  suggestions: {
    marker: process.env.SECTION_SUGGESTIONS || '### Suggestions Log',
    label: 'Suggestions Log',
    icon: '💡',
  },
  lessons: {
    marker: process.env.SECTION_LESSONS || '### Lessons Learned',
    label: 'Lessons Learned',
    icon: '🎓',
  },
  observations: {
    marker: process.env.SECTION_OBSERVATIONS || '### AI Observations',
    label: 'AI Observations',
    icon: '🧠',
  },
};

function parseLearningFile() {
  const content = fs.readFileSync(LEARNING_FILE, 'utf8');
  const result = { sections: {}, raw: content };
  for (const [key, sec] of Object.entries(SECTIONS)) {
    const parsed = extractSectionTable(content, sec.marker);
    result.sections[key] = parsed
      ? { headers: parsed.headers, rows: parsed.rows, label: sec.label, icon: sec.icon }
      : { headers: [], rows: [], label: sec.label, icon: sec.icon };
  }
  return result;
}

function saveLearningFile(updates) {
  let content = fs.readFileSync(LEARNING_FILE, 'utf8');

  // Backup before writing
  const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
  fs.writeFileSync(path.join(BACKUP_DIR, `LEARNING_${ts}.md`), content, 'utf8');

  for (const [key, { rows }] of Object.entries(updates)) {
    const sec = SECTIONS[key];
    if (!sec) continue;
    const parsed = extractSectionTable(content, sec.marker);
    if (!parsed) continue;
    const newTable = serializeTable(parsed.headers, rows);
    content = content.slice(0, parsed.absStart) + newTable + content.slice(parsed.absEnd);
  }

  fs.writeFileSync(LEARNING_FILE, content, 'utf8');
}

// ─── STATS ────────────────────────────────────────────────────────────────────

function computeStats(data) {
  const sug = data.sections.suggestions;
  const les = data.sections.lessons;
  const req = data.sections.requests;
  const obs = data.sections.observations;

  const voteCol = sug.headers.indexOf('Vote');
  const implCol = sug.headers.indexOf('Implemented');

  const votes = { yes: 0, no: 0, warm: 0, rejected: 0 };
  let implemented = 0;

  for (const row of sug.rows) {
    const vote = (row[voteCol] || '').trim();
    const impl = (row[implCol] || '').toLowerCase().trim();
    if (impl === 'yes') implemented++;
    if (vote === '✗') votes.rejected++;
    else if (parseFloat(vote) >= 4) votes.yes++;
    else if (parseFloat(vote) >= 2.5) votes.warm++;
    else if (parseFloat(vote) > 0) votes.no++;
  }

  return {
    totalRequests: req.rows.length,
    totalLessons: les.rows.length,
    totalSuggestions: sug.rows.length,
    implemented,
    implementedPct: sug.rows.length ? Math.round(implemented / sug.rows.length * 100) : 0,
    totalObservations: obs.rows.length,
    votes,
  };
}

// ─── HTTP SERVER ───────────────────────────────────────────────────────────────

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

function respond(res, data, status = 200) {
  cors(res);
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

function bodyJson(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}')); } catch { reject(new Error('Invalid JSON')); } });
    req.on('error', reject);
  });
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') { cors(res); res.writeHead(204); res.end(); return; }

  const url = new URL(req.url, `http://localhost:${PORT}`);
  const p = url.pathname;

  if (req.method === 'GET' && p === '/health') return respond(res, { ok: true });
  if (req.method === 'POST' && p === '/shutdown') { respond(res, { ok: true }); setTimeout(() => process.exit(0), 100); return; }

  if (req.method === 'GET' && p === '/data') {
    try {
      const data = parseLearningFile();
      return respond(res, { ...data, stats: computeStats(data) });
    } catch (e) { return respond(res, { error: e.message }, 500); }
  }

  if (req.method === 'POST' && p === '/save') {
    try {
      const { updates } = await bodyJson(req);
      saveLearningFile(updates);
      return respond(res, { ok: true });
    } catch (e) { return respond(res, { error: e.message }, 500); }
  }

  if (req.method === 'GET' && p === '/raw') {
    try {
      const content = fs.readFileSync(LEARNING_FILE, 'utf8');
      cors(res);
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end(content);
    } catch (e) { return respond(res, { error: e.message }, 500); }
    return;
  }

  if (req.method === 'GET' && p === '/backups') {
    try {
      const files = fs.readdirSync(BACKUP_DIR).filter(f => f.endsWith('.md')).sort().reverse().slice(0, 20);
      return respond(res, files);
    } catch { return respond(res, []); }
  }

  respond(res, { error: 'Not found' }, 404);
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`LearnBoard → http://127.0.0.1:${PORT}`);
  console.log(`Learning file: ${LEARNING_FILE}`);
  console.log(`Backups: ${BACKUP_DIR}`);
});
