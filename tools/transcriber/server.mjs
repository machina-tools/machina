import { createServer } from 'http';
import { exec }          from 'child_process';
import { writeFileSync, unlinkSync, readFileSync } from 'fs';
import { tmpdir }        from 'os';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { pipeline, env } from '@xenova/transformers';

const __dirname = dirname(fileURLToPath(import.meta.url));

const PORT     = parseInt(process.env.TRANSCRIBER_PORT     || '4324');
const MODEL    = process.env.TRANSCRIBER_MODEL             || 'Xenova/whisper-base';
const LANGUAGE = process.env.TRANSCRIBER_LANGUAGE          || 'english';

// Store models inside the tool directory so they survive npm installs
env.cacheDir = join(__dirname, 'models');
env.allowLocalModels = false;

let whisperPipe = null;
let loadErr     = null;

async function getPipeline() {
  if (whisperPipe) return whisperPipe;
  if (loadErr) throw new Error(loadErr);
  console.log(`[Transcriber] Loading model ${MODEL}…`);
  whisperPipe = await pipeline('automatic-speech-recognition', MODEL, { quantized: true });
  console.log('[Transcriber] Model ready');
  return whisperPipe;
}

// Pre-load at startup so the first transcription has no extra wait
getPipeline().catch(e => { loadErr = e.message; console.error('[Transcriber] Model load error:', e.message); });

function toWav(inPath, outPath) {
  return new Promise((resolve, reject) =>
    exec(`ffmpeg -y -i "${inPath}" -ar 16000 -ac 1 -f wav "${outPath}" 2>/dev/null`,
      err => err ? reject(new Error('ffmpeg: ' + err.message)) : resolve()));
}

// Read a 16-bit PCM WAV and return Float32Array in [-1, 1]
// (AudioContext is not available in Node — parse raw PCM directly)
function readWavPCM(wavPath) {
  const buf = readFileSync(wavPath);
  let offset = 12;
  while (offset + 8 <= buf.length) {
    const id   = buf.slice(offset, offset + 4).toString('ascii');
    const size = buf.readUInt32LE(offset + 4);
    if (id === 'data') {
      const nSamples = Math.floor(size / 2);
      const float32  = new Float32Array(nSamples);
      for (let i = 0; i < nSamples; i++)
        float32[i] = buf.readInt16LE(offset + 8 + i * 2) / 32768.0;
      return float32;
    }
    offset += 8 + size + (size & 1);
  }
  throw new Error('"data" chunk not found in WAV file');
}

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
}

function reply(res, data) {
  cors(res);
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(data));
}

createServer(async (req, res) => {
  if (req.method === 'OPTIONS') { cors(res); res.writeHead(204); res.end(); return; }

  if (req.url === '/health')
    return reply(res, { ok: true, ready: !!whisperPipe, model: MODEL, language: LANGUAGE });

  if (req.url === '/shutdown' && req.method === 'POST') {
    reply(res, { ok: true });
    setTimeout(() => process.exit(0), 100);
    return;
  }

  if (req.url === '/transcribe' && req.method === 'POST') {
    const chunks = [];
    req.on('data', c => chunks.push(c));
    req.on('end', async () => {
      const mime  = (req.headers['content-type'] || 'audio/webm').split(';')[0].trim();
      const ext   = mime.includes('ogg') ? 'ogg' : 'webm';
      const tmpIn  = join(tmpdir(), `transcriber_${Date.now()}.${ext}`);
      const tmpWav = tmpIn + '.wav';
      try {
        writeFileSync(tmpIn, Buffer.concat(chunks));
        await toWav(tmpIn, tmpWav);
        const pcm  = readWavPCM(tmpWav);
        const pipe = await getPipeline();
        const out  = await pipe(pcm, { sampling_rate: 16000, language: LANGUAGE, task: 'transcribe' });
        reply(res, { ok: true, text: (out.text || '').trim() });
      } catch (err) {
        reply(res, { ok: false, error: err.message });
      } finally {
        try { unlinkSync(tmpIn);  } catch {}
        try { unlinkSync(tmpWav); } catch {}
      }
    });
    return;
  }

  cors(res); res.writeHead(404); res.end('{}');

}).listen(PORT, '127.0.0.1', () => {
  console.log(`[Transcriber] http://127.0.0.1:${PORT}`);
  console.log(`[Transcriber] Model: ${MODEL} · Language: ${LANGUAGE}`);
  console.log(`[Transcriber] Used by: PromptBoard, BugCapture Web, LearnBoard Web`);
});
