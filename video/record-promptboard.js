const { chromium } = require('playwright');
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const HTML_FILE = path.resolve(__dirname, 'promptboard-autoplay.html');
const OUT_DIR   = __dirname;
const WEBM_OUT  = path.join(OUT_DIR, 'promptboard-video.webm');
const MP4_OUT   = path.join(OUT_DIR, 'promptboard-video.mp4');
const VIDEO_DURATION_MS = 76000; // 68s + 8s buffer

(async () => {
  console.log('Launching browser...');
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 1,
    recordVideo: { dir: OUT_DIR, size: { width: 1920, height: 1080 } },
  });
  const page = await context.newPage();
  console.log('Loading HTML...');
  await page.goto(`file://${HTML_FILE}`);
  await page.waitForSelector('#play-btn', { timeout: 10000 });
  console.log('Clicking PLAY...');
  await page.click('#play-btn');
  console.log(`Recording for ${VIDEO_DURATION_MS / 1000}s...`);
  await page.waitForTimeout(VIDEO_DURATION_MS);
  console.log('Saving WebM...');
  const video = await page.video();
  await context.close();
  await browser.close();
  const savedPath = await video.path();
  fs.renameSync(savedPath, WEBM_OUT);
  console.log('Converting to MP4...');
  execSync(
    `ffmpeg -y -i "${WEBM_OUT}" -c:v libopenh264 -b:v 4M -pix_fmt yuv420p "${MP4_OUT}"`,
    { stdio: 'inherit' }
  );
  console.log(`\n✓ Done: ${MP4_OUT}`);
  fs.unlinkSync(WEBM_OUT);
})();
