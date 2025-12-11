// Multi-threading and clustering for high performance
import cluster from 'cluster';
import os from 'os';
import { Worker } from 'worker_threads';

import express from 'express';
import compression from 'compression';
import http from 'http';
import baileys from '@whiskeysockets/baileys';
const {
  makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  downloadMediaMessage,
  getContentType,
  extractMessageContent,
  delay,
  jidNormalizedUser,
  areJidsSameUser,
  jidDecode,
} = baileys;
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import axios from 'axios';
import qrcode from 'qrcode-terminal';
import QRCode from 'qrcode';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import winston from 'winston';
import { initializeDependencies, enhancedInitialSync } from './loadChatUtils.js';
import PersistentStorage from './persistentStorage.js';
import sharp from 'sharp';
import ffmpeg from 'ffmpeg-static';
import { exec, spawn } from 'child_process';
import iconv from 'iconv-lite';

// ESM __dirname replacement
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ============ PRODUCTION-GRADE HTTP CLIENT CONFIGURATION ============
// Configure axios for optimal performance with connection pooling
const axiosAgent = new http.Agent({
  keepAlive: true,
  keepAliveMsecs: 30000,
  maxSockets: 50,
  maxFreeSockets: 10,
  timeout: 60000,
  scheduling: 'lifo' // Last In First Out for better performance
});

axios.defaults.httpAgent = axiosAgent;
axios.defaults.timeout = 30000;
axios.defaults.maxRedirects = 5;

// Configurazione Whisper e TTS
let transcriber = null;
let transcriptionEnabled = false;
let ttsModel = null;
let ttsSpeakerEmbeddings = null;
let ttsEnabled = false;

// Inizializza il modello Whisper con import dinamico
// Using whisper-base for better accuracy with Italian and English
async function initWhisperModel() {
  try {
    console.log("Initializing Whisper model (base - better accuracy)...");

    // Usa import dinamico per caricare il modulo ESM
    const { pipeline } = await import("@xenova/transformers");

    // Using whisper-base for better transcription quality
    // whisper-tiny: 39MB, lower accuracy
    // whisper-base: 74MB, much better accuracy for Italian & English
    transcriber = await pipeline(
      "automatic-speech-recognition",
      "Xenova/whisper-base",
      {
        // ⚡ 74MB - better accuracy than tiny (39MB)
        quantized: true, // 🔥 Usa 8-bit per ridurre RAM
        local_files_only: false,
      }
    );

    transcriptionEnabled = true;
    console.log("✓ Whisper model loaded successfully (base model - high accuracy)");
    console.log("✓ Supported languages: Italian (it) + English (en) with auto-detection");
  } catch (error) {
    console.error("Whisper model initialization failed:", error);
    transcriptionEnabled = false;
  }
}

// Check if espeak is available (local TTS for Raspberry Pi)
async function initTTSModel() {
  try {
    console.log("Checking for local TTS engine (espeak)...");

    // Check if espeak is installed
    await new Promise((resolve, reject) => {
      exec('which espeak || which espeak-ng', (error, stdout) => {
        if (error || !stdout.trim()) {
          reject(new Error('espeak not found. Install with: sudo apt-get install espeak'));
        } else {
          resolve(stdout.trim());
        }
      });
    });

    ttsEnabled = true;
    console.log("✓ Local TTS ready (espeak) - English & Italian, fully offline");
  } catch (error) {
    console.error("⚠ Local TTS not available:", error.message);
    console.error("Install espeak: sudo apt-get install espeak");
    ttsEnabled = false;
  }
}

// Funzione di trascrizione con Whisper - IMPROVED VERSION
// Supports Italian, English, and auto-detection for better accuracy
async function transcribeAudioWithWhisper(audioBuffer, language = 'auto') {
  if (!transcriptionEnabled || !transcriber) {
    return "Trascrizione non disponibile (Whisper model not loaded)";
  }

  const tempOutput = path.join(__dirname, `temp_${Date.now()}.wav`);

  try {
    console.log(`Converting audio to WAV format (language: ${language})...`);

    // Usa ffmpeg-static
    const ffmpegPath = ffmpeg;

    await new Promise((resolve, reject) => {
      const process = spawn(ffmpegPath, [
        "-i",
        "pipe:0", // Input da stdin
        "-ar",
        "16000", // Sample rate 16kHz
        "-ac",
        "1", // Mono
        "-c:a",
        "pcm_s16le", // 16-bit PCM (signed little-endian)
        "-f",
        "wav", // Formato WAV
        "-", // Output su stdout
      ]);

      const outputStream = fs.createWriteStream(tempOutput);

      process.stdin.end(audioBuffer);
      process.stdout.pipe(outputStream);

      process.on("error", reject);
      outputStream.on("finish", resolve);
    });

    console.log("Converting WAV to Float32Array...");

    // Leggi il file WAV e converti in Float32Array
    const wavBuffer = await fs.promises.readFile(tempOutput);

    // Estrai i dati audio PCM dal file WAV
    const float32Array = wavBufferToFloat32Array(wavBuffer);

    console.log(`Transcribing with Whisper (base model, language: ${language})...`);

    // Configure transcription options based on language
    const transcriptionOptions = {
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: false,
    };

    // Language selection: 'auto', 'it' (Italian), or 'en' (English)
    if (language === 'auto') {
      // Auto-detect language (Whisper will choose the best match)
      console.log("Using automatic language detection (Italian/English)");
      // Don't specify language - let Whisper detect it
    } else if (language === 'it' || language === 'en') {
      // Explicit language selection for better accuracy
      transcriptionOptions.language = language;
      console.log(`Using explicit language: ${language === 'it' ? 'Italian' : 'English'}`);
    } else {
      // Fallback to Italian for unknown languages
      console.log(`Unknown language '${language}', defaulting to Italian`);
      transcriptionOptions.language = 'it';
    }

    const result = await transcriber(float32Array, transcriptionOptions);

    const transcription = result.text || "Nessuna trascrizione disponibile";

    // Clean up transcription (remove extra whitespace)
    const cleanedTranscription = transcription.trim();

    console.log(`✓ Whisper transcription successful (${cleanedTranscription.length} characters)`);
    console.log(`Transcribed text: "${cleanedTranscription.substring(0, 100)}${cleanedTranscription.length > 100 ? '...' : ''}"`);

    return cleanedTranscription;
  } catch (error) {
    console.error("Transcription failed:", error);
    return "Errore nella trascrizione";
  } finally {
    try {
      if (fs.existsSync(tempOutput)) {
        await fs.promises.unlink(tempOutput);
        console.log("Temp file cleaned up successfully");
      }
    } catch (cleanupError) {
      console.error("Cleanup failed:", cleanupError);
    }
  }
}

// Funzione per convertire WAV buffer in Float32Array
function wavBufferToFloat32Array(wavBuffer) {
  // WAV file header is 44 bytes
  const dataOffset = 44;
  const audioData = wavBuffer.slice(dataOffset);

  // I dati sono in formato PCM 16-bit signed little-endian
  const int16Array = new Int16Array(
    audioData.buffer,
    audioData.byteOffset,
    audioData.byteLength / 2
  );

  // Converti Int16 a Float32 (range -1.0 to 1.0)
  const float32Array = new Float32Array(int16Array.length);
  for (let i = 0; i < int16Array.length; i++) {
    float32Array[i] = int16Array[i] / 32768.0; // 32768 = 2^15
  }

  return float32Array;
}
// Inizializza i modelli all'avvio
initWhisperModel();
initTTSModel();

// ============ VIDEO FRAME EXTRACTION ============
// Cache directory for video frames
const VIDEO_FRAMES_DIR = path.join(__dirname, "video_frames_cache");
if (!fs.existsSync(VIDEO_FRAMES_DIR)) {
  fs.mkdirSync(VIDEO_FRAMES_DIR, { recursive: true });
}

// Extract frames from video at 1 FPS
async function extractVideoFrames(videoBuffer, messageId) {
  const framesDir = path.join(VIDEO_FRAMES_DIR, messageId);

  // Check if frames already exist
  if (fs.existsSync(framesDir)) {
    const existingFrames = fs.readdirSync(framesDir).filter(f => f.endsWith('.png'));
    if (existingFrames.length > 0) {
      console.log(`Using cached frames for ${messageId}: ${existingFrames.length} frames`);
      return { frameCount: existingFrames.length, framesDir };
    }
  }

  // Create frames directory
  if (!fs.existsSync(framesDir)) {
    fs.mkdirSync(framesDir, { recursive: true });
  }

  const tempVideoPath = path.join(__dirname, `temp_video_${messageId}_${Date.now()}.mp4`);
  const framePattern = path.join(framesDir, 'frame_%04d.png');

  try {
    // Write video buffer to temp file
    await fs.promises.writeFile(tempVideoPath, videoBuffer);
    console.log(`Extracting frames from video ${messageId} at 1 FPS...`);

    // Extract frames using FFmpeg at 1 FPS
    await new Promise((resolve, reject) => {
      const ffmpegPath = ffmpeg;
      const process = spawn(ffmpegPath, [
        '-i', tempVideoPath,
        '-vf', 'fps=1',  // 1 frame per second
        '-s', '128x128',  // Small size for WAP devices
        '-q:v', '2',      // High quality
        framePattern
      ]);

      let stderr = '';
      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`FFmpeg exited with code ${code}: ${stderr}`));
        }
      });

      process.on('error', reject);
    });

    // Count extracted frames
    const frames = await fs.promises.readdir(framesDir);
    const pngFrames = frames.filter(f => f.endsWith('.png'));
    console.log(`Extracted ${pngFrames.length} frames from video ${messageId}`);

    return { frameCount: pngFrames.length, framesDir };
  } catch (error) {
    console.error('Frame extraction error:', error);
    throw error;
  } finally {
    // Clean up temp video file
    try {
      if (fs.existsSync(tempVideoPath)) {
        await fs.promises.unlink(tempVideoPath);
        console.log('Temp video file cleaned up');
      }
    } catch (cleanupError) {
      console.error('Temp video cleanup failed:', cleanupError);
    }
  }
}

// Clean up old video frames (older than 1 hour)
function cleanupOldVideoFrames() {
  try {
    if (!fs.existsSync(VIDEO_FRAMES_DIR)) return;

    const oneHourAgo = Date.now() - (60 * 60 * 1000);
    const dirs = fs.readdirSync(VIDEO_FRAMES_DIR);

    for (const dir of dirs) {
      const dirPath = path.join(VIDEO_FRAMES_DIR, dir);
      const stats = fs.statSync(dirPath);

      if (stats.isDirectory() && stats.mtimeMs < oneHourAgo) {
        console.log(`Cleaning up old video frames: ${dir}`);
        fs.rmSync(dirPath, { recursive: true, force: true });
      }
    }
  } catch (error) {
    console.error('Error cleaning up video frames:', error);
  }
}

// Run cleanup every 30 minutes
setInterval(cleanupOldVideoFrames, 30 * 60 * 1000);

// ============ TEXT-TO-SPEECH ============
// Local TTS using espeak (English and Italian, offline)
async function textToSpeech(text, language = 'en') {
  try {
    console.log(`TTS request: "${text.substring(0, 50)}..." (language: ${language})`);

    if (!ttsEnabled) {
      throw new Error('Local TTS not available. Install espeak: sudo apt-get install espeak');
    }

    // Support only English and Italian
    if (language !== 'en' && language !== 'it') {
      console.log(`⚠ Only English (en) and Italian (it) supported. Using English for ${language}.`);
      language = 'en';
    }

    return await textToSpeechLocal(text, language);
  } catch (error) {
    console.error('TTS conversion error:', error.message);
    throw new Error(`TTS conversion failed: ${error.message}`);
  }
}

// Local TTS using espeak (Raspberry Pi compatible, English and Italian, offline)
async function textToSpeechLocal(text, language = 'en') {
  const tempWav = path.join(__dirname, `temp_espeak_${Date.now()}.wav`);

  try {
    const langName = language === 'it' ? 'Italian' : 'English';
    console.log(`Generating speech with espeak (${langName}, local, offline)...`);

    // Use espeak to generate WAV audio
    await new Promise((resolve, reject) => {
      // espeak command: text to WAV file
      // -v: Voice (en for English, it for Italian)
      // -s 150: Speed (words per minute)
      // -a 200: Amplitude (volume)
      // -w: Write to WAV file
      const espeak = spawn('espeak', [
        '-v', language,       // Voice: 'en' or 'it'
        '-s', '150',          // Speed: 150 wpm (natural)
        '-a', '200',          // Volume: maximum
        '-w', tempWav,        // Output WAV file
        text
      ]);

      let stderr = '';
      espeak.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      espeak.on('close', (code) => {
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`espeak failed with code ${code}: ${stderr}`));
        }
      });

      espeak.on('error', (err) => {
        reject(new Error(`espeak execution failed: ${err.message}`));
      });
    });

    // Read the generated WAV file
    const wavBuffer = await fs.promises.readFile(tempWav);

    console.log(`✓ Generated ${wavBuffer.length} bytes of audio (WAV)`);

    return wavBuffer;
  } catch (error) {
    console.error('espeak TTS error:', error.message);
    throw error;
  } finally {
    // Cleanup temp file
    try {
      if (fs.existsSync(tempWav)) {
        await fs.promises.unlink(tempWav);
      }
    } catch (cleanupError) {
      console.error('Cleanup error:', cleanupError);
    }
  }
}

// Google TTS (for non-English languages or fallback)
async function textToSpeechGoogle(text, language = 'en') {
  try {
    // Google Translate TTS API endpoint
    const ttsUrl = `https://translate.google.com/translate_tts`;

    // Split text into chunks if longer than 200 characters (Google TTS limit)
    const maxLength = 200;
    const chunks = [];

    if (text.length <= maxLength) {
      chunks.push(text);
    } else {
      // Split by sentences
      const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
      let currentChunk = '';

      for (const sentence of sentences) {
        if ((currentChunk + sentence).length <= maxLength) {
          currentChunk += sentence;
        } else {
          if (currentChunk) chunks.push(currentChunk.trim());
          currentChunk = sentence;
        }
      }
      if (currentChunk) chunks.push(currentChunk.trim());
    }

    console.log(`Split into ${chunks.length} chunk(s) for Google TTS`);

    // Fetch audio for each chunk
    const audioBuffers = [];
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      console.log(`Fetching Google TTS audio for chunk ${i + 1}/${chunks.length}`);

      const response = await axios.get(ttsUrl, {
        params: {
          ie: 'UTF-8',
          tl: language,
          client: 'tw-ob',
          q: chunk,
          textlen: chunk.length
        },
        responseType: 'arraybuffer',
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
          'Referer': 'https://translate.google.com/'
        },
        timeout: 30000, // 30 second timeout
        maxContentLength: 10 * 1024 * 1024, // 10MB max
        maxBodyLength: 10 * 1024 * 1024
      });

      audioBuffers.push(Buffer.from(response.data));
    }

    // Concatenate audio buffers if multiple chunks
    if (audioBuffers.length === 1) {
      return audioBuffers[0];
    } else {
      // Simple concatenation works for MP3 files
      return Buffer.concat(audioBuffers);
    }
  } catch (error) {
    console.error('Google TTS error:', error.message);
    throw error;
  }
}

// Helper function to convert Float32Array audio to WAV buffer
async function convertToWav(audioData, samplingRate = 16000) {
  // Create WAV header
  const numChannels = 1;
  const bitsPerSample = 16;
  const byteRate = samplingRate * numChannels * bitsPerSample / 8;
  const blockAlign = numChannels * bitsPerSample / 8;
  const dataSize = audioData.length * 2; // 16-bit = 2 bytes per sample

  const buffer = Buffer.alloc(44 + dataSize);

  // RIFF header
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write('WAVE', 8);

  // fmt chunk
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16); // fmt chunk size
  buffer.writeUInt16LE(1, 20); // audio format (1 = PCM)
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(samplingRate, 24);
  buffer.writeUInt32LE(byteRate, 28);
  buffer.writeUInt16LE(blockAlign, 32);
  buffer.writeUInt16LE(bitsPerSample, 34);

  // data chunk
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataSize, 40);

  // Convert float32 to int16
  for (let i = 0; i < audioData.length; i++) {
    const sample = Math.max(-1, Math.min(1, audioData[i]));
    const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    buffer.writeInt16LE(int16, 44 + i * 2);
  }

  return buffer;
}

// Helper function to concatenate multiple audio files using FFmpeg
async function concatenateAudioFiles(audioBuffers) {
  const tempFiles = [];
  const tempOutput = path.join(__dirname, `temp_concat_${Date.now()}.wav`);
  const concatList = path.join(__dirname, `concat_list_${Date.now()}.txt`);

  try {
    // Save each buffer to a temporary file (async)
    for (let i = 0; i < audioBuffers.length; i++) {
      const tempFile = path.join(__dirname, `temp_chunk_${Date.now()}_${i}.wav`);
      await fs.promises.writeFile(tempFile, audioBuffers[i]);
      tempFiles.push(tempFile);
    }

    // Create concat file list (async)
    const listContent = tempFiles.map(f => `file '${f}'`).join('\n');
    await fs.promises.writeFile(concatList, listContent);

    // Use FFmpeg to concatenate
    await new Promise((resolve, reject) => {
      const ffmpegPath = ffmpeg;
      const process = spawn(ffmpegPath, [
        '-f', 'concat',
        '-safe', '0',
        '-i', concatList,
        '-c', 'copy',
        tempOutput
      ]);

      process.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`FFmpeg concat failed with code ${code}`));
      });
    });

    // Read concatenated file (async)
    const concatenated = await fs.promises.readFile(tempOutput);

    return concatenated;
  } catch (error) {
    console.error('Audio concatenation error:', error);
    throw error;
  } finally {
    // Cleanup all temp files (async)
    try {
      for (const f of tempFiles) {
        if (fs.existsSync(f)) await fs.promises.unlink(f);
      }
      if (fs.existsSync(concatList)) await fs.promises.unlink(concatList);
      if (fs.existsSync(tempOutput)) await fs.promises.unlink(tempOutput);
    } catch (cleanupError) {
      console.error('Cleanup error in concatenateAudioFiles:', cleanupError);
    }
  }
}

// Helper function to convert WAV to OGG Opus format
async function convertWavToOgg(wavBuffer) {
  const tempWav = path.join(__dirname, `temp_tts_${Date.now()}.wav`);
  const tempOgg = path.join(__dirname, `temp_tts_${Date.now()}.ogg`);

  try {
    // Write WAV to temp file (async)
    await fs.promises.writeFile(tempWav, wavBuffer);

    // Convert to OGG Opus using FFmpeg
    await new Promise((resolve, reject) => {
      const ffmpegPath = ffmpeg;
      const process = spawn(ffmpegPath, [
        '-i', tempWav,
        '-c:a', 'libopus',      // Use Opus codec
        '-b:a', '64k',          // 64kbps bitrate (good for voice)
        '-vbr', 'on',           // Variable bitrate
        '-compression_level', '10', // Max compression
        '-frame_duration', '60', // 60ms frames (good for voice)
        '-application', 'voip',  // Optimize for voice
        tempOgg
      ]);

      let stderr = '';
      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code === 0) resolve();
        else reject(new Error(`FFmpeg conversion failed with code ${code}: ${stderr}`));
      });
    });

    // Read OGG file (async)
    const oggBuffer = await fs.promises.readFile(tempOgg);

    return oggBuffer;
  } catch (error) {
    console.error('WAV to OGG conversion error:', error);
    throw error;
  } finally {
    // Cleanup temp files (async)
    try {
      if (fs.existsSync(tempWav)) await fs.promises.unlink(tempWav);
      if (fs.existsSync(tempOgg)) await fs.promises.unlink(tempOgg);
    } catch (cleanupError) {
      console.error('Cleanup error in convertWavToOgg:', cleanupError);
    }
  }
}

const app = express();
const port = process.env.PORT || 3500;
const isDev = process.env.NODE_ENV !== "production";
let sock = null;

// ============ PRODUCTION-GRADE PERFORMANCE MONITORING ============
const performanceMetrics = {
  requests: { total: 0, success: 0, errors: 0 },
  responseTime: { total: 0, count: 0, min: Infinity, max: 0 },
  cache: { hits: 0, misses: 0 },
  startTime: Date.now(),

  recordRequest(success, responseTime) {
    this.requests.total++;
    if (success) this.requests.success++;
    else this.requests.errors++;

    this.responseTime.total += responseTime;
    this.responseTime.count++;
    this.responseTime.min = Math.min(this.responseTime.min, responseTime);
    this.responseTime.max = Math.max(this.responseTime.max, responseTime);
  },

  getStats() {
    const uptime = Date.now() - this.startTime;
    const avgResponseTime = this.responseTime.count > 0
      ? (this.responseTime.total / this.responseTime.count).toFixed(2)
      : 0;
    const successRate = this.requests.total > 0
      ? ((this.requests.success / this.requests.total) * 100).toFixed(2)
      : 100;
    const cacheHitRate = (this.cache.hits + this.cache.misses) > 0
      ? ((this.cache.hits / (this.cache.hits + this.cache.misses)) * 100).toFixed(2)
      : 0;

    return {
      uptime: `${Math.floor(uptime / 1000)} seconds`,
      requests: this.requests,
      responseTime: {
        avg: `${avgResponseTime}ms`,
        min: `${this.responseTime.min === Infinity ? 0 : this.responseTime.min}ms`,
        max: `${this.responseTime.max}ms`
      },
      successRate: `${successRate}%`,
      cache: {
        ...this.cache,
        hitRate: `${cacheHitRate}%`
      },
      memory: {
        rss: `${(process.memoryUsage().rss / 1024 / 1024).toFixed(2)} MB`,
        heapUsed: `${(process.memoryUsage().heapUsed / 1024 / 1024).toFixed(2)} MB`
      }
    };
  }
};

// ============ PRODUCTION-GRADE MIDDLEWARE STACK ============

// 1. COMPRESSION - Gzip/Brotli for 70-90% size reduction
app.use(compression({
  level: 6, // Balance between speed and compression
  threshold: 1024, // Only compress responses > 1KB
  filter: (req, res) => {
    if (req.headers['x-no-compression']) return false;
    return compression.filter(req, res);
  }
}));

// 2. SECURITY - Production-grade helmet configuration
app.use(
  helmet({
    contentSecurityPolicy: false, // Disabled for WML compatibility
    frameguard: { action: "deny" },
    hsts: {
      maxAge: 31536000, // 1 year
      includeSubDomains: true,
      preload: true
    },
    noSniff: true,
    xssFilter: true
  })
);

// 3. PERFORMANCE TRACKING - Request timing middleware
app.use((req, res, next) => {
  const startTime = Date.now();

  res.on('finish', () => {
    const responseTime = Date.now() - startTime;
    const success = res.statusCode < 400;
    performanceMetrics.recordRequest(success, responseTime);

    // Log slow requests (> 1 second)
    if (responseTime > 1000) {
      logger.warn(`Slow request: ${req.method} ${req.path} - ${responseTime}ms`);
    }
  });

  next();
});

// 4. RATE LIMITING - Advanced protection with different tiers
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: isDev ? 1000 : 100, // API requests per window
  message: 'Too many API requests, please try again later',
  standardHeaders: true,
  legacyHeaders: false,
  skip: (req) => isDev && req.ip === '127.0.0.1' // Skip localhost in dev
});

const wmlLimiter = rateLimit({
  windowMs: 1 * 60 * 1000, // 1 minute
  max: isDev ? 500 : 60, // WML page requests per minute
  message: 'Too many requests, please slow down',
  standardHeaders: true,
  legacyHeaders: false
});

app.use("/api", apiLimiter);
app.use("/wml", wmlLimiter);

// ============ AUTHENTICATION MIDDLEWARE ============
// Protect all WML pages - redirect to QR if not logged in
/*app.use("/wml", (req, res, next) => {
  // Allow access to QR code page and logout page without authentication
  const publicPages = ['/wml/qr.wml', '/wml/qr-display.wml', '/wml/logout.wml'];

  if (publicPages.includes(req.path)) {
    return next();
  }

  // Check if WhatsApp is connected
  const isConnected = !!sock?.authState?.creds && connectionState === 'open';

  if (!isConnected) {
    // Redirect directly to QR page - no intermediate page
    return res.redirect(302, '/wml/qr.wml');
  }

  next();
});
*/
// 5. LOGGING - Production-grade Winston logger
const logger = winston.createLogger({
  level: isDev ? "debug" : "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({
      filename: "error.log",
      level: "error",
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }),
    new winston.transports.File({
      filename: "app.log",
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }),
  ],
});

// 6. BODY PARSERS - With size limits for security
app.use(express.urlencoded({
  extended: true,
  limit: '10mb',
  parameterLimit: 1000
}));
app.use(express.json({
  limit: '10mb'
}));

// Storage with better persistence
const storage = new PersistentStorage("./data");
const persistentData = storage.loadAllData();

let messageStore = persistentData.messages;
let contactStore = persistentData.contacts;
let chatStore = persistentData.chats;
let connectionState = "disconnected";
let currentQR ; // Store the current QR code
let isFullySynced = persistentData.meta.isFullySynced;
let syncAttempts = persistentData.meta.syncAttempts;
let isConnecting = false; // Prevent race conditions in connection logic

// ============ PRODUCTION-GRADE ADVANCED CACHING SYSTEM ============
// Multi-layer LRU cache with automatic memory management
class ProductionCache {
  constructor(maxSize = 999999999999, ttl = 60000) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.ttl = ttl;
    this.hits = 0;
    this.misses = 0;
  }

  get(key) {
    const item = this.cache.get(key);

    if (!item) {
      this.misses++;
      performanceMetrics.cache.misses++;
      return null;
    }

    // Check TTL
    if (Date.now() - item.timestamp > this.ttl) {
      this.cache.delete(key);
      this.misses++;
      performanceMetrics.cache.misses++;
      return null;
    }

    // LRU: Move to end (most recently used)
    this.cache.delete(key);
    this.cache.set(key, item);

    this.hits++;
    performanceMetrics.cache.hits++;
    return item.data;
  }

  set(key, data) {
    // Evict oldest if at capacity (LRU)
    if (this.cache.size >= this.maxSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  invalidate(key) {
    if (key) {
      this.cache.delete(key);
    } else {
      this.cache.clear();
    }
  }

  getStats() {
    const total = this.hits + this.misses;
    const hitRate = total > 0 ? ((this.hits / total) * 100).toFixed(2) : 0;

    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hits: this.hits,
      misses: this.misses,
      hitRate: `${hitRate}%`
    };
  }
}

// Initialize caches optimized for 4GB Raspberry Pi 4
const groupsCache = new ProductionCache(100, 120000); // 100 groups, 2min TTL (4GB RAM optimized)
const contactsCache = new ProductionCache(1000, 600000); // 1000 contacts, 10min TTL (4GB RAM optimized)
const chatsCache = new ProductionCache(200, 300000); // 200 chats, 5min TTL (4GB RAM optimized)
const messagesCache = new ProductionCache(2000, 120000); // 2000 messages, 2min TTL (4GB RAM optimized)

// ============ USER SETTINGS & FAVORITES ============
// Load user settings with defaults
let userSettings = persistentData.settings || {
  defaultLanguage: 'en',
  defaultImageFormat: 'wbmp',
  defaultVideoFormat: 'wbmp',
  paginationLimit: 10,
  autoRefresh: false,
  showHelp: true,
  ttsEnabled: true,
  favorites: [] // Array of JIDs
};

// Save settings
function saveSettings() {
  storage.queueSave("settings", userSettings);
}

// Favorite contacts helpers
function addFavorite(jid) {
  if (!userSettings.favorites.includes(jid)) {
    userSettings.favorites.push(jid);
    saveSettings();
    return true;
  }
  return false;
}

function removeFavorite(jid) {
  const index = userSettings.favorites.indexOf(jid);
  if (index > -1) {
    userSettings.favorites.splice(index, 1);
    saveSettings();
    return true;
  }
  return false;
}

function isFavorite(jid) {
  return userSettings.favorites.includes(jid);
}

// Get unread message count
function getUnreadCount() {
  let count = 0;
  for (const [jid, messages] of chatStore.entries()) {
    for (const msg of messages) {
      if (!msg.key.fromMe && !msg.messageStubType && msg.message) {
        // Simple heuristic: if we haven't marked it read, count it
        count++;
      }
    }
  }
  return count;
}

// Get recent chats (last 5)
function getRecentChats(limit = 5) {
  const chatsWithTime = [];
  for (const [jid, messages] of chatStore.entries()) {
    if (messages.length > 0) {
      const lastMsg = messages[messages.length - 1];
      chatsWithTime.push({
        jid,
        timestamp: lastMsg.messageTimestamp || 0,
        lastMessage: lastMsg
      });
    }
  }
  chatsWithTime.sort((a, b) => Number(b.timestamp) - Number(a.timestamp));
  return chatsWithTime.slice(0, limit);
}

// WML Constants
const WML_DTD =
  '<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.3//EN" "http://www.wapforum.org/DTD/wml13.dtd">';
const WMLSCRIPT_DTD =
  '<!DOCTYPE wmls PUBLIC "-//WAPFORUM//DTD WMLScript 1.3//EN" "http://www.wapforum.org/DTD/wmls13.dtd">';

// WML Helper Functions
function esc(s = "") {
  return String(s).replace(
    /[&<>"']/g,
    (c) =>
      ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[c])
  );
}

function saveContacts() {
  storage.queueSave("contacts", contactStore);
}

function saveChats() {
  storage.queueSave("chats", chatStore);
}

function saveMessages() {
  storage.queueSave("messages", messageStore);
}

function saveMeta() {
  const meta = {
    isFullySynced,
    syncAttempts,
    lastSync: new Date().toISOString(),
    contactsCount: contactStore.size,
    chatsCount: chatStore.size,
    messagesCount: messageStore.size,
  };
  storage.queueSave("meta", meta);
}

// Update saveAll function to include new auth state keys
function saveAll() {
  saveContacts();
  saveChats();
  saveMessages();
  saveMeta();
  
  // Ensure new auth state keys are saved
  if (sock && sock.authState) {
    storage.queueSave("auth_state", sock.authState);
  }
}
function wmlDoc(cards, scripts = "") {
  const head = scripts
    ? `<head><meta http-equiv="Cache-Control" content="max-age=0"/>${scripts}</head>`
    : '<head><meta http-equiv="Cache-Control" content="max-age=0"/></head>';
  return `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.0//EN" "http://www.wapforum.org/DTD/wml_1.0.xml">
<wml>${head}${cards}</wml>`;
}

function sendWml(res, cards, scripts = "") {
  // MODIFICA: Imposta il Content-Type corretto per WML
  res.setHeader("Content-Type", "text/vnd.wap.wml; charset=UTF-8");
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");
  res.setHeader("Accept-Ranges", "none");
  
  // MODIFICA: Usa la codifica ISO-8859-1 per compatibilità Nokia
  const wmlContent = wmlDoc(cards, scripts);
  const encodedBuffer = iconv.encode(wmlContent, 'iso-8859-1');
  
  // MODIFICA: Aggiorna il Content-Type per ISO-8859-1
  res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
  res.send(encodedBuffer);
}

function card(id, title, inner, ontimer = null, scripts = '') {
  const timerAttr = ontimer ? ` ontimer="${ontimer}"` : "";
  return `<card id="${esc(id)}" title="${esc(title)}"${timerAttr}>
    ${scripts}
    ${inner}
  </card>`;
}

function truncate(s = "", max = 64) {
  const str = String(s);
  return str.length > max ? str.slice(0, max - 1) + "…" : str;
}



async function getContactName(jid, sock) {
  if (!jid) return "Unknown";

  const isGroup = jid.endsWith("@g.us");

  // Try to get from contactStore first (cached)
  let contact = contactStore.get(jid);

  // If not found and it's not a group, try alternative lookups
  if (!contact && !isGroup) {
    // Try with formatted JID
    const formattedJid = formatJid(jid);
    if (formattedJid !== jid) {
      contact = contactStore.get(formattedJid);
    }

    // If still not found, try by phone number
    if (!contact) {
      const phoneNumber = jidFriendly(jid);
      // Look through all contacts to find a match by phone number
      for (const [key, value] of contactStore.entries()) {
        if (value.phoneNumber === phoneNumber || 
            (value.id && value.id.includes(phoneNumber)) ||
            (key.includes(phoneNumber))) {
          contact = value;
          break;
        }
      }
    }
  }

  if (isGroup) {
    // For groups, try multiple sources
    if (contact?.subject) return contact.subject;
    if (contact?.name) return contact.name;

    // Try to fetch group metadata if sock is available
    if (sock) {
      try {
        const groupMetadata = await sock.groupMetadata(jid);
        if (groupMetadata?.subject) {
          // Cache it
          contactStore.set(jid, { ...contact, subject: groupMetadata.subject });
          return groupMetadata.subject;
        }
      } catch (error) {
        // Silently fail, use fallback
      }
    }

    // Fallback for groups
    const groupId = jid.replace("@g.us", "");
    return `Group ${groupId.slice(-8)}`;
  } else {
    // For individual contacts - handle LID vs PN
    if (contact?.id) {
      // If we have a contact with id field
      if (contact.phoneNumber) {
        return contact.phoneNumber; // Show phone number if available
      } else if (contact.lid) {
        return `LID:${contact.lid.substring(4)}`; // Show LID
      }
    }
    
    // Fallback to traditional fields
    if (contact?.name) return contact.name;
    if (contact?.notify) return contact.notify;
    if (contact?.verifiedName) return contact.verifiedName;

    // Try to get from WhatsApp if sock is available
    if (sock) {
      try {
        // Check if it's a LID or PN
        const isLid = jid.startsWith('lid:');
        const queryJid = isLid ? jid : formatJid(jid);
        
        const [result] = await sock.onWhatsApp(queryJid);
        if (result?.exists) {
          const name = result.name || result.notify;
          if (name) {
            // Cache it with new structure
            contactStore.set(jid, { 
              id: queryJid,
              name: name,
              phoneNumber: isLid ? jidFriendly(jid) : undefined,
              lid: isLid ? jid : undefined
            });
            return name;
          }
        }
      } catch (error) {
        // Silently fail, use fallback
      }
    }

    // Fallback to formatted phone number
    return jidFriendly(jid);
  }
}

function parseList(str = "") {
  return String(str)
    .split(/[,;\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

// Update formatJid function to handle LIDs
// Update formatJid function to handle LIDs
function formatJid(raw = "") {
  const s = String(raw).trim();
  if (!s) return s;

  // If it's already a LID (contains colon), return as-is
  if (s.includes(':')) {
    return s;
  }

  // For phone numbers, add domain
  return s.includes("@") ? s : `${s}@s.whatsapp.net`;
}


function jidFriendly(jid = "") {
  if (!jid) return "";
  
  // Handle LIDs
  if (jid.startsWith('lid:')) {
    return `LID:${jid.substring(4)}`;
  }
  
  if (jid.endsWith("@s.whatsapp.net"))
    return jid.replace("@s.whatsapp.net", "");
  if (jid.endsWith("@g.us")) return `Group ${jid.slice(0, -5)}`;
  return jid;
}

function ensureGroupJid(raw = "") {
  const s = String(raw).trim();
  if (!s) return s;
  return s.endsWith("@g.us") ? s : `${s}@g.us`;
}

function messageText(msg) {
  try {
    if (!msg) return "[No message]";

    const c = extractMessageContent(msg?.message);
    if (!c) return "[Unsupported message]";

    if (c.conversation) return c.conversation || "[Empty message]";
    if (c.extendedTextMessage?.text)
      return c.extendedTextMessage.text || "[Empty text]";
    if (c.imageMessage?.caption) return `[IMG] ${c.imageMessage.caption || ""}`;
    if (c.videoMessage?.caption) return `[VID] ${c.videoMessage.caption || ""}`;
    if (c.audioMessage) {
      const duration = c.audioMessage.seconds || 0;
      const transcription = msg.transcription || "";

      let result = `[AUDIO ${duration}s]`;

      // Add transcription indicator if available
      if (
        transcription &&
        transcription !== "[Trascrizione fallita]" &&
        transcription !== "[Audio troppo lungo per la trascrizione]"
      ) {
        result += " 📝";
      }

      return result;
    }
    if (c.documentMessage)
      return `[DOC] ${c.documentMessage.fileName || "Document"}`;
    if (c.stickerMessage) return "[Sticker]";

    const type = getContentType(msg?.message) || "unknown";
    return `[${type.toUpperCase()}]`;
  } catch (error) {
    console.error("Error in messageText:", error);
    return "[Error]";
  }
}
function resultCard(
  title,
  lines = [],
  backHref = "/wml/home.wml",
  autoRefresh = true
) {
  const refreshTimer = autoRefresh ? "" : "";
  const onTimer = autoRefresh ? ` ontimer="${backHref}"` : "";

  const body = `
    ${refreshTimer}
    <p><b>${esc(title)}</b></p>
    ${lines.map((l) => `<p>${esc(l)}</p>`).join("")}
    <p>
      <a href="${backHref}" accesskey="0">[0] Back</a> 
      <a href="/wml/home.wml" accesskey="9">[9] Home</a>
    </p>
    <do type="accept" label="OK">
      <go href="${backHref}"/>
    </do>
    <do type="options" label="Menu">
      <go href="/wml/home.wml"/>
    </do>
  `;
  return `<card id="result" title="${esc(title)}"${onTimer}>${body}</card>`;
}

function navigationBar() {
  return `
    <p>
      <a href="/wml/home.wml" accesskey="1">[1] Home</a> 
      <a href="/wml/chats.wml" accesskey="2">[2] Chats</a> 
      <a href="/wml/contacts.wml" accesskey="3">[3] Contacts</a> 
      <a href="/wml/send-menu.wml" accesskey="4">[4] Send</a>
    </p>
  `;
}

function searchBox(action, placeholder = "Search...") {
  return `
    <p>
      <input name="q" title="${esc(placeholder)}" size="20" maxlength="50"/>
      <do type="accept" label="Search">
        <go href="${action}" method="get">
          <postfield name="q" value="$(q)"/>
        </go>
      </do>
    </p>
  `;
}

// WMLScript functions
function wmlScript(name, functions) {
  return `<script src="/wmlscript/${name}.wmls" type="text/vnd.wap.wmlscriptc"/>`;
}

// WMLScript files endpoint
app.get("/wmlscript/:filename", (req, res) => {
  const { filename } = req.params;
  let script = "";

  res.setHeader("Content-Type", "text/vnd.wap.wmlscript");
  res.setHeader("Cache-Control", "max-age=3600");

  switch (filename) {
    case "utils.wmls":
      script = `
extern function refresh();
extern function confirmAction(message);
extern function showAlert(text);

function refresh() {
  WMLBrowser.refresh();
}

function confirmAction(message) {
  var result = Dialogs.confirm(message, "Confirm", "Yes", "No");
  return result;
}

function showAlert(text) {
  Dialogs.alert(text);
}
`;
      break;
    case "wtai.wmls":
      script = `
extern function makeCall(number);
extern function sendSMS(number, message);
extern function addContact(name, number);

function makeCall(number) {
  WTAVoice.setup("wtai://wp/mc;" + number, "");
}

function sendSMS(number, message) {
  WTASMS.send("wtai://wp/ms;" + number + ";" + message, "");
}

function addContact(name, number) {
  WTAPhoneBook.write("wtai://wp/ap;" + name + ";" + number, "");
}
`;
      break;
    default:
      return res.status(404).send("Script not found");
  }

  res.send(script);
});

// Enhanced Home page with WMLScript integration
app.get(["/wml", "/wml/home.wml"], (req, res) => {
  const connected = !!sock?.authState?.creds;
  const unreadCount = 0; // TODO: Implement unread tracking
  const recentChats = getRecentChats(3);

  const scripts = `
    ${wmlScript("utils")}
    ${wmlScript("wtai")}
  `;

  // Favorites section
  let favoritesHtml = '';
  if (userSettings.favorites.length > 0) {
    favoritesHtml = '<p><b>Favorites:</b></p><p>';
    for (let i = 0; i < Math.min(3, userSettings.favorites.length); i++) {
      const jid = userSettings.favorites[i];
      const contact = contactStore.get(jid);
      const name = contact?.name || contact?.notify || jidFriendly(jid);
      favoritesHtml += `<a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}">${esc(name.substring(0, 15))}</a><br/>`;
    }
    if (userSettings.favorites.length > 3) {
      favoritesHtml += `<a href="/wml/favorites.wml">[View All ${userSettings.favorites.length}]</a>`;
    }
    favoritesHtml += '</p>';
  }

  // Recent chats section
  let recentHtml = '';
  if (recentChats.length > 0) {
    recentHtml = '<p><b>Recent Chats:</b></p><p>';
    for (const chat of recentChats) {
      const contact = contactStore.get(chat.jid);
      const name = contact?.name || contact?.notify || jidFriendly(chat.jid);
      recentHtml += `<a href="/wml/chat.wml?jid=${encodeURIComponent(chat.jid)}">${esc(name.substring(0, 15))}</a><br/>`;
    }
    recentHtml += '</p>';
  }

  const body = `
    <p><b>WhatsApp WAP</b></p>
    <p>${connected ? "Online" : "Offline"} | ${
    isFullySynced ? "Synced" : "Syncing..."
  }</p>
    <p>Contacts: ${contactStore.size} | Chats: ${chatStore.size}</p>

    ${favoritesHtml}
    ${recentHtml}

    <p><b>Main Menu:</b></p>
    <p>
      <a href="/wml/contacts.wml" accesskey="1">[1] Contacts</a><br/>
      <a href="/wml/chats.wml" accesskey="2">[2] Chats</a><br/>
      <a href="/wml/favorites.wml" accesskey="3">[3] Favorites ⭐</a><br/>
      <a href="/wml/send-menu.wml" accesskey="4">[4] Send</a><br/>
      <a href="/wml/groups.wml" accesskey="5">[5] Groups</a><br/>
      <a href="/wml/status-broadcast.wml" accesskey="6">[6] Post Status</a>
    </p>

    <p><b>Tools:</b></p>
    <p>
      <a href="/wml/search.wml" accesskey="7">[7] Search</a><br/>
      <a href="/wml/settings.wml" accesskey="8">[8] Settings</a><br/>
      <a href="/wml/help.wml" accesskey="9">[9] Help</a><br/>
      <a href="/wml/me.wml">[*] Profile</a>
    </p>

    <p><b>System:</b></p>
    <p>
      <a href="/wml/status.wml">[Sys Status]</a><br/>
      <a href="/wml/qr.wml">[*] QR Code</a><br/>
      <a href="/wml/logout.wml" accesskey="0">[0] Logout</a>
    </p>

    <p><small>Server: Port ${port}</small></p>

    <do type="accept" label="Refresh">
      <go href="/wml/home.wml"/>
    </do>
    <do type="options" label="Settings">
      <go href="/wml/settings.wml"/>
    </do>
  `;

  sendWml(res, card("home", "Home", body, "/wml/home.wml"), scripts);
});

/*
app.get('/wml/chat.wml', async (req, res) => {
  const raw = req.query.jid || ''
  const jid = formatJid(raw)
  const limit = 10 // Aumentato da 6 a 10 messaggi per pagina
  const offset = Math.max(0, parseInt(req.query.offset || '0'))
  const search = (req.query.search || '').trim().toLowerCase()

  // Carica cronologia se mancante
  if ((!chatStore.get(jid) || chatStore.get(jid).length === 0) && sock) {
    try { 
      await loadChatHistory(jid, limit * 5) // carica più messaggi per navigazione
    } catch (e) { 
      logger.warn(`Failed to load chat history for ${jid}: ${e.message}`) 
    }
  }

  let allMessages = (chatStore.get(jid) || []).slice()
  
  // Ordinamento cronologico CRESCENTE (dal più vecchio al più recente)
  allMessages.sort((a, b) => Number(a.messageTimestamp) - Number(b.messageTimestamp))

  // Applica filtro di ricerca se presente
  if (search) {
    allMessages = allMessages.filter(m => (messageText(m) || '').toLowerCase().includes(search))
  }

  // Per la paginazione con ordinamento crescente, prendiamo gli ultimi messaggi
  // ma li mostriamo nell'ordine corretto (dal più vecchio al più recente)
  const totalMessages = allMessages.length
  const startIndex = Math.max(0, totalMessages - limit - offset)
  const endIndex = totalMessages - offset
  const slice = allMessages.slice(startIndex, endIndex)

  const contact = contactStore.get(jid)
  const chatName = contact?.name || contact?.notify || contact?.verifiedName || jidFriendly(jid)
  const number = jidFriendly(jid)

  // Escape sicuro e rimuove caratteri non ASCII
  const escWml = text => (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/[^\x20-\x7E]/g, '?')

  let messageList
  if (slice.length === 0) {
    messageList = `<p>No messages found.</p>
      <p>
        <a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}" accesskey="2">[Clear Search]</a> |
        <a href="/wml/chats.wml" accesskey="0">[Back to Chats]</a>
      </p>`
  } else {
    messageList = slice.map((m, idx) => {
      const who = m.key.fromMe ? 'Me' : chatName
      const text = truncate(messageText(m), 100)
      const ts = new Date(Number(m.messageTimestamp) * 1000).toLocaleTimeString('en-GB', {
        hour: '2-digit', 
        minute: '2-digit'
      })
      const mid = m.key.id
      
      return `<p><b>${idx + 1}. ${escWml(who)}</b> (${ts})<br/>
        ${escWml(text)}<br/>
        <a href="/wml/msg.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}" accesskey="${Math.min(idx + 1, 9)}">[Actions]</a>
      </p>`
    }).join('')
  }

  // Navigazione corretta per ordinamento crescente
  const olderOffset = offset + limit
  const olderLink = olderOffset < totalMessages
    ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;offset=${olderOffset}&amp;search=${encodeURIComponent(search)}" accesskey="2">[2] Older</a>` : ''
  
  const newerOffset = Math.max(0, offset - limit)
  const newerLink = offset > 0
    ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;offset=${newerOffset}&amp;search=${encodeURIComponent(search)}" accesskey="3">[3] Newer</a>` : ''

  // Search form sempre visibile
  const searchForm = `
    <p><b>Search Messages:</b></p>
    <p>
      <input name="searchQuery" title="Search..." value="${escWml(search)}" size="15" maxlength="50"/>
      <do type="accept" label="Search">
        <go href="/wml/chat.wml" method="get">
          <postfield name="jid" value="${escWml(jid)}"/>
          <postfield name="search" value="$(searchQuery)"/>
          <postfield name="offset" value="0"/>
        </go>
      </do>
    </p>
    ${search ? `<p>Searching: <b>${escWml(search)}</b> | <a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}">[Clear]</a></p>` : ''}
  `

  // Indicatori di paginazione migliorati
  const currentPage = Math.floor(offset / limit) + 1
  const totalPages = Math.ceil(totalMessages / limit)
  const paginationInfo = `
    <p><b>Messages ${Math.max(1, totalMessages - endIndex + 1)}-${totalMessages - startIndex} of ${totalMessages}</b></p>
    <p>Page ${currentPage}/${totalPages}</p>
  `

  const body = `
    <p><b>${escWml(chatName)}</b></p>
    <p>${escWml(number)} | Total: ${totalMessages} messages</p>

    ${searchForm}

    ${paginationInfo}

    ${messageList}

    <p><b>Navigation:</b></p>
    <p>${olderLink} ${olderLink && newerLink ? ' | ' : ''} ${newerLink}</p>

    <p><b>Quick Actions:</b></p>
    <p>
      <a href="/wml/send.text.wml?to=${encodeURIComponent(jid)}" accesskey="1">[1] Send Text</a> |
      <a href="/wml/contact.wml?jid=${encodeURIComponent(jid)}" accesskey="4">[4] Contact Info</a>
      ${number && !jid.endsWith('@g.us') ? ` | <a href="wtai://wp/mc;${number}" accesskey="9">[9] Call</a>` : ''}
    </p>

    <p>
      <a href="/wml/chats.wml" accesskey="0">[0] Back to Chats</a> |
      <a href="/wml/home.wml" accesskey="*">[*] Home</a>
    </p>

    <do type="accept" label="Send">
      <go href="/wml/send.text.wml?to=${encodeURIComponent(jid)}"/>
    </do>
    <do type="options" label="Refresh">
      <go href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;offset=${offset}&amp;search=${encodeURIComponent(search)}"/>
    </do>
  `

  res.setHeader('Content-Type', 'text/vnd.wap.wml; charset=UTF-8')
  res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate')
  res.setHeader('Pragma', 'no-cache')
  res.setHeader('Expires', '0')
  
  res.send(`<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <head>
    <meta http-equiv="Cache-Control" content="max-age=0"/>
  </head>
  <card id="chat" title="${escWml(chatName)}">
    ${body}
  </card>
</wml>`)
})*/

// Enhanced Status page
app.get("/wml/status.wml", (req, res) => {
  const connected = !!sock?.authState?.creds;
  const uptime = Math.floor(process.uptime() / 60);

  const body = `
    <p><b>System Status</b></p>
    <p>Connection: ${connected ? "<b>Active</b>" : "<em>Inactive</em>"}</p>
    <p>State: ${esc(connectionState)}</p>
    <p>QR Available: ${currentQR ? "Yes" : "No"}</p>
    <p>Uptime: ${uptime} minutes</p>

    <p>Sync Status: ${
      isFullySynced ? "<b>Complete</b>" : "<em>In Progress</em>"
    }</p>
    <p>Sync Attempts: ${syncAttempts}</p>
    <p>Contacts: ${contactStore.size}</p>
    <p>Chats: ${chatStore.size}</p>
    <p>Messages: ${messageStore.size}</p>

    <p><b>Sync Actions:</b></p>
    <p>
      <a href="/wml/sync.full.wml" accesskey="1"> Sync</a>
    </p>

    ${navigationBar()}

    <do type="accept" label="Refresh">
      <go href="/wml/status.wml"/>
    </do>
  `;

  sendWml(res, card("status", "Status", body, "/wml/status.wml"));
});

// Enhanced QR Code page
app.get("/wml/qr.wml", (req, res) => {
  const isConnected = !!sock?.authState?.creds && connectionState === 'open';

  const body = isConnected
    ? `
      <p>WhatsApp Connected</p>
      <p>You are logged in</p>
      <p>
        <a href="/wml/home.wml">Go to Home</a>
      </p>
    `
    : currentQR
    ? `
      <p>Scan QR Code</p>
      <p>1. Open WhatsApp</p>
      <p>2. Menu - Linked Devices</p>
      <p>3. Link a Device</p>
      <p>4. Scan QR:</p>
      <p>
        <img src="/api/qr/image?format=wbmp"/>
      </p>
      <p>Status: ${esc(connectionState)}</p>
      <p>Press OK to refresh</p>
    `
    : `
      <p>Connecting...</p>
      <p>Status: ${esc(connectionState)}</p>
      <p>QR code loading</p>
      <p>Please wait</p>
    `;

  const body_full = `
    ${body}
    <p>Port ${port}</p>
    <do type="accept" label="Refresh">
      <go href="/wml/qr.wml"/>
    </do>
  `;

  sendWml(res, card("qr", "QR Code", body_full));
});

app.get("/api/qr/wml-wbmp", (req, res) => {
  const isConnected = !!sock?.authState?.creds && connectionState === 'open';

  if (isConnected) {
    res.set("Content-Type", "text/vnd.wap.wml");
    return res.send(`<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.0//EN" "http://www.wapforum.org/DTD/wml_1.0.xml">
<wml>
  <card id="connected" title="WhatsApp">
    <p>WhatsApp Connected</p>
    <p>You are logged in</p>
    <p>
      <a href="/wml/home.wml">Go to Home</a>
    </p>
  </card>
</wml>`);
  }

  if (!currentQR) {
    res.set("Content-Type", "text/vnd.wap.wml");
    return res.send(`<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.0//EN" "http://www.wapforum.org/DTD/wml_1.0.xml">
<wml>
  <card id="noqr" title="QR">
    <p>Connecting...</p>
    <p>QR code loading</p>
    <p>Please wait</p>
  </card>
</wml>`);
  }

  // WAP 1.0 compatible page with WBMP QR code
  res.set("Content-Type", "text/vnd.wap.wml");
  res.send(`<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.0//EN" "http://www.wapforum.org/DTD/wml_1.0.xml">
<wml>
  <card id="qr" title="WhatsApp QR">
    <p>Scan QR Code</p>
    <p>1. Open WhatsApp</p>
    <p>2. Menu - Linked Devices</p>
    <p>3. Link a Device</p>
    <p>4. Scan QR:</p>
    <p><img src="/api/qr/image?format=wbmp"/></p>
    <p>Status: ${connectionState}</p>
    <do type="accept" label="Refresh">
      <go href="/api/qr/wml-wbmp"/>
    </do>
  </card>
</wml>`);
});

// ============ SETTINGS PAGE ============
app.get("/wml/settings.wml", (req, res) => {
  const body = `
    <p><b>Settings</b></p>

    <p><b>Defaults:</b></p>
    <p>TTS Language: ${esc(userSettings.defaultLanguage)}<br/>
    <a href="/wml/settings-tts.wml">[Change Language]</a></p>

    <p>Image Format: ${esc(userSettings.defaultImageFormat.toUpperCase())}<br/>
    <a href="/wml/settings-format.wml?type=image">[Change]</a></p>

    <p>Video Format: ${esc(userSettings.defaultVideoFormat.toUpperCase())}<br/>
    <a href="/wml/settings-format.wml?type=video">[Change]</a></p>

    <p><b>Display:</b></p>
    <p>Items per page: ${userSettings.paginationLimit}<br/>
    <a href="/wml/settings-pagination.wml">[Change]</a></p>

    <p><b>Favorites:</b></p>
    <p>${userSettings.favorites.length} contacts<br/>
    <a href="/wml/favorites.wml">[Manage]</a></p>

    <p><b>About:</b></p>
    <p>WhatsApp WAP Client v1.0<br/>
    For Nokia 7210 &amp; WAP devices<br/>
    <a href="/wml/help.wml">[Help &amp; Guide]</a></p>

    <p><a href="/wml/home.wml" accesskey="0">[0] Home</a></p>

    <do type="accept" label="Home">
      <go href="/wml/home.wml"/>
    </do>
  `;

  sendWml(res, card("settings", "Settings", body));
});

// Settings - TTS Language
app.post("/wml/settings-tts.wml", (req, res) => {
  const { language } = req.body;
  if (language) {
    userSettings.defaultLanguage = language;
    saveSettings();
  }
  res.redirect("/wml/settings.wml");
});

app.get("/wml/settings-tts.wml", (req, res) => {
  const ttsStatus = ttsEnabled ? '✓ Ready' : '⚠ espeak not installed';
  const sttStatus = transcriptionEnabled ? '✓ Ready' : '⚠ Model not loaded';

  const body = `
    <p><b>Speech Settings</b></p>

    <p><b>TTS (Text-to-Speech):</b></p>
    <p>Engine: espeak (Offline)</p>
    <p>Languages: English + Italian</p>
    <p>Status: ${ttsStatus}</p>

    <p><b>STT (Speech-to-Text):</b></p>
    <p>Engine: Whisper Base (Offline)</p>
    <p>Languages: English + Italian</p>
    <p>Status: ${sttStatus}</p>
    <p>Detection: Auto + Manual</p>

    <p><b>Supported Languages:</b></p>
    <p><small>• English (en)<br/>• Italian (it)<br/>• Auto-detection</small></p>

    <p><b>STT Quality:</b></p>
    <p><small>Model: whisper-base (74MB)<br/>High accuracy transcription<br/>Works offline, no internet</small></p>

    <p><b>Info:</b></p>
    <p><small>TTS: espeak (local)<br/>STT: Whisper (high accuracy)<br/>Both fully offline</small></p>

    ${!ttsEnabled ? '<p><small>TTS: sudo apt-get install espeak</small></p>' : ''}
    ${!transcriptionEnabled ? '<p><small>STT: Whisper model loading...</small></p>' : ''}

    <p><a href="/wml/settings.wml" accesskey="0">[0] Back</a></p>
  `;

  sendWml(res, card("settings-speech", "Speech Settings", body));
});

// Settings - Format
app.post("/wml/settings-format.wml", (req, res) => {
  const { type, format } = req.body;
  if (type === 'image' && format) {
    userSettings.defaultImageFormat = format;
  } else if (type === 'video' && format) {
    userSettings.defaultVideoFormat = format;
  }
  saveSettings();
  res.redirect("/wml/settings.wml");
});

app.get("/wml/settings-format.wml", (req, res) => {
  const type = req.query.type || 'image';
  const current = type === 'image' ? userSettings.defaultImageFormat : userSettings.defaultVideoFormat;

  const body = `
    <p><b>Default ${esc(type.charAt(0).toUpperCase() + type.slice(1))} Format</b></p>
    <p>Current: ${esc(current.toUpperCase())}</p>

    <p>Select format:</p>
    <select name="format" title="Format">
      <option value="wbmp"${current === 'wbmp' ? ' selected="selected"' : ''}>WBMP (B&amp;W, Small)</option>
      <option value="jpg"${current === 'jpg' ? ' selected="selected"' : ''}>JPEG (Color, Medium)</option>
      <option value="png"${current === 'png' ? ' selected="selected"' : ''}>PNG (Color, Large)</option>
    </select>

    <do type="accept" label="Save">
      <go method="post" href="/wml/settings-format.wml">
        <postfield name="type" value="${esc(type)}"/>
        <postfield name="format" value="$format"/>
      </go>
    </do>

    <p><a href="/wml/settings.wml" accesskey="0">[0] Back</a></p>
  `;

  sendWml(res, card("settings-format", "Format", body));
});

// Settings - Pagination
app.post("/wml/settings-pagination.wml", (req, res) => {
  const { limit } = req.body;
  if (limit) {
    userSettings.paginationLimit = parseInt(limit, 10) || 10;
    saveSettings();
  }
  res.redirect("/wml/settings.wml");
});

app.get("/wml/settings-pagination.wml", (req, res) => {
  const body = `
    <p><b>Items Per Page</b></p>
    <p>Current: ${userSettings.paginationLimit}</p>

    <p>Select:</p>
    <select name="limit" title="Limit">
      <option value="5"${userSettings.paginationLimit === 5 ? ' selected="selected"' : ''}>5 items</option>
      <option value="10"${userSettings.paginationLimit === 10 ? ' selected="selected"' : ''}>10 items</option>
      <option value="15"${userSettings.paginationLimit === 15 ? ' selected="selected"' : ''}>15 items</option>
      <option value="20"${userSettings.paginationLimit === 20 ? ' selected="selected"' : ''}>20 items</option>
    </select>

    <do type="accept" label="Save">
      <go method="post" href="/wml/settings-pagination.wml">
        <postfield name="limit" value="$limit"/>
      </go>
    </do>

    <p><a href="/wml/settings.wml" accesskey="0">[0] Back</a></p>
  `;

  sendWml(res, card("settings-pagination", "Pagination", body));
});

// ============ HELP PAGE ============
app.get("/wml/help.wml", (req, res) => {
  const section = req.query.section || 'main';

  let body = '';

  if (section === 'main') {
    body = `
      <p><b>Help &amp; Guide</b></p>

      <p><b>Quick Start:</b></p>
      <p>1. Home has favorites &amp; recent chats<br/>
      2. Use number keys for shortcuts<br/>
      3. [0] always goes back/home<br/>
      4. Add favorites for quick access</p>

      <p><b>Topics:</b></p>
      <p>
        <a href="/wml/help.wml?section=keys" accesskey="1">[1] Keyboard Shortcuts</a><br/>
        <a href="/wml/help.wml?section=messages" accesskey="2">[2] Sending Messages</a><br/>
        <a href="/wml/help.wml?section=media" accesskey="3">[3] Media &amp; Files</a><br/>
        <a href="/wml/help.wml?section=video" accesskey="4">[4] Video Playback</a><br/>
        <a href="/wml/help.wml?section=tts" accesskey="5">[5] Voice Messages (TTS)</a><br/>
        <a href="/wml/help.wml?section=favorites" accesskey="6">[6] Favorites</a><br/>
        <a href="/wml/help.wml?section=tips" accesskey="7">[7] Tips &amp; Tricks</a>
      </p>

      <p><a href="/wml/home.wml" accesskey="0">[0] Home</a></p>
    `;
  } else if (section === 'keys') {
    body = `
      <p><b>Keyboard Shortcuts</b></p>

      <p><b>Home Screen:</b></p>
      <p>[1] Contacts | [2] Chats<br/>
      [3] Send | [4] Groups<br/>
      [5] Search | [6] Settings<br/>
      [7] Help | [8] Profile<br/>
      [9] Status | [0] Logout</p>

      <p><b>Universal:</b></p>
      <p>[0] Back / Home<br/>
      [*] Quick action<br/>
      [#] Options menu</p>

      <p><a href="/wml/help.wml" accesskey="0">[0] Back</a></p>
    `;
  } else if (section === 'messages') {
    body = `
      <p><b>Sending Messages</b></p>

      <p><b>Types:</b></p>
      <p>1. Text - Regular message<br/>
      2. Voice (TTS) - Type to speech<br/>
      3. Image - Send photos<br/>
      4. Video - Send videos<br/>
      5. Audio - Send audio files<br/>
      6. Location - Share location<br/>
      7. Contact - Share contacts<br/>
      8. Poll - Create polls</p>

      <p><b>Quick Send:</b></p>
      <p>From Home: [3] Send<br/>
      Select contact &amp; type</p>

      <p><a href="/wml/help.wml" accesskey="0">[0] Back</a></p>
    `;
  } else if (section === 'media') {
    body = `
      <p><b>Media &amp; Files</b></p>

      <p><b>Format Selection:</b></p>
      <p>Images &amp; Videos support:<br/>
      - WBMP: B&amp;W, smallest<br/>
      - JPEG: Color, medium<br/>
      - PNG: Color, high quality</p>

      <p><b>Change Format:</b></p>
      <p>During viewing, press [7]<br/>
      Or set default in Settings</p>

      <p><b>Download:</b></p>
      <p>View media info page<br/>
      Select download format</p>

      <p><a href="/wml/help.wml" accesskey="0">[0] Back</a></p>
    `;
  } else if (section === 'video') {
    body = `
      <p><b>Video Playback (1 FPS)</b></p>

      <p><b>How it works:</b></p>
      <p>Videos play frame-by-frame<br/>
      1 frame per second<br/>
      Perfect for WAP devices!</p>

      <p><b>Controls:</b></p>
      <p>[4] Previous frame<br/>
      [5] Play / Auto-play<br/>
      [6] Next frame<br/>
      [7] Change format<br/>
      [0] Back</p>

      <p><b>Formats:</b></p>
      <p>WBMP: Fast, tiny files<br/>
      JPEG/PNG: Color, larger</p>

      <p><a href="/wml/help.wml" accesskey="0">[0] Back</a></p>
    `;
  } else if (section === 'tts') {
    body = `
      <p><b>Voice Messages (TTS)</b></p>

      <p><b>Text-to-Speech:</b></p>
      <p>Type text on your Nokia<br/>
      Converts to voice message<br/>
      Sends as WhatsApp audio!</p>

      <p><b>Features:</b></p>
      <p>- 15 languages supported<br/>
      - Up to 500 characters<br/>
      - Voice note or audio file<br/>
      - Free Google TTS</p>

      <p><b>Usage:</b></p>
      <p>Send Menu &gt; Voice (TTS)<br/>
      Type message &gt; Send</p>

      <p><b>Default Language:</b></p>
      <p>Set in Settings &gt; TTS Language</p>

      <p><a href="/wml/help.wml" accesskey="0">[0] Back</a></p>
    `;
  } else if (section === 'favorites') {
    body = `
      <p><b>Favorites System</b></p>

      <p><b>Quick Access:</b></p>
      <p>Add contacts to favorites<br/>
      Appears on home screen<br/>
      One-tap chat access</p>

      <p><b>Add Favorite:</b></p>
      <p>1. Open contact<br/>
      2. Select [Add to Favorites]<br/>
      3. See on home screen</p>

      <p><b>Manage:</b></p>
      <p>Settings &gt; Favorites<br/>
      View all &amp; remove</p>

      <p><a href="/wml/help.wml" accesskey="0">[0] Back</a></p>
    `;
  } else if (section === 'tips') {
    body = `
      <p><b>Tips &amp; Tricks</b></p>

      <p><b>Speed Tips:</b></p>
      <p>1. Use keyboard shortcuts<br/>
      2. Add frequent contacts to favorites<br/>
      3. Set WBMP as default format<br/>
      4. Use TTS for quick voice msgs</p>

      <p><b>Data Saving:</b></p>
      <p>- Use WBMP format<br/>
      - Reduce items per page<br/>
      - Disable auto-refresh</p>

      <p><b>Best Practices:</b></p>
      <p>- Sync when on WiFi<br/>
      - Keep favorites updated<br/>
      - Use search for old msgs</p>

      <p><a href="/wml/help.wml" accesskey="0">[0] Back</a></p>
    `;
  }

  sendWml(res, card("help", "Help", body));
});

// ============ FAVORITES PAGE ============
app.get("/wml/favorites.wml", (req, res) => {
  let body = '<p><b>Favorites</b></p>';

  if (userSettings.favorites.length === 0) {
    body += `<p>No favorites yet.<br/>
    Add contacts and groups from their info pages.</p>
    <p>
      <a href="/wml/contacts.wml">[Contacts]</a> |
      <a href="/wml/groups.wml">[Groups]</a>
    </p>`;
  } else {
    // Separate contacts and groups
    const contactFavs = [];
    const groupFavs = [];

    for (const jid of userSettings.favorites) {
      if (jid.endsWith('@g.us')) {
        groupFavs.push(jid);
      } else {
        contactFavs.push(jid);
      }
    }

    body += `<p>Total: ${userSettings.favorites.length} (${contactFavs.length} contacts, ${groupFavs.length} groups)</p>`;

    // Show contacts
    if (contactFavs.length > 0) {
      body += '<p><b>Contacts:</b></p>';
      for (let i = 0; i < contactFavs.length; i++) {
        const jid = contactFavs[i];
        const contact = contactStore.get(jid);
        const name = contact?.name || contact?.notify || jidFriendly(jid);
        const accessKey = i < 9 ? ` accesskey="${i + 1}"` : '';
        body += `<p>
          <a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}"${accessKey}>${i < 9 ? `[${i + 1}] ` : ''}${esc(name)}</a><br/>
          <a href="/wml/contact.wml?jid=${encodeURIComponent(jid)}">[Info]</a> |
          <a href="/wml/favorites-remove.wml?jid=${encodeURIComponent(jid)}">[Remove]</a>
        </p>`;
      }
    }

    // Show groups
    if (groupFavs.length > 0) {
      body += '<p><b>Groups:</b></p>';
      for (const jid of groupFavs) {
        const contact = contactStore.get(jid);
        const name = contact?.name || contact?.subject || jidFriendly(jid);
        body += `<p>
          <a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}">${esc(name)}</a><br/>
          <a href="/wml/group.view.wml?gid=${encodeURIComponent(jid)}">[Info]</a> |
          <a href="/wml/favorites-remove.wml?jid=${encodeURIComponent(jid)}">[Remove]</a>
        </p>`;
      }
    }
  }

  body += `
    <p><b>Actions:</b></p>
    <p>
      <a href="/wml/contacts.wml">[Browse Contacts]</a><br/>
      <a href="/wml/groups.wml">[Browse Groups]</a>
    </p>
    <p><a href="/wml/home.wml" accesskey="0">[0] Home</a></p>
  `;

  sendWml(res, card("favorites", "Favorites", body));
});

// Remove from favorites
app.get("/wml/favorites-remove.wml", (req, res) => {
  const jid = req.query.jid;
  const back = req.query.back || "favorites";

  if (jid) {
    removeFavorite(jid);
  }

  // Smart redirect based on 'back' parameter
  const redirectMap = {
    contact: `/wml/contact.wml?jid=${encodeURIComponent(jid)}`,
    group: `/wml/group.view.wml?gid=${encodeURIComponent(jid)}`,
    chat: `/wml/chat.wml?jid=${encodeURIComponent(jid)}`,
    favorites: "/wml/favorites.wml"
  };

  res.redirect(redirectMap[back] || "/wml/favorites.wml");
});

// Add to favorites
app.get("/wml/favorites-add.wml", (req, res) => {
  const jid = req.query.jid;
  const back = req.query.back || "chat";
  let message = "Error";
  let linkBack = "/wml/contacts.wml";

  if (jid) {
    const isGroup = jid.endsWith('@g.us');

    if (addFavorite(jid)) {
      const contact = contactStore.get(jid);
      const name = contact?.name || contact?.notify || contact?.subject || jidFriendly(jid);
      message = `${esc(name)} added to favorites!`;

      // Smart back link based on context
      if (back === "contact") {
        linkBack = `/wml/contact.wml?jid=${encodeURIComponent(jid)}`;
      } else if (back === "group") {
        linkBack = `/wml/group.view.wml?gid=${encodeURIComponent(jid)}`;
      } else if (back === "chat") {
        linkBack = `/wml/chat.wml?jid=${encodeURIComponent(jid)}`;
      }
    } else {
      message = "Already in favorites";
      linkBack = `/wml/favorites.wml`;
    }
  }

  const body = `
    <p><b>${message}</b></p>
    <p>
      <a href="${linkBack}" accesskey="1">[1] Back</a><br/>
      <a href="/wml/favorites.wml" accesskey="2">[2] View Favorites</a><br/>
      <a href="/wml/home.wml" accesskey="0">[0] Home</a>
    </p>
  `;

  sendWml(res, card("fav-add", "Success", body));
});

// Enhanced Contacts with search and pagination

app.get("/wml/contacts.wml", (req, res) => {
  const userAgent = req.headers["user-agent"] || "";

  // Usa req.query per GET. Se il form usa POST, i dati sarebbero in req.body.
  // La <go> con method="get" mette i dati in query string.
  const query = req.query;

  const page = Math.max(1, parseInt(query.page || "1"));
  let limit = Math.max(1, Math.min(20, parseInt(query.limit || "10")));

  // Limiti più restrittivi per dispositivi WAP 1.0
  if (userAgent.includes("Nokia") || userAgent.includes("UP.Browser")) {
    limit = Math.min(5, limit); // Max 5 elementi per pagina
  }

  const search = query.q || "";

  let contacts = Array.from(contactStore.values());

  // Applica filtro di ricerca
  if (search) {
    const searchLower = search.toLowerCase();
    contacts = contacts.filter((c) => {
      const name = (c.name || c.notify || c.verifiedName || "").toLowerCase();
      const number = c.id.replace("@s.whatsapp.net", "");
      return name.includes(searchLower) || number.includes(searchLower);
    });
  }

  // Sort contacts by last message timestamp (most recent first)
  contacts.sort((a, b) => {
    const messagesA = chatStore.get(a.id) || [];
    const messagesB = chatStore.get(b.id) || [];

    const lastMessageA = messagesA.length > 0 ? messagesA[messagesA.length - 1] : null;
    const lastMessageB = messagesB.length > 0 ? messagesB[messagesB.length - 1] : null;

    const timestampA = lastMessageA ? Number(lastMessageA.messageTimestamp) : 0;
    const timestampB = lastMessageB ? Number(lastMessageB.messageTimestamp) : 0;

    return timestampB - timestampA; // Most recent first
  });

  const total = contacts.length;
  const start = (page - 1) * limit;
  const items = contacts.slice(start, start + limit);

  // Funzione di escaping sicura per WML
  const escWml = (text) =>
    (text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&apos;");

  // Header della pagina
  const searchHeader = search
    ? `<p><b>Risultati per:</b> ${escWml(search)} (${total})</p>`
    : `<p><b>Tutti i contatti</b> (${total})</p>`;

  // Lista contatti
  const list =
    items
      .map((c, idx) => {
        const name = c.name || c.notify || c.verifiedName || "Sconosciuto";
        const jid = c.id;
        const number = jidFriendly(jid);
        return `<p>${start + idx + 1}. ${escWml(name)}<br/>
      <small>${escWml(number)}</small><br/>
      <a href="/wml/contact.wml?jid=${encodeURIComponent(jid)}">[Dettagli]</a> 
      <a href="/wml/chat.wml?jid=${encodeURIComponent(
        jid
      )}&amp;limit=10">[Chat]</a></p>`;
      })
      .join("") || "<p>Nessun contatto trovato.</p>";

  // Paginazione con First/Last e numeri di pagina
  const totalPages = Math.ceil(total / limit) || 1;

  const firstPage =
    page > 1
      ? `<a href="/wml/contacts.wml?page=1&amp;limit=${limit}&amp;q=${encodeURIComponent(
          search
        )}">[First]</a>`
      : "";

  const prevPage =
    page > 1
      ? `<a href="/wml/contacts.wml?page=${
          page - 1
        }&amp;limit=${limit}&amp;q=${encodeURIComponent(search)}">[Prev]</a>`
      : "";

  const nextPage =
    page < totalPages
      ? `<a href="/wml/contacts.wml?page=${
          page + 1
        }&amp;limit=${limit}&amp;q=${encodeURIComponent(search)}">[Next]</a>`
      : "";

  const lastPage =
    page < totalPages
      ? `<a href="/wml/contacts.wml?page=${totalPages}&amp;limit=${limit}&amp;q=${encodeURIComponent(
          search
        )}">[Last]</a>`
      : "";

  // numeri di pagina (mostra ±2 intorno alla pagina corrente)
  let pageNumbers = "";
  const startPage = Math.max(1, page - 2);
  const endPage = Math.min(totalPages, page + 2);
  for (let p = startPage; p <= endPage; p++) {
    if (p === page) {
      pageNumbers += `<b>[${p}]</b> `;
    } else {
      pageNumbers += `<a href="/wml/contacts.wml?page=${p}&amp;limit=${limit}&amp;q=${encodeURIComponent(
        search
      )}">${p}</a> `;
    }
  }

  const pagination = `
    <p>
      ${firstPage} ${firstPage && prevPage ? "" : ""} ${prevPage}
      ${pageNumbers}
      ${nextPage} ${nextPage && lastPage ? "" : ""} ${lastPage}
    </p>`;

  // Form di ricerca semplificato
  const searchForm = `
    <p><b>Cerca contatti:</b></p>
    <p>
      <input name="q" title="Cerca..." value="${escWml(
        search
      )}" emptyok="true" size="15" maxlength="30"/>
      <do type="accept" label="Cerca">
        <go href="/wml/contacts.wml" method="get">
          <postfield name="q" value="$(q)"/>
          <postfield name="page" value="1"/>
          <postfield name="limit" value="${limit}"/>
        </go>
      </do>
    </p>`;

  // Corpo della card WML
  const body = `
    <p><b>Contatti - Pagina ${page}/${Math.ceil(total / limit) || 1}</b></p>
    ${searchHeader}
    ${searchForm}
    ${list}
    ${pagination}
    <p>
      <a href="/wml/home.wml">[Home]</a> 
      <a href="/wml/chats.wml">[Chats]</a>
    </p>
    <do type="accept" label="Aggiorna">
      <go href="/wml/contacts.wml?page=${page}&amp;limit=${limit}&amp;q=${encodeURIComponent(
    search
  )}"/>
    </do>
   `;

  // Crea la stringa WML completa
  const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <head>
    <meta http-equiv="Cache-Control" content="max-age=0"/>
  </head>
  <card id="contacts" title="Contatti">
    ${body}
  </card>
</wml>`;

  // --- MODIFICHE CHIAVE PER LA COMPATIBILITÀ ---

  // 1. Imposta gli header per WAP 1.0 con la codifica corretta (ISO-8859-1)
  res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");

  // 2. Codifica l'intera stringa WML in un buffer ISO-8859-1
  const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");

  // 3. Invia il buffer codificato
  res.send(encodedBuffer);
});

// Enhanced Contact Detail page with WTAI integration
app.get("/wml/contact.wml", async (req, res) => {
  try {
    if (!sock) throw new Error("Not connected");
    const jid = formatJid(req.query.jid || "");
    const contact = contactStore.get(jid);
    const number = jidFriendly(jid);

    // Try to fetch additional info
    let status = null;
    let businessProfile = null;

    try {
      status = await sock.fetchStatus(jid);
      businessProfile = await sock.getBusinessProfile(jid);
    } catch (e) {
      // Silently fail for these optional features
    }

    const body = `
      <p><b>${esc(
        contact?.name ||
          contact?.notify ||
          contact?.verifiedName ||
          "Unknown Contact"
      )}</b></p>
      <p>Number: ${esc(number)}</p>
      <p>JID: <small>${esc(jid)}</small></p>
      ${status ? `<p>Status: <em>${esc(status.status || "")}</em></p>` : ""}
      ${businessProfile ? "<p><b>[BUSINESS]</b></p>" : ""}
      
      <p><b>Quick Actions:</b></p>
      <p>
        <a href="wtai://wp/mc;${number}" accesskey="1">[1] Call</a><br/>
        <a href="wtai://wp/sms;${number};" accesskey="2">[2] SMS</a><br/>
        <a href="wtai://wp/ap;${esc(
          contact?.name || number
        )};${number}" accesskey="3">[3] Add to Phone</a><br/>
      </p>
      
      <p><b>WhatsApp Actions:</b></p>
      <p>
        <a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;limit=15" accesskey="4">[4] Open Chat</a><br/>
        <a href="/wml/send-quick.wml?to=${encodeURIComponent(
          jid
        )}" accesskey="5">[5] Send Message</a><br/>
        ${
          isFavorite(jid)
            ? `<a href="/wml/favorites-remove.wml?jid=${encodeURIComponent(
                jid
              )}&amp;back=contact" accesskey="6">[6] Remove from Favorites</a><br/>`
            : `<a href="/wml/favorites-add.wml?jid=${encodeURIComponent(
                jid
              )}&amp;back=contact" accesskey="6">[6] Add to Favorites</a><br/>`
        }
        <a href="/wml/block.wml?jid=${encodeURIComponent(
          jid
        )}" accesskey="7">[7] Block</a><br/>
        <a href="/wml/unblock.wml?jid=${encodeURIComponent(
          jid
        )}" accesskey="8">[8] Unblock</a><br/>
      </p>
      
      ${navigationBar()}
      
      <do type="accept" label="Chat">
        <go href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;limit=15"/>
      </do>
      <do type="options" label="Call">
        <go href="wtai://wp/mc;${number}"/>
      </do>
    `;

    sendWml(res, card("contact", "Contact Info", body));
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Error",
        [e.message || "Failed to load contact"],
        "/wml/contacts.wml"
      )
    );
  }
});
/*
app.get('/wml/chat.wml', async (req, res) => {
  const userAgent = req.headers['user-agent'] || ''
  const isOldNokia = /Nokia|Series40|MAUI|UP\.Browser/i.test(userAgent)
  
  const raw = req.query.jid || ''
  const jid = formatJid(raw)
  const offset = Math.max(0, parseInt(req.query.offset || '0'))
  const search = (req.query.search || '').trim().toLowerCase()
  
  // Very small limits for Nokia 7210
  const limit = isOldNokia ? 3 : 10
  
  // Load chat history if missing
  if ((!chatStore.get(jid) || chatStore.get(jid).length === 0) && sock) {
    try {
      await loadChatHistory(jid, limit * 3)
    } catch (e) {
      logger.warn(`Failed to load chat history for ${jid}: ${e.message}`)
    }
  }
  
  let allMessages = (chatStore.get(jid) || []).slice()
  
  // Sort by timestamp - MOST RECENT FIRST
  allMessages.sort((a, b) => {
    const tsA = Number(a.messageTimestamp) || 0
    const tsB = Number(b.messageTimestamp) || 0
    return tsB - tsA // Most recent first
  })
  
  // Apply search filter if present
  if (search) {
    allMessages = allMessages.filter(m => (messageText(m) || '').toLowerCase().includes(search))
  }
  
  const total = allMessages.length
  const items = allMessages.slice(offset, offset + limit)
  
  const contact = contactStore.get(jid)
  const chatName = contact?.name || contact?.notify || contact?.verifiedName || jidFriendly(jid)
  const number = jidFriendly(jid)
  const isGroup = jid.endsWith('@g.us')
  
  // Simple escaping for Nokia 7210
  const esc = text => (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  
  // Simple truncate
  const truncate = (text, maxLength) => {
    if (!text) return ''
    if (text.length <= maxLength) return text
    return text.substring(0, maxLength - 3) + '...'
  }
  
  // Simple timestamp for Nokia
  const formatTime = (timestamp) => {
    const date = new Date(Number(timestamp) * 1000)
    if (isNaN(date.getTime())) return ''
    
    const day = date.getDate().toString().padStart(2, '0')
    const month = (date.getMonth() + 1).toString().padStart(2, '0')
    const hours = date.getHours().toString().padStart(2, '0')
    const mins = date.getMinutes().toString().padStart(2, '0')
    
    return `${day}/${month} ${hours}:${mins}`
  }
  
  let messageList = ''
  
  if (items.length === 0) {
    messageList = '<p>No messages</p>'
  } else {
    messageList = items.map((m, idx) => {
      const who = m.key.fromMe ? 'Me' : (chatName.length > 10 ? chatName.substring(0, 10) : chatName)
      const time = formatTime(m.messageTimestamp)
      const msgNumber = idx + 1
      const mid = m.key.id
      
      // Handle different message types for Nokia
      let text = ''
      let mediaLink = ''
      
      if (m.message) {
        if (m.message.imageMessage) {
          const img = m.message.imageMessage
          const size = Math.round((img.fileLength || 0) / 1024)
          text = `[IMG ${size}KB]`
          if (img.caption) text += ` ${truncate(img.caption, 30)}`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View IMG]</a>`
          
        } else if (m.message.videoMessage) {
          const vid = m.message.videoMessage
          const size = Math.round((vid.fileLength || 0) / 1024)
          text = `[VID ${size}KB]`
          if (vid.caption) text += ` ${truncate(vid.caption, 30)}`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View VID]</a>`
          
        } else if (m.message.audioMessage) {
          const aud = m.message.audioMessage
          const size = Math.round((aud.fileLength || 0) / 1024)
          const duration = aud.seconds || 0
          text = `[AUD ${size}KB ${duration}s]`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View AUD]</a>`
          
        } else if (m.message.documentMessage) {
          const doc = m.message.documentMessage
          const size = Math.round((doc.fileLength || 0) / 1024)
          const filename = doc.fileName || 'file'
          text = `[DOC ${size}KB] ${truncate(filename, 20)}`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View DOC]</a>`
          
        } else if (m.message.stickerMessage) {
          text = '[STICKER]'
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View STK]</a>`
          
        } else {
          text = truncate(messageText(m) || '', 50)
        }
      } else {
        text = truncate(messageText(m) || '', 50)
      }
      
      return `<p>${msgNumber}. ${esc(who)} (${time})<br/>${esc(text)}${mediaLink}</p>`
    }).join('')
  }
  
  // Simple navigation for Nokia
  const olderOffset = offset + limit
  const olderLink = olderOffset < total ? 
    `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;offset=${olderOffset}&amp;search=${encodeURIComponent(search)}" accesskey="2">2-Older</a></p>` : ''
  
  const newerOffset = Math.max(0, offset - limit)
  const newerLink = offset > 0 ? 
    `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;offset=${newerOffset}&amp;search=${encodeURIComponent(search)}" accesskey="3">3-Newer</a></p>` : ''
  
  // Simple search for Nokia
  const searchBox = search ? 
    `<p>Search: ${esc(search)}</p><p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}">Clear</a></p>` : 
    `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;search=prompt">Search</a></p>`
  
  const body = `<p>${esc(chatName.length > 15 ? chatName.substring(0, 15) : chatName)}</p>
<p>Msgs ${offset + 1}-${Math.min(offset + limit, total)}/${total}</p>
${searchBox}
${messageList}
${newerLink}
${olderLink}
<p><a href="/wml/send.text.wml?to=${encodeURIComponent(jid)}" accesskey="1">1-Send</a></p>
<p><a href="/wml/chats.wml" accesskey="0">0-Back</a></p>`
  
  // Nokia 7210 compatible WML 1.1
  const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Con\l" content="max-age=0"/></head>
<card id="chat" title="Chat">
${body}
</card>
</wml>`
  
  // Nokia 7210 headers
  res.setHeader('Content-Type', 'text/vnd.wap.wml; charset=iso-8859-1')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Pragma', 'no-cache')
  
  const encodedBuffer = iconv.encode(wmlOutput, 'iso-8859-1')
  res.send(encodedBuffer)
})*/
/*
// Route per scaricare media - compatibile Nokia 7210
app.get('/wml/chat.wml', async (req, res) => {
  const userAgent = req.headers['user-agent'] || ''
  const isOldNokia = /Nokia|Series40|MAUI|UP\.Browser/i.test(userAgent)
  
  const raw = req.query.jid || ''
  const jid = formatJid(raw)
  const offset = Math.max(0, parseInt(req.query.offset || '0'))
  const search = (req.query.search || '').trim().toLowerCase()
  
  // Very small limits for Nokia 7210
  const limit = isOldNokia ? 3 : 10
  
  // Load chat history if missing
  if ((!chatStore.get(jid) || chatStore.get(jid).length === 0) && sock) {
    try {
      await loadChatHistory(jid, limit * 3)
    } catch (e) {
      logger.warn(`Failed to load chat history for ${jid}: ${e.message}`)
    }
  }
  
  let allMessages = (chatStore.get(jid) || []).slice()
  
  // Sort by timestamp - MOST RECENT FIRST
  allMessages.sort((a, b) => {
    const tsA = Number(a.messageTimestamp) || 0
    const tsB = Number(b.messageTimestamp) || 0
    return tsB - tsA // Most recent first
  })
  
  // Apply search filter if present
  if (search) {
    allMessages = allMessages.filter(m => (messageText(m) || '').toLowerCase().includes(search))
  }
  
  const total = allMessages.length
  const items = allMessages.slice(offset, offset + limit)
  
  const contact = contactStore.get(jid)
  const chatName = contact?.name || contact?.notify || contact?.verifiedName || jidFriendly(jid)
  const number = jidFriendly(jid)
  const isGroup = jid.endsWith('@g.us')
  
  // Simple escaping for Nokia 7210
  const esc = text => (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  
  // Simple truncate
  const truncate = (text, maxLength) => {
    if (!text) return ''
    if (text.length <= maxLength) return text
    return text.substring(0, maxLength - 3) + '...'
  }
  
  // Simple timestamp for Nokia
  const formatTime = (timestamp) => {
    const date = new Date(Number(timestamp) * 1000)
    if (isNaN(date.getTime())) return ''
    
    const day = date.getDate().toString().padStart(2, '0')
    const month = (date.getMonth() + 1).toString().padStart(2, '0')
    const hours = date.getHours().toString().padStart(2, '0')
    const mins = date.getMinutes().toString().padStart(2, '0')
    
    return `${day}/${month} ${hours}:${mins}`
  }
  
  let messageList = ''
  
  if (items.length === 0) {
    messageList = '<p>No messages</p>'
  } else {
    messageList = items.map((m, idx) => {
      const who = m.key.fromMe ? 'Me' : (chatName.length > 10 ? chatName.substring(0, 10) : chatName)
      const time = formatTime(m.messageTimestamp)
      const msgNumber = idx + 1
      const mid = m.key.id
      
      // Handle different message types for Nokia
      let text = ''
      let mediaLink = ''
      
      if (m.message) {
        if (m.message.imageMessage) {
          const img = m.message.imageMessage
          const size = Math.round((img.fileLength || 0) / 1024)
          text = `[IMG ${size}KB]`
          if (img.caption) text += ` ${truncate(img.caption, 30)}`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View IMG]</a>`
          
        } else if (m.message.videoMessage) {
          const vid = m.message.videoMessage
          const size = Math.round((vid.fileLength || 0) / 1024)
          text = `[VID ${size}KB]`
          if (vid.caption) text += ` ${truncate(vid.caption, 30)}`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View VID]</a>`
          
        } else if (m.message.audioMessage) {
          const aud = m.message.audioMessage
          const size = Math.round((aud.fileLength || 0) / 1024)
          const duration = aud.seconds || 0
          text = `[AUD ${size}KB ${duration}s]`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View AUD]</a>`
          
        } else if (m.message.documentMessage) {
          const doc = m.message.documentMessage
          const size = Math.round((doc.fileLength || 0) / 1024)
          const filename = doc.fileName || 'file'
          text = `[DOC ${size}KB] ${truncate(filename, 20)}`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View DOC]</a>`
          
        } else if (m.message.stickerMessage) {
          text = '[STICKER]'
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View STK]</a>`
          
        } else {
          text = truncate(messageText(m) || '', 50)
        }
      } else {
        text = truncate(messageText(m) || '', 50)
      }
      
      return `<p>${msgNumber}. ${esc(who)} (${time})<br/>${esc(text)}${mediaLink}</p>`
    }).join('')
  }
  
  // Simple navigation for Nokia
  const olderOffset = offset + limit
  const olderLink = olderOffset < total ? 
    `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;offset=${olderOffset}&amp;search=${encodeURIComponent(search)}" accesskey="2">2-Older</a></p>` : ''
  
  const newerOffset = Math.max(0, offset - limit)
  const newerLink = offset > 0 ? 
    `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;offset=${newerOffset}&amp;search=${encodeURIComponent(search)}" accesskey="3">3-Newer</a></p>` : ''
  
  // Simple search for Nokia
  const searchBox = search ? 
    `<p>Search: ${esc(search)}</p><p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}">Clear</a></p>` : 
    `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;search=prompt">Search</a></p>`
  
  const body = `<p>${esc(chatName.length > 15 ? chatName.substring(0, 15) : chatName)}</p>
<p>Msgs ${offset + 1}-${Math.min(offset + limit, total)}/${total}</p>
${searchBox}
${messageList}
${newerLink}
${olderLink}
<p><a href="/wml/send.text.wml?to=${encodeURIComponent(jid)}" accesskey="1">1-Send</a></p>
<p><a href="/wml/chats.wml" accesskey="0">0-Back</a></p>`
  
  // Nokia 7210 compatible WML 1.1
  const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="max-age=0"/></head>
<card id="chat" title="Chat">
${body}
</card>
</wml>`
  
  // Nokia 7210 headers
  res.setHeader('Content-Type', 'text/vnd.wap.wml; charset=iso-8859-1')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Pragma', 'no-cache')
  
  const encodedBuffer = iconv.encode(wmlOutput, 'iso-8859-1')
  res.send(encodedBuffer)
})*/

app.get("/wml/chat.wml", async (req, res) => {
  const userAgent = req.headers["user-agent"] || "";
  const isOldNokia = true;

  const raw = req.query.jid || "";
  const jid = formatJid(raw);
  const offset = Math.max(0, parseInt(req.query.offset || "0"));
  const search = (req.query.search || "").trim().toLowerCase();

  // Fixed limit to 5 elements per page
  const limit = 5;

  // Load chat history if missing
  if ((!chatStore.get(jid) || chatStore.get(jid).length === 0) && sock) {
    try {
      await loadChatHistory(jid, limit * 5);
    } catch (e) {
      logger.warn(`Failed to load chat history for ${jid}: ${e.message}`);
    }
  }

  let allMessages = (chatStore.get(jid) || []).slice();

  // Sort by timestamp - MOST RECENT FIRST (descending order)
  allMessages.sort((a, b) => {
    const tsA = Number(a.messageTimestamp) || 0;
    const tsB = Number(b.messageTimestamp) || 0;
    return tsB - tsA; // Most recent first
  });

  // Apply search filter if present
  if (search) {
    allMessages = allMessages.filter((m) =>
      (messageText(m) || "").toLowerCase().includes(search)
    );
  }

  const totalMessages = allMessages.length;
  const items = allMessages.slice(offset, offset + limit);

  // Use getContactName for better name resolution
  const chatName = await getContactName(jid, sock);
  const number = jidFriendly(jid);
  const isGroup = jid.endsWith("@g.us");

  // Enhanced escaping that works for both Nokia and modern devices
  const escWml = (text) =>
    (text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&apos;");

  // Truncation function
  const truncate = (text, maxLength) => {
    if (!text) return "";
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength - 3) + "...";
  };

  // Enhanced timestamp formatting with date and time
  const formatMessageTimestamp = (timestamp) => {
    const date = new Date(Number(timestamp) * 1000);
    if (isNaN(date.getTime())) return "Invalid date";

    if (isOldNokia) {
      // Simple format for Nokia: dd/mm hh:mm
      const day = date.getDate().toString().padStart(2, "0");
      const month = (date.getMonth() + 1).toString().padStart(2, "0");
      const hours = date.getHours().toString().padStart(2, "0");
      const mins = date.getMinutes().toString().padStart(2, "0");
      return `${day}/${month} ${hours}:${mins}`;
    } else {
      // Full format for modern devices: 30 Dec 2024 14:30
      const timeStr = date.toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
      });
      const dateStr = date.toLocaleDateString("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      });
      return `${dateStr} ${timeStr}`;
    }
  };

  // Message list with full media support
  let messageList = "";

  if (items.length === 0) {
    messageList = "<p>No messages</p>";
  } else {
    messageList = items
      .map((m, idx) => {
        const who = m.key.fromMe
          ? "Me"
          : isOldNokia
          ? chatName.length > 10
            ? chatName.substring(0, 10)
            : chatName
          : isGroup
          ? m.pushName || "Unknown"
          : chatName;
        const time = formatMessageTimestamp(m.messageTimestamp);
        const msgNumber = idx + 1; // 1 = most recent
        const mid = m.key.id;

        // Handle different message types with full media support
        let text = "";
        let mediaLink = "";

        if (m.message) {
          if (m.message.imageMessage) {
            const img = m.message.imageMessage;
            const size = Math.round((img.fileLength || 0) / 1024);
            text = isOldNokia ? `[IMG ${size}KB]` : `[IMAGE ${size}KB]`;
            if (img.caption)
              text += ` ${truncate(img.caption, isOldNokia ? 30 : 50)}`;

            if (isOldNokia) {
              mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(
                mid
              )}&amp;jid=${encodeURIComponent(jid)}">[View IMG]</a>`;
            } else {
              mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(
                mid
              )}&amp;jid=${encodeURIComponent(
                jid
              )}">[View Image]</a> | <a href="/wml/media/${encodeURIComponent(
                mid
              )}.jpg">[Download]</a>`;
            }
          } else if (m.message.videoMessage) {
            const vid = m.message.videoMessage;
            const size = Math.round((vid.fileLength || 0) / 1024);
            const duration = vid.seconds || 0;
            text = isOldNokia
              ? `[VID ${size}KB]`
              : `[VIDEO ${size}KB, ${duration}s]`;
            if (vid.caption)
              text += ` ${truncate(vid.caption, isOldNokia ? 30 : 50)}`;

            if (isOldNokia) {
              mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(
                mid
              )}&amp;jid=${encodeURIComponent(jid)}">[View VID]</a>`;
            } else {
              mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(
                mid
              )}&amp;jid=${encodeURIComponent(
                jid
              )}">[View Video]</a> | <a href="/wml/media/${encodeURIComponent(
                mid
              )}.mp4">[Download]</a>`;
            }
          } // Nel blocco che genera la lista dei messaggi, sostituisci la gestione degli audio con questo:
          else if (m.message.audioMessage) {
            const aud = m.message.audioMessage;
            const size = Math.round((aud.fileLength || 0) / 1024);
            const duration = aud.seconds || 0;
            text = `[AUDIO ${size}KB ${duration}s]`;

            // Aggiungi il link per la trascrizione se disponibile
            if (
              m &&
              m.transcription &&
              m.transcription !== "[Trascrizione fallita]" &&
              m.transcription !== "[Audio troppo lungo per la trascrizione]"
            ) {
              mediaLink = `<br/>
 
      <a href="/wml/audio-transcription.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(jid)}">[View Transcription]</a>`;
            }
          } else if (m.message.documentMessage) {
            const doc = m.message.documentMessage;
            const size = Math.round((doc.fileLength || 0) / 1024);
            const filename = doc.fileName || "file";
            text = isOldNokia
              ? `[DOC ${size}KB] ${truncate(filename, 20)}`
              : `[DOCUMENT ${size}KB] ${truncate(filename, 40)}`;

            const ext = filename.split(".").pop() || "bin";
            if (isOldNokia) {
              mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(
                mid
              )}&amp;jid=${encodeURIComponent(jid)}">[View DOC]</a>`;
            } else {
              mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(
                mid
              )}&amp;jid=${encodeURIComponent(
                jid
              )}">[View Document]</a> | <a href="/wml/media/${encodeURIComponent(
                mid
              )}.${ext}">[Download]</a>`;
            }
          } else if (m.message.stickerMessage) {
            const sticker = m.message.stickerMessage;
            const size = Math.round((sticker.fileLength || 0) / 1024);
            text = isOldNokia ? "[STICKER]" : `[STICKER ${size}KB]`;

            if (isOldNokia) {
              mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(
                mid
              )}&amp;jid=${encodeURIComponent(jid)}">[View STK]</a>`;
            } else {
              mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(
                mid
              )}&amp;jid=${encodeURIComponent(
                jid
              )}">[View Sticker]</a> | <a href="/wml/media/${encodeURIComponent(
                mid
              )}.webp">[Download]</a>`;
            }
          } else {
            text = truncate(messageText(m) || "", isOldNokia ? 50 : 100);
          }
        } else {
          text = truncate(messageText(m) || "", isOldNokia ? 50 : 100);
        }

        // Format message entry
        if (isOldNokia) {
          return `<p>${msgNumber}. ${escWml(who)} (${time})<br/>${escWml(
            text
          )}${mediaLink}</p>`;
        } else {
          const typeIndicator = m.key.fromMe ? "[OUT]" : "[IN]";
          const isVeryRecent = idx < 3;
          const recentIndicator = isVeryRecent ? "🔥" : "";

          return `<p>${recentIndicator}<b>${msgNumber}. ${typeIndicator} ${escWml(
            who
          )}</b><br/>
          <small><b>Time:</b> ${time}</small><br/>
          <small><b>Message:</b> ${escWml(text)}</small>${mediaLink}<br/>
          <a href="/wml/msg.wml?mid=${encodeURIComponent(
            mid
          )}&amp;jid=${encodeURIComponent(jid)}">[Details]</a> 
          <a href="/wml/send.text.wml?to=${encodeURIComponent(
            jid
          )}&amp;reply=${encodeURIComponent(mid)}">[Reply]</a>
        </p>`;
        }
      })
      .join("");
  }

  // Enhanced navigation with First/Last buttons
  const totalPages = Math.ceil(totalMessages / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  // Calculate navigation offsets
  const firstOffset = 0;
  const lastOffset = Math.max(0, (totalPages - 1) * limit);
  const olderOffset = offset + limit;
  const newerOffset = Math.max(0, offset - limit);

  // Build navigation links
  let navigationLinks = [];

  // First button (only if not on first page)
  if (offset > 0) {
    const firstLink = isOldNokia
      ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${firstOffset}&amp;search=${encodeURIComponent(
          search
        )}" accesskey="7">7-First</a>`
      : `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${firstOffset}&amp;search=${encodeURIComponent(
          search
        )}" accesskey="7">[7] First</a>`;
    navigationLinks.push(firstLink);
  }

  // Newer/Previous button
  if (offset > 0) {
    const newerLink = isOldNokia
      ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${newerOffset}&amp;search=${encodeURIComponent(
          search
        )}" accesskey="3">3-Newer</a>`
      : `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${newerOffset}&amp;search=${encodeURIComponent(
          search
        )}" accesskey="3">[3] Newer</a>`;
    navigationLinks.push(newerLink);
  }

  // Older/Next button
  if (olderOffset < totalMessages) {
    const olderLink = isOldNokia
      ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${olderOffset}&amp;search=${encodeURIComponent(
          search
        )}" accesskey="2">2-Older</a>`
      : `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${olderOffset}&amp;search=${encodeURIComponent(
          search
        )}" accesskey="2">[2] Older</a>`;
    navigationLinks.push(olderLink);
  }

  // Last button (only if not on last page)
  if (offset < lastOffset) {
    const lastLink = isOldNokia
      ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${lastOffset}&amp;search=${encodeURIComponent(
          search
        )}" accesskey="8">8-Last</a>`
      : `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${lastOffset}&amp;search=${encodeURIComponent(
          search
        )}" accesskey="8">[8] Last</a>`;
    navigationLinks.push(lastLink);
  }

  // Numeri di pagina (max 5 visibili: due prima, attuale, due dopo)
  let pageNumbers = "";
  const startPage = Math.max(1, currentPage - 2);
  const endPage = Math.min(totalPages, currentPage + 2);

  for (let p = startPage; p <= endPage; p++) {
    if (p === currentPage) {
      pageNumbers += `<b>[${p}]</b> `;
    } else {
      const offsetForPage = (p - 1) * limit;
      pageNumbers += `<a href="/wml/chat.wml?jid=${encodeURIComponent(
        jid
      )}&amp;offset=${offsetForPage}&amp;search=${encodeURIComponent(
        search
      )}">${p}</a> `;
    }
  }

  // Bottoni First/Last e Prev/Next
  const firstPage =
    currentPage > 1
      ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=0&amp;search=${encodeURIComponent(search)}">[First]</a>`
      : "";

  const prevPage =
    currentPage > 1
      ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${newerOffset}&amp;search=${encodeURIComponent(
          search
        )}">[Previous]</a>`
      : "";

  const nextPage =
    currentPage < totalPages
      ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${olderOffset}&amp;search=${encodeURIComponent(
          search
        )}">[Next]</a>`
      : "";

  const lastPage =
    currentPage < totalPages
      ? `<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;offset=${lastOffset}&amp;search=${encodeURIComponent(
          search
        )}">[Last]</a>`
      : "";

  // Sezione completa di navigazione
  const navigationSection = `
  <p>
    ${firstPage} ${firstPage && prevPage ? "" : ""} ${prevPage}
    ${pageNumbers}
    ${nextPage} ${nextPage && lastPage ? "" : ""} ${lastPage}
  </p>`;

  // Search form adapted to device capability
  const searchForm = isOldNokia
    ? search
      ? `<p>Search: ${escWml(
          search
        )}</p><p><a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}">Clear</a></p>`
      : `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;search=prompt">Search</a></p>`
    : `<p><b>Search Messages:</b></p>
     <p>
       <input name="searchQuery" title="Search..." value="${escWml(
         search
       )}" size="15" maxlength="50"/>
       <do type="accept" label="Search">
         <go href="/wml/chat.wml" method="get">
           <postfield name="jid" value="${escWml(jid)}"/>
           <postfield name="search" value="$(searchQuery)"/>
           <postfield name="offset" value="0"/>
         </go>
       </do>
     </p>
     ${
       search
         ? `<p>Searching: <b>${escWml(
             search
           )}</b> | <a href="/wml/chat.wml?jid=${encodeURIComponent(
             jid
           )}">[Clear]</a></p>`
         : ""
     }`;

  // Page info adapted to device
  const pageInfo = isOldNokia
    ? `<p>Page ${currentPage}/${totalPages}</p><p>Msgs ${offset + 1}-${Math.min(
        offset + limit,
        totalMessages
      )}/${totalMessages}</p>`
    : `<p><b>Page ${currentPage} of ${totalPages}</b></p>
     <p><b>Messages ${offset + 1}-${Math.min(
        offset + limit,
        totalMessages
      )} of ${totalMessages}</b></p>
     <p>Showing 5 messages per page (most recent first)</p>`;

  // Quick actions adapted to device
  const favoriteLink = isFavorite(jid)
    ? `<a href="/wml/favorites-remove.wml?jid=${encodeURIComponent(jid)}&amp;back=chat" accesskey="5">${isOldNokia ? '5-Unfav' : '[5] Remove Favorite'}</a>`
    : `<a href="/wml/favorites-add.wml?jid=${encodeURIComponent(jid)}&amp;back=chat" accesskey="5">${isOldNokia ? '5-Fav' : '[5] Add to Favorites'}</a>`;

  const quickActions = isOldNokia
    ? `<p><a href="/wml/send-quick.wml?to=${encodeURIComponent(
        jid
      )}" accesskey="1">1-Send</a></p>
     <p>${favoriteLink}</p>
     <p><a href="/wml/chats.wml" accesskey="0">0-Back</a></p>`
    : `<p><b>Quick Actions:</b></p>
     <p>
       <a href="/wml/send-quick.wml?to=${encodeURIComponent(
         jid
       )}" accesskey="1">[1] Send Message</a>
       <a href="/wml/contact.wml?jid=${encodeURIComponent(
         jid
       )}" accesskey="4">[4] Contact Info</a>
       ${
         number && !isGroup
           ? ` | <a href="wtai://wp/mc;${number}" accesskey="9">[9] Call</a>`
           : ""
       }
       ${
         number && !isGroup
           ? ` | <a href="wtai://wp/ms;${number};">[SMS]</a>`
           : ""
       }
       <br/>
       ${favoriteLink}
     </p>
     <p>
       <a href="/wml/chats.wml" accesskey="0">[0] Back to Chats</a> |
       <a href="/wml/home.wml" accesskey="*">[*] Home</a>
     </p>`;

  // Build final body
  const chatTitle = isOldNokia
    ? chatName.length > 15
      ? chatName.substring(0, 15)
      : chatName
    : chatName;

  const body = isOldNokia
    ? `<p>${escWml(chatTitle)}</p>
<p>${escWml(number)}</p>
${pageInfo}
${searchForm}
${messageList}
${navigationSection}
${quickActions}`
    : `<p><b>${escWml(chatTitle)}</b> ${isGroup ? "[GROUP]" : "[CHAT]"}</p>
<p>${escWml(number)} | Total: ${totalMessages} messages</p>
${searchForm}
${pageInfo}
${messageList}
${navigationSection}
${quickActions}`;

  // Create WML output with appropriate DOCTYPE
  const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="max-age=0"/></head>
<card id="chat" title="Chat">
${body}
<do type="accept" label="Send">
  <go href="/wml/send-quick.wml?to=${encodeURIComponent(jid)}"/>
</do>
<do type="options" label="Refresh">
  <go href="/wml/chat.wml?jid=${encodeURIComponent(
    jid
  )}&amp;offset=${offset}&amp;search=${encodeURIComponent(search)}"/>
</do>
</card>
</wml>`;

  // Set appropriate headers
  res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");

  const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
  res.send(encodedBuffer);
});
// Route per visualizzare info media - WAP friendly come QR

function encodeMultiByte(value) {
  if (value < 128) {
    return Buffer.from([value]);
  }

  const bytes = [];
  let remaining = value;

  // Encoding multi-byte WBMP standard
  while (remaining >= 128) {
    bytes.unshift(remaining & 0x7f);
    remaining = remaining >> 7;
  }
  bytes.unshift(remaining | 0x80);

  return Buffer.from(bytes);
}

function createWBMP(pixels, width, height) {
  // Header WBMP standard
  const typeField = Buffer.from([0x00]);
  const fixHeader = Buffer.from([0x00]);
  const widthBytes = encodeMultiByte(width);
  const heightBytes = encodeMultiByte(height);

  // Calcola dimensioni data
  const bytesPerRow = Math.ceil(width / 8);
  const dataSize = bytesPerRow * height;
  const data = Buffer.alloc(dataSize, 0x00); // Inizializza a 0 (bianco)

  // Converte pixel in 1-bit
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIndex = y * width + x;
      const grayscale = pixels[pixelIndex]; // Già grayscale

      // WBMP: 1 = nero, 0 = bianco
      const isBlack = grayscale < 128;

      if (isBlack) {
        const byteIndex = y * bytesPerRow + Math.floor(x / 8);
        const bitPosition = 7 - (x % 8);
        data[byteIndex] |= 1 << bitPosition;
      }
    }
  }

  return Buffer.concat([typeField, fixHeader, widthBytes, heightBytes, data]);
}

app.get("/wml/audio-transcription.wml", async (req, res) => {
  try {
    const mid = req.query.mid || "";
    const jid = req.query.jid || "";

    // Trova il messaggio nella chat specifica
    const messages = chatStore.get(jid) || [];
    const targetMessage = messages.find((m) => m.key.id === mid);

    if (!targetMessage) {
      sendWml(
        res,
        resultCard(
          "Errore",
          ["Messaggio non trovato"],
          `/wml/chat.wml?jid=${encodeURIComponent(jid)}&limit=15`
        )
      );
      return;
    }

    // Verifica che sia un messaggio audio
    if (!targetMessage.message?.audioMessage) {
      sendWml(
        res,
        resultCard(
          "Errore",
          ["Questo messaggio non contiene un audio"],
          `/wml/chat.wml?jid=${encodeURIComponent(jid)}&limit=15`
        )
      );
      return;
    }

    const contact = contactStore.get(jid);
    const chatName = contact?.name || contact?.notify || jidFriendly(jid);
    const aud = targetMessage.message.audioMessage;
    const duration = aud.seconds || 0;
    const size = Math.round((aud.fileLength || 0) / 1024);

    // Prepara la trascrizione
    let transcriptionText = "";
    let transcriptionStatus = "";

    if (targetMessage.transcription) {
      if (targetMessage.transcription === "[Trascrizione fallita]") {
        transcriptionStatus = "❌ Trascrizione fallita";
        transcriptionText =
          "Non è stato possibile trascrivere questo messaggio audio.";
      } else if (
        targetMessage.transcription ===
        "[Audio troppo lungo per la trascrizione]"
      ) {
        transcriptionStatus = "⚠️ Audio troppo lungo";
        transcriptionText =
          "Questo messaggio audio supera i 10MB e non può essere trascritto.";
      } else {
        transcriptionStatus = "✅ Trascrizione completata";
        transcriptionText = targetMessage.transcription;
      }
    } else {
      transcriptionStatus = "⏳ In elaborazione";
      transcriptionText = "La trascrizione è in corso...";
    }

    const body = `
      <p><b>Trascrizione Messaggio Audio</b></p>
      
      <p><b>Da:</b> ${esc(chatName)}</p>
      <p><b>Durata:</b> ${duration} secondi</p>
      <p><b>Dimensione:</b> ${size}KB</p>
      <p><b>Stato:</b> ${transcriptionStatus}</p>
      
      <p><b>Trascrizione:</b></p>
      <p>${esc(transcriptionText)}</p>
      
      <p><b>Azioni:</b></p>
      <p>
       
        <a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}&amp;limit=15" accesskey="0">[0] Torna alla Chat</a>
      </p>
      
      <do type="accept" label="Ascolta">
        <go href="/wml/media/${encodeURIComponent(mid)}.wav"/>
      </do>
      <do type="options" label="Indietro">
        <go href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;limit=15"/>
      </do>
    `;

    sendWml(res, card("audio-transcription", "Trascrizione Audio", body));
  } catch (error) {
    logger.error("Audio transcription page error:", error);
    sendWml(
      res,
      resultCard(
        "Errore",
        [error.message || "Errore durante il caricamento della trascrizione"],
        "/wml/home.wml"
      )
    );
  }
});

app.get("/wml/media/:filename", async (req, res) => {
  try {
    const filename = req.params.filename;
    const messageId = filename.split(".")[0];
    const forceWbmp = filename.endsWith(".wbmp");

    // Debug logging
    console.log(`Richiesta media: ${filename}, WBMP: ${forceWbmp}`);

    // Find message in all chats
    let targetMessage = null;

    for (const [jid, messages] of chatStore.entries()) {
      const found = messages.find(
        (m) => m.key.id === decodeURIComponent(messageId)
      );
      if (found) {
        targetMessage = found;
        break;
      }
    }

    if (!targetMessage || !sock) {
      console.log("Messaggio non trovato o sock non disponibile");
      res.status(404).send("Media not found");
      return;
    }

    // Download media
    let mediaData = null;
    let mimeType = "application/octet-stream";
    let filename_out = filename;
    let isImage = false;

    if (targetMessage.message?.imageMessage) {
      mediaData = await downloadMediaMessage(
        targetMessage,
        "buffer",
        {},
        {
          logger,
          reuploadRequest: sock.updateMediaMessage,
        }
      );
      mimeType = forceWbmp ? "image/vnd.wap.wbmp" : "image/jpeg";
      filename_out = forceWbmp
        ? `image_${messageId}.wbmp`
        : `image_${messageId}.jpg`;
      isImage = true;
    } else if (targetMessage.message?.stickerMessage) {
      mediaData = await downloadMediaMessage(
        targetMessage,
        "buffer",
        {},
        {
          logger,
          reuploadRequest: sock.updateMediaMessage,
        }
      );
      mimeType = forceWbmp ? "image/vnd.wap.wbmp" : "image/jpeg";
      filename_out = forceWbmp
        ? `sticker_${messageId}.wbmp`
        : `sticker_${messageId}.jpg`;
      isImage = true;
    } else if (targetMessage.message?.documentMessage) {
      const doc = targetMessage.message.documentMessage;
      mediaData = await downloadMediaMessage(
        targetMessage,
        "buffer",
        {},
        {
          logger,
          reuploadRequest: sock.updateMediaMessage,
        }
      );

      if (doc.mimetype && doc.mimetype.startsWith("image/")) {
        isImage = true;
        mimeType = forceWbmp ? "image/vnd.wap.wbmp" : "image/jpeg";
        filename_out = forceWbmp
          ? `doc_${messageId}.wbmp`
          : `doc_${messageId}.jpg`;
      } else {
        mimeType = doc.mimetype || "application/octet-stream";
        filename_out = doc.fileName || `document_${messageId}.bin`;
      }
    }

    if (!mediaData) {
      console.log("Impossibile scaricare il media");
      res.status(404).send("Could not download media");
      return;
    }

    console.log(
      `Media scaricato, dimensione: ${mediaData.length} bytes, isImage: ${isImage}`
    );

    // Processa le immagini
    if (isImage) {
      try {
        const maxSizeBytes = 35 * 1024; // 35KB limit

        if (forceWbmp) {
          console.log("Conversione a WBMP per dispositivi WAP...");

          // Ottieni metadata dell'immagine originale per controllo dimensioni
          const originalMetadata = await sharp(mediaData).metadata();
          console.log(
            `Immagine originale: ${originalMetadata.width}x${originalMetadata.height}`
          );

          // Calcola dimensioni ottimali per dispositivi WAP - PIÙ LARGHE
          const maxWidth = 240; // Larghezza aumentata per schermi più grandi
          const maxHeight = 280; // Altezza aumentata per massima visibilità

          let targetWidth = originalMetadata.width;
          let targetHeight = originalMetadata.height;

          // Se l'immagine è troppo grande, ridimensiona mantenendo proporzioni
          if (targetWidth > maxWidth || targetHeight > maxHeight) {
            const widthRatio = maxWidth / targetWidth;
            const heightRatio = maxHeight / targetHeight;
            const ratio = Math.min(widthRatio, heightRatio);

            targetWidth = Math.round(targetWidth * ratio);
            targetHeight = Math.round(targetHeight * ratio);
            console.log(
              `Ridimensionando a ${targetWidth}x${targetHeight} per compatibilità WAP estesa`
            );
          }

          // Processamento pre-resize per preservare i dettagli
          let processedImage = sharp(mediaData)
            .linear(1.2, -(128 * 0.2)) // Aumenta contrasto lineare
            .modulate({
              brightness: 1.05,
              saturation: 0.8,
              hue: 0,
            });

          // Applica resize solo se necessario
          if (
            targetWidth !== originalMetadata.width ||
            targetHeight !== originalMetadata.height
          ) {
            processedImage = processedImage.resize(targetWidth, targetHeight, {
              kernel: sharp.kernel.lanczos3,
              fit: "contain",
              position: "center",
              background: { r: 255, g: 255, b: 255, alpha: 1 },
            });
          }

          // Converti in grayscale con altissima qualità
          const { data: pixels, info } = await processedImage
            .greyscale()
            .linear(1.3, -30) // Contrasto aggressivo pre-normalizzazione
            .normalise({
              lower: 1, // Percentile inferiore per bianco puro
              upper: 99, // Percentile superiore per nero puro
            })
            .sharpen({
              sigma: 1.5, // Raggio sharpening
              flat: 2, // Soglia flat areas
              jagged: 3, // Soglia jagged areas
            })
            .threshold(128, {
              greyscale: true,
              grayscale: true,
            }) // Soglia ottimale per bianco/nero
            .raw()
            .toBuffer({ resolveWithObject: true });

          console.log(
            `Pixel estratti: ${pixels.length}, dimensioni finali: ${info.width}x${info.height}`
          );

          // Crea WBMP con dimensioni ottimizzate per WAP
          mediaData = createWBMP(pixels, info.width, info.height);
          console.log(
            `WBMP esteso per WAP ${info.width}x${info.height} creato, dimensione finale: ${mediaData.length} bytes`
          );
        } else {
          console.log("Conversione a JPEG...");
          let quality = 80;

          do {
            mediaData = await sharp(mediaData).jpeg({ quality }).toBuffer();

            console.log(
              `JPEG creato, dimensione: ${mediaData.length} bytes (limite: ${maxSizeBytes})`
            );

            // Se troppo grande, riduci solo la qualità
            if (mediaData.length > maxSizeBytes) {
              if (quality > 10) {
                quality = Math.max(10, quality - 10);
                console.log(`Riducendo qualità a ${quality}%`);
              } else {
                console.log(
                  `Qualità minima raggiunta (${quality}%), dimensione finale: ${mediaData.length} bytes`
                );
                break; // Cannot reduce quality further
              }
            } else {
              console.log(
                `JPEG ottimizzato con successo: ${mediaData.length} bytes`
              );
              break;
            }
          } while (mediaData.length > maxSizeBytes && quality >= 10);
        }

        // Controllo finale dimensione
        if (mediaData.length > maxSizeBytes) {
          console.log(
            `ATTENZIONE: File ancora sopra il limite: ${mediaData.length} bytes (max: ${maxSizeBytes})`
          );
        }
      } catch (conversionError) {
        console.error("Errore conversione immagine:", conversionError);
        // Continua con l'immagine originale
      }
    }

    // Controllo finale per tutti i file (anche non-immagini)
    const maxSizeBytes = 35 * 1024; // 35KB limit
    if (mediaData.length > maxSizeBytes) {
      console.log(
        `ATTENZIONE: File troppo grande: ${mediaData.length} bytes (max: ${maxSizeBytes})`
      );
      // Opzionalmente potresti ritornare un errore:
      // res.status(413).send('File too large')
      // return
    }

    // Headers
    if (forceWbmp && isImage) {
      console.log("Invio come WBMP");
      // Headers WAP semplificati
      res.setHeader("Content-Type", "image/vnd.wap.wbmp");
      res.setHeader("Content-Length", mediaData.length);
      res.setHeader("Cache-Control", "no-cache"); // Disabilita cache per debug
    } else if (isImage) {
      console.log("Invio come JPEG");
      res.setHeader("Content-Type", "image/jpeg");
      res.setHeader("Content-Length", mediaData.length);
      res.setHeader("Cache-Control", "no-cache");
    } else {
      console.log("Invio come download");
      res.setHeader("Content-Type", mimeType);
      res.setHeader(
        "Content-Disposition",
        `attachment; filename="${filename_out}"`
      );
      res.setHeader("Content-Length", mediaData.length);
    }

    console.log(
      `Invio risposta: ${mediaData.length} bytes, Content-Type: ${res.getHeader(
        "Content-Type"
      )}`
    );
    res.send(mediaData);
  } catch (error) {
    console.error("Errore generale:", error);
    res.status(500).send("Internal Server Error");
  }
});
// Route esistente modificata per includere link alla pagina WBMP dedicata
app.get("/wml/media-info.wml", async (req, res) => {
  try {
    const messageId = req.query.mid || "";
    const jid = req.query.jid || "";

    // Find message in the specific chat
    const messages = chatStore.get(jid) || [];
    const targetMessage = messages.find((m) => m.key.id === messageId);

    const contact = contactStore.get(jid);
    const chatName = contact?.name || contact?.notify || jidFriendly(jid);

    // Simple escaping
    const esc = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    const body = targetMessage
      ? (() => {
          if (targetMessage.message?.imageMessage) {
            const img = targetMessage.message.imageMessage;
            const size = Math.round((img.fileLength || 0) / 1024);
            const caption = img.caption
              ? `<p><b>Caption:</b> ${esc(img.caption)}</p>`
              : "";

            return `<p><b>Image Message</b></p>
<p>From: ${esc(chatName)}</p>
<p>Size: ${size}KB</p>
<p>Type: ${img.mimetype || "image/jpeg"}</p>
${caption}
<p><b>View Image (WBMP):</b></p>
<p>
<a href="/wml/view-image.wml?mid=${encodeURIComponent(
              messageId
            )}&amp;jid=${encodeURIComponent(jid)}" accesskey="5">[5] View in Page</a>
</p>
<p><small>WAP only supports WBMP inline.<br/>
JPG/PNG are download-only.</small></p>
<p><b>Download Formats:</b></p>
<p>
<a href="/wml/media/${encodeURIComponent(messageId)}.jpg" accesskey="6">[6] Download JPG</a><br/>
<a href="/wml/media/${encodeURIComponent(messageId)}.png" accesskey="7">[7] Download PNG</a><br/>
<a href="/wml/media/${encodeURIComponent(messageId)}.wbmp" accesskey="8">[8] Download WBMP</a>
</p>
`;
          } else if (targetMessage.message?.videoMessage) {
            const vid = targetMessage.message.videoMessage;
            const size = Math.round((vid.fileLength || 0) / 1024);
            const duration = vid.seconds || 0;
            const caption = vid.caption
              ? `<p><b>Caption:</b> ${esc(vid.caption)}</p>`
              : "";

            return `<p><b>Video Message</b></p>
<p>From: ${esc(chatName)}</p>
<p>Size: ${size}KB | Duration: ${duration}s</p>
<p>Type: ${vid.mimetype || "video/mp4"}</p>
${caption}
<p><b>WAP Playback (WBMP):</b></p>
<p>
<a href="/wml/view-video.wml?mid=${encodeURIComponent(
              messageId
            )}&amp;jid=${encodeURIComponent(jid)}" accesskey="5">[5] Play Frame-by-Frame (1 FPS)</a>
</p>
<p><small>Displays WBMP frames inline.<br/>
Press [7]/[8] to download JPG/PNG.</small></p>
<p><b>Download Video:</b></p>
<p>
<a href="/wml/media/${encodeURIComponent(messageId)}.3gp">[3GP Mobile]</a><br/>
<a href="/wml/media/${encodeURIComponent(messageId)}.original.mp4">[MP4 Original]</a>
</p>`;
          } // Nel blocco che gestisce i messaggi audio, aggiungi questo:
          else if (targetMessage.message?.audioMessage) {
            const aud = targetMessage.message.audioMessage;
            const size = Math.round((aud.fileLength || 0) / 1024);
            const duration = aud.seconds || 0;

            body = `<p><b>Audio Message</b></p>
    <p>From: ${esc(chatName)}</p>
    <p>Size: ${size}KB | Duration: ${duration}s</p>
    <p>Type: ${aud.mimetype || "audio/ogg"}</p>
    
    ${
      targetMessage.transcription &&
      targetMessage.transcription !== "[Trascrizione fallita]" &&
      targetMessage.transcription !== "[Audio troppo lungo per la trascrizione]"
        ? `<p><b>Trascrizione disponibile:</b></p>
      <p><a href="/wml/audio-transcription.wml?mid=${encodeURIComponent(
        messageId
      )}&amp;jid=${encodeURIComponent(
            jid
          )}" accesskey="4">[4] View Transcription</a></p>`
        : ""
    }
    
    <p><b>Download Options:</b></p>
    <p>
      <a href="/wml/media/${encodeURIComponent(
        messageId
      )}.wav" accesskey="5">[5] Download WAV</a> |
      <a href="/wml/media/${encodeURIComponent(
        messageId
      )}.ogg" accesskey="6">[6] Download OGG</a>
    </p>`;
          } else if (targetMessage.message?.documentMessage) {
            const doc = targetMessage.message.documentMessage;
            const size = Math.round((doc.fileLength || 0) / 1024);
            const filename = doc.fileName || "document";
            const ext = filename.split(".").pop() || "bin";

            return `<p><b>Document</b></p>
<p>From: ${esc(chatName)}</p>
<p>Name: ${esc(filename)}</p>
<p>Size: ${size}KB</p>
<p>Type: ${doc.mimetype || "unknown"}</p>
<p><b>Download Options:</b></p>
<p>
<a href="/wml/media/${encodeURIComponent(messageId)}.${ext}">[Original]</a> 
<a href="/wml/media-text/${encodeURIComponent(messageId)}">[Text View]</a>
</p>`;
          } else if (targetMessage.message?.stickerMessage) {
            const sticker = targetMessage.message.stickerMessage;
            const size = Math.round((sticker.fileLength || 0) / 1024);

            return `<p><b>Sticker</b></p>
<p>From: ${esc(chatName)}</p>
<p>Size: ${size}KB</p>
<p>Type: image/webp</p>
<p><b>Nokia Compatible:</b></p>
<p>
<a href="/wml/view-wbmp.wml?mid=${encodeURIComponent(
              messageId
            )}&amp;jid=${encodeURIComponent(jid)}">[WBMP View]</a> 
<a href="/wml/media/${encodeURIComponent(messageId)}.jpg">[Small JPG]</a>
</p>
<p><b>Other Formats:</b></p>
<p>
<a href="/wml/media/${encodeURIComponent(
              messageId
            )}.original.webp">[Original WEBP]</a> 
<a href="/wml/media/${encodeURIComponent(messageId)}.png">[PNG]</a>
</p>`;
          }

          return "<p><b>Unknown Media Type</b></p>";
        })()
      : `<p><b>Media Not Found</b></p>
<p>Message may have been deleted</p>
<p>Please try refreshing the chat</p>`;

    const body_full = `${body}
<p>
<a href="/wml/chat.wml?jid=${encodeURIComponent(
      jid
    )}" accesskey="0">[0] Back to Chat</a> 
<a href="/wml/chats.wml" accesskey="9">[9] All Chats</a>
</p>
<do type="accept" label="Back">
<go href="/wml/chat.wml?jid=${encodeURIComponent(jid)}"/>
</do>
<do type="options" label="Refresh">
<go href="/wml/media-info.wml?mid=${encodeURIComponent(
      messageId
    )}&amp;jid=${encodeURIComponent(jid)}"/>
</do>`;

    const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="max-age=0"/></head>
<card id="media" title="Media Info">
${body_full}
</card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Pragma", "no-cache");

    const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
    res.send(encodedBuffer);
  } catch (error) {
    logger.error("Media info error:", error);
    res.status(500).send("Error loading media info");
  }
});

// ============ VIDEO FRAME PLAYBACK ENDPOINTS ============

// Serve individual video frame (with WBMP conversion)
app.get("/wml/video-frame/:messageId/:frameNumber", async (req, res) => {
  try {
    const { messageId, frameNumber } = req.params;
    const format = req.query.format || 'wbmp'; // wbmp, jpg, png

    const framesDir = path.join(VIDEO_FRAMES_DIR, messageId);
    const framePath = path.join(framesDir, `frame_${String(frameNumber).padStart(4, '0')}.png`);

    if (!fs.existsSync(framePath)) {
      return res.status(404).send("Frame not found");
    }

    const frameBuffer = await fs.promises.readFile(framePath);

    if (format === 'wbmp') {
      // Convert to WBMP for Nokia compatibility
      const { data: pixels, info } = await sharp(frameBuffer)
        .greyscale()
        .resize(96, 96, { // Smaller for WAP devices
          kernel: sharp.kernel.lanczos3,
          fit: "contain",
          position: "center",
          background: { r: 255, g: 255, b: 255, alpha: 1 },
        })
        .linear(1.3, -30)
        .normalise({ lower: 1, upper: 99 })
        .sharpen({ sigma: 1.5, flat: 2, jagged: 3 })
        .threshold(128, { greyscale: true, grayscale: true })
        .raw()
        .toBuffer({ resolveWithObject: true });

      // Create WBMP header
      const width = info.width;
      const height = info.height;
      const header = Buffer.from([
        0x00, // Type 0
        0x00, // FixHeaderField
        width,
        height,
      ]);

      // Convert pixels to WBMP 1-bit format
      const rowBytes = Math.ceil(width / 8);
      const wbmpData = Buffer.alloc(rowBytes * height);

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const pixelIndex = y * width + x;
          const pixel = pixels[pixelIndex];
          const isBlack = pixel < 128;

          if (isBlack) {
            const byteIndex = y * rowBytes + Math.floor(x / 8);
            const bitIndex = 7 - (x % 8);
            wbmpData[byteIndex] |= 1 << bitIndex;
          }
        }
      }

      const wbmpBuffer = Buffer.concat([header, wbmpData]);

      res.setHeader("Content-Type", "image/vnd.wap.wbmp");
      res.setHeader("Cache-Control", "public, max-age=3600");
      res.send(wbmpBuffer);
    } else if (format === 'jpg') {
      const jpegBuffer = await sharp(frameBuffer)
        .resize(128, 128, { fit: "contain", background: { r: 255, g: 255, b: 255 } })
        .jpeg({ quality: 70 })
        .toBuffer();

      res.setHeader("Content-Type", "image/jpeg");
      res.setHeader("Cache-Control", "public, max-age=3600");
      res.send(jpegBuffer);
    } else {
      res.setHeader("Content-Type", "image/png");
      res.setHeader("Cache-Control", "public, max-age=3600");
      res.send(frameBuffer);
    }
  } catch (error) {
    logger.error("Video frame error:", error);
    res.status(500).send("Error loading frame");
  }
});

// Video playback WML page with frame-by-frame controls
app.get("/wml/view-video.wml", async (req, res) => {
  try {
    const messageId = req.query.mid || "";
    const jid = req.query.jid || "";
    const frameNum = parseInt(req.query.frame || "1", 10);
    const autoplay = req.query.autoplay === "1";
    const format = req.query.format || "wbmp"; // wbmp, jpg, png

    // Find message
    const messages = chatStore.get(jid) || [];
    const targetMessage = messages.find((m) => m.key.id === messageId);

    if (!targetMessage || !targetMessage.message?.videoMessage) {
      return sendWml(
        res,
        resultCard("Error", ["Video message not found"], "/wml/chats.wml")
      );
    }

    const contact = contactStore.get(jid);
    const chatName = contact?.name || contact?.notify || jidFriendly(jid);

    const esc = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // Check if frames exist, if not extract them
    const framesDir = path.join(VIDEO_FRAMES_DIR, messageId);
    let frameCount = 0;

    if (fs.existsSync(framesDir)) {
      const frames = fs.readdirSync(framesDir).filter(f => f.endsWith('.png'));
      frameCount = frames.length;
    }

    if (frameCount === 0) {
      // Need to extract frames
      try {
        const mediaData = await downloadMediaMessage(
          targetMessage,
          "buffer",
          {},
          {
            logger,
            reuploadRequest: sock.updateMediaMessage,
          }
        );

        const result = await extractVideoFrames(mediaData, messageId);
        frameCount = result.frameCount;
      } catch (error) {
        logger.error("Frame extraction error:", error);
        return sendWml(
          res,
          resultCard(
            "Error",
            ["Failed to extract video frames", error.message],
            `/wml/media-info.wml?mid=${encodeURIComponent(messageId)}&jid=${encodeURIComponent(jid)}`
          )
        );
      }
    }

    const currentFrame = Math.max(1, Math.min(frameNum, frameCount));
    const vid = targetMessage.message.videoMessage;
    const duration = vid.seconds || 0;

    // Navigation with format parameter
    const prevFrame = currentFrame > 1 ? currentFrame - 1 : frameCount;
    const nextFrame = currentFrame < frameCount ? currentFrame + 1 : 1;

    const prevLink = `/wml/view-video.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;frame=${prevFrame}&amp;format=${format}`;
    const nextLink = `/wml/view-video.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;frame=${nextFrame}&amp;format=${format}`;
    const autoplayLink = `/wml/view-video.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;frame=${nextFrame}&amp;autoplay=1&amp;format=${format}`;

    // WAP only supports WBMP inline - force WBMP for display
    const displayFormat = 'wbmp';

    const body = `<p><b>Video Playback</b></p>
<p>From: ${esc(chatName)}</p>
<p>Frame ${currentFrame}/${frameCount}</p>
<p>Duration: ${duration}s (1 FPS)</p>

<p align="center">
  <img src="/wml/video-frame/${encodeURIComponent(messageId)}/${currentFrame}?format=wbmp" alt="Frame ${currentFrame}"/>
</p>

<p>
  <a href="${prevLink}" accesskey="4">[4] Prev</a> |
  <a href="${nextLink}" accesskey="6">[6] Next</a>
</p>
<p>
  <a href="${autoplayLink}" accesskey="5">[5] ${autoplay ? 'Playing...' : 'Play'}</a>
</p>

<p><b>Download Frame:</b></p>
<p>
  <a href="/wml/video-frame/${encodeURIComponent(messageId)}/${currentFrame}?format=jpg" accesskey="7">[7] JPG</a>
  <a href="/wml/video-frame/${encodeURIComponent(messageId)}/${currentFrame}?format=png" accesskey="8">[8] PNG</a>
</p>

<p>
  <a href="/wml/media-info.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}" accesskey="0">[0] Back</a>
</p>

${autoplay ? `<onevent type="ontimer"><go href="${nextLink}&amp;autoplay=1"/></onevent><timer value="10"/>` : ''}

<do type="prev" label="Prev">
  <go href="${prevLink}"/>
</do>
<do type="accept" label="Next">
  <go href="${nextLink}"/>
</do>`;

    const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="no-cache"/></head>
<card id="video" title="Video">
${body}
</card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Pragma", "no-cache");

    const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
    res.send(encodedBuffer);
  } catch (error) {
    logger.error("Video playback error:", error);
    res.status(500).send("Error loading video");
  }
});

// Video format selection page
app.get("/wml/video-format.wml", async (req, res) => {
  try {
    const messageId = req.query.mid || "";
    const jid = req.query.jid || "";
    const frame = req.query.frame || "1";

    const esc = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    const body = `<p><b>Select Video Format</b></p>
<p>Choose format for frame display:</p>

<p><b>1. WBMP (B&amp;W)</b></p>
<p>Best for old Nokia devices<br/>
96x96 pixels, 1-bit<br/>
Smallest size, fastest</p>
<p><a href="/wml/view-video.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;frame=${frame}&amp;format=wbmp" accesskey="1">[1] Select WBMP</a></p>

<p><b>2. JPEG (Color)</b></p>
<p>Color support<br/>
128x128 pixels<br/>
Medium size</p>
<p><a href="/wml/view-video.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;frame=${frame}&amp;format=jpg" accesskey="2">[2] Select JPEG</a></p>

<p><b>3. PNG (Color)</b></p>
<p>High quality color<br/>
128x128 pixels<br/>
Larger size</p>
<p><a href="/wml/view-video.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;frame=${frame}&amp;format=png" accesskey="3">[3] Select PNG</a></p>

<p><a href="/wml/view-video.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;frame=${frame}" accesskey="0">[0] Back</a></p>

<do type="accept" label="Back">
  <go href="/wml/view-video.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;frame=${frame}"/>
</do>`;

    const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="no-cache"/></head>
<card id="format" title="Format">
${body}
</card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Pragma", "no-cache");

    const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
    res.send(encodedBuffer);
  } catch (error) {
    logger.error("Video format selection error:", error);
    res.status(500).send("Error loading format selection");
  }
});

// Image format selection page
app.get("/wml/image-format.wml", async (req, res) => {
  try {
    const messageId = req.query.mid || "";
    const jid = req.query.jid || "";

    const esc = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    const body = `<p><b>Select Image Format</b></p>
<p>Choose format for image display:</p>

<p><b>1. WBMP (B&amp;W)</b></p>
<p>Best for old Nokia devices<br/>
Black &amp; white<br/>
Smallest size, fastest</p>
<p><a href="/wml/view-image.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;format=wbmp" accesskey="1">[1] Select WBMP</a></p>

<p><b>2. JPEG (Color)</b></p>
<p>Color support<br/>
Medium quality<br/>
Medium size</p>
<p><a href="/wml/view-image.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;format=jpg" accesskey="2">[2] Select JPEG</a></p>

<p><b>3. PNG (Color)</b></p>
<p>High quality color<br/>
Best quality<br/>
Larger size</p>
<p><a href="/wml/view-image.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}&amp;format=png" accesskey="3">[3] Select PNG</a></p>

<p><a href="/wml/media-info.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}" accesskey="0">[0] Back</a></p>

<do type="accept" label="Back">
  <go href="/wml/media-info.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}"/>
</do>`;

    const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="no-cache"/></head>
<card id="format" title="Format">
${body}
</card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Pragma", "no-cache");

    const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
    res.send(encodedBuffer);
  } catch (error) {
    logger.error("Image format selection error:", error);
    res.status(500).send("Error loading format selection");
  }
});

// Image viewer with format selection
app.get("/wml/view-image.wml", async (req, res) => {
  try {
    const messageId = req.query.mid || "";
    const jid = req.query.jid || "";
    const format = req.query.format || "wbmp"; // wbmp, jpg, png

    // Find message
    const messages = chatStore.get(jid) || [];
    const targetMessage = messages.find((m) => m.key.id === messageId);

    if (!targetMessage || !targetMessage.message?.imageMessage) {
      return sendWml(
        res,
        resultCard("Error", ["Image message not found"], "/wml/chats.wml")
      );
    }

    const contact = contactStore.get(jid);
    const chatName = contact?.name || contact?.notify || jidFriendly(jid);
    const img = targetMessage.message.imageMessage;
    const caption = img.caption || "";

    const esc = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    // WAP only supports WBMP inline - always display WBMP
    const body = `<p><b>Image Viewer</b></p>
<p>From: ${esc(chatName)}</p>
${caption ? `<p>Caption: ${esc(caption)}</p>` : ''}

<p align="center">
  <img src="/wml/media/${encodeURIComponent(messageId)}.wbmp?wbmp=1" alt="Image"/>
</p>

<p><b>Download Other Formats:</b></p>
<p>
  <a href="/wml/media/${encodeURIComponent(messageId)}.jpg" accesskey="7">[7] Download JPG</a><br/>
  <a href="/wml/media/${encodeURIComponent(messageId)}.png" accesskey="8">[8] Download PNG</a>
</p>

<p>
  <a href="/wml/media-info.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}" accesskey="0">[0] Back</a>
</p>

<do type="accept" label="Back">
  <go href="/wml/media-info.wml?mid=${encodeURIComponent(messageId)}&amp;jid=${encodeURIComponent(jid)}"/>
</do>`;

    const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="no-cache"/></head>
<card id="image" title="Image">
${body}
</card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Pragma", "no-cache");

    const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
    res.send(encodedBuffer);
  } catch (error) {
    logger.error("Image viewer error:", error);
    res.status(500).send("Error loading image");
  }
});

// Nuova route per visualizzare WBMP in una pagina WAP dedicata
app.get("/wml/view-wbmp.wml", async (req, res) => {
  try {
    const messageId = req.query.mid || "";
    const jid = req.query.jid || "";

    // Find message in the specific chat
    const messages = chatStore.get(jid) || [];
    const targetMessage = messages.find((m) => m.key.id === messageId);

    const contact = contactStore.get(jid);
    const chatName = contact?.name || contact?.notify || jidFriendly(jid);

    // Simple escaping
    const esc = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

    let body = "";
    let title = "WBMP Image";

    if (targetMessage) {
      const isImage = targetMessage.message?.imageMessage;
      const isSticker = targetMessage.message?.stickerMessage;

      if (isImage || isSticker) {
        const mediaObj = isImage
          ? targetMessage.message.imageMessage
          : targetMessage.message.stickerMessage;
        const size = Math.round((mediaObj.fileLength || 0) / 1024);
        const caption = mediaObj.caption
          ? `<p><b>Caption:</b> ${esc(mediaObj.caption)}</p>`
          : "";

        title = isImage ? "Image (WBMP)" : "Sticker (WBMP)";

        body = `<p><b>${isImage ? "Image" : "Sticker"}</b></p>
<p>From: ${esc(chatName)}</p>
<p>Size: ${size}KB</p>
${caption}
<p>
<img src="/wml/media/${encodeURIComponent(messageId)}.wbmp" alt="WBMP Image"/>
</p>
<p>
<a href="/wml/media-info.wml?mid=${encodeURIComponent(
          messageId
        )}&amp;jid=${encodeURIComponent(
          jid
        )}" accesskey="0">[0] Back to Media Info</a>
</p>
<p>
<a href="/wml/chat.wml?jid=${encodeURIComponent(
          jid
        )}" accesskey="1">[1] Back to Chat</a> |
<a href="/wml/chats.wml" accesskey="9">[9] All Chats</a>
</p>`;
      } else {
        body = `<p><b>Not an Image</b></p>
<p>This message does not contain an image or sticker</p>
<p>
<a href="/wml/media-info.wml?mid=${encodeURIComponent(
          messageId
        )}&amp;jid=${encodeURIComponent(
          jid
        )}" accesskey="0">[0] Back to Media Info</a>
</p>`;
      }
    } else {
      body = `<p><b>Media Not Found</b></p>
<p>Message may have been deleted</p>
<p>
<a href="/wml/chats.wml" accesskey="9">[9] All Chats</a>
</p>`;
    }

    const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="max-age=0"/></head>
<card id="wbmp" title="${title}">
${body}
<do type="accept" label="Back">
<go href="/wml/media-info.wml?mid=${encodeURIComponent(
      messageId
    )}&amp;jid=${encodeURIComponent(jid)}"/>
</do>
<do type="options" label="Chat">
<go href="/wml/chat.wml?jid=${encodeURIComponent(jid)}"/>
</do>
</card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Pragma", "no-cache");

    const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
    res.send(encodedBuffer);
  } catch (error) {
    logger.error("WBMP view error:", error);
    res.status(500).send("Error loading WBMP view");
  }
});

/*
app.get('/wml/chat.wml', async (req, res) => {
  const userAgent = req.headers['user-agent'] || ''
  const isOldNokia = /Nokia|Series40|MAUI|UP\.Browser/i.test(userAgent)
  
  const raw = req.query.jid || ''
  const jid = formatJid(raw)
  const offset = Math.max(0, parseInt(req.query.offset || '0'))
  const search = (req.query.search || '').trim().toLowerCase()
  
  // Very small limits for Nokia 7210
  const limit = isOldNokia ? 3 : 10
  
  // Load chat history if missing
  if ((!chatStore.get(jid) || chatStore.get(jid).length === 0) && sock) {
    try {
      await loadChatHistory(jid, limit * 3)
    } catch (e) {
      logger.warn(`Failed to load chat history for ${jid}: ${e.message}`)
    }
  }
  
  let allMessages = (chatStore.get(jid) || []).slice()
  
  // Sort by timestamp - MOST RECENT FIRST
  allMessages.sort((a, b) => {
    const tsA = Number(a.messageTimestamp) || 0
    const tsB = Number(b.messageTimestamp) || 0
    return tsB - tsA // Most recent first
  })
  
  // Apply search filter if present
  if (search) {
    allMessages = allMessages.filter(m => (messageText(m) || '').toLowerCase().includes(search))
  }
  
  const total = allMessages.length
  const items = allMessages.slice(offset, offset + limit)
  
  const contact = contactStore.get(jid)
  const chatName = contact?.name || contact?.notify || contact?.verifiedName || jidFriendly(jid)
  const number = jidFriendly(jid)
  const isGroup = jid.endsWith('@g.us')
  
  // Simple escaping for Nokia 7210
  const esc = text => (text || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  
  // Simple truncate
  const truncate = (text, maxLength) => {
    if (!text) return ''
    if (text.length <= maxLength) return text
    return text.substring(0, maxLength - 3) + '...'
  }
  
  // Simple timestamp for Nokia
  const formatTime = (timestamp) => {
    const date = new Date(Number(timestamp) * 1000)
    if (isNaN(date.getTime())) return ''
    
    const day = date.getDate().toString().padStart(2, '0')
    const month = (date.getMonth() + 1).toString().padStart(2, '0')
    const hours = date.getHours().toString().padStart(2, '0')
    const mins = date.getMinutes().toString().padStart(2, '0')
    
    return `${day}/${month} ${hours}:${mins}`
  }
  
  let messageList = ''
  
  if (items.length === 0) {
    messageList = '<p>No messages</p>'
  } else {
    messageList = items.map((m, idx) => {
      const who = m.key.fromMe ? 'Me' : (chatName.length > 10 ? chatName.substring(0, 10) : chatName)
      const time = formatTime(m.messageTimestamp)
      const msgNumber = idx + 1
      const mid = m.key.id
      
      // Handle different message types for Nokia
      let text = ''
      let mediaLink = ''
      
      if (m.message) {
        if (m.message.imageMessage) {
          const img = m.message.imageMessage
          const size = Math.round((img.fileLength || 0) / 1024)
          text = `[IMG ${size}KB]`
          if (img.caption) text += ` ${truncate(img.caption, 30)}`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View IMG]</a>`
          
        } else if (m.message.videoMessage) {
          const vid = m.message.videoMessage
          const size = Math.round((vid.fileLength || 0) / 1024)
          text = `[VID ${size}KB]`
          if (vid.caption) text += ` ${truncate(vid.caption, 30)}`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View VID]</a>`
          
        } else if (m.message.audioMessage) {
          const aud = m.message.audioMessage
          const size = Math.round((aud.fileLength || 0) / 1024)
          const duration = aud.seconds || 0
          text = `[AUD ${size}KB ${duration}s]`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View AUD]</a>`
          
        } else if (m.message.documentMessage) {
          const doc = m.message.documentMessage
          const size = Math.round((doc.fileLength || 0) / 1024)
          const filename = doc.fileName || 'file'
          text = `[DOC ${size}KB] ${truncate(filename, 20)}`
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View DOC]</a>`
          
        } else if (m.message.stickerMessage) {
          text = '[STICKER]'
          mediaLink = `<br/><a href="/wml/media-info.wml?mid=${encodeURIComponent(mid)}&amp;jid=${encodeURIComponent(jid)}">[View STK]</a>`
          
        } else {
          text = truncate(messageText(m) || '', 50)
        }
      } else {
        text = truncate(messageText(m) || '', 50)
      }
      
      return `<p>${msgNumber}. ${esc(who)} (${time})<br/>${esc(text)}${mediaLink}</p>`
    }).join('')
  }
  
  // Simple navigation for Nokia
  const olderOffset = offset + limit
  const olderLink = olderOffset < total ? 
    `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;offset=${olderOffset}&amp;search=${encodeURIComponent(search)}" accesskey="2">2-Older</a></p>` : ''
  
  const newerOffset = Math.max(0, offset - limit)
  const newerLink = offset > 0 ? 
    `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;offset=${newerOffset}&amp;search=${encodeURIComponent(search)}" accesskey="3">3-Newer</a></p>` : ''
  
  // Simple search for Nokia
  const searchBox = search ? 
    `<p>Search: ${esc(search)}</p><p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}">Clear</a></p>` : 
    `<p><a href="/wml/chat.wml?jid=${encodeURIComponent(jid)}&amp;search=prompt">Search</a></p>`
  
  const body = `<p>${esc(chatName.length > 15 ? chatName.substring(0, 15) : chatName)}</p>
<p>Msgs ${offset + 1}-${Math.min(offset + limit, total)}/${total}</p>
${searchBox}
${messageList}
${newerLink}
${olderLink}
<p><a href="/wml/send.text.wml?to=${encodeURIComponent(jid)}" accesskey="1">1-Send</a></p>
<p><a href="/wml/chats.wml" accesskey="0">0-Back</a></p>`
  
  // Nokia 7210 compatible WML 1.1
  const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="max-age=0"/></head>
<card id="chat" title="Chat">
${body}
</card>
</wml>`
  
  // Nokia 7210 headers
  res.setHeader('Content-Type', 'text/vnd.wap.wml; charset=iso-8859-1')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Pragma', 'no-cache')
  
  const encodedBuffer = iconv.encode(wmlOutput, 'iso-8859-1')
  res.send(encodedBuffer)
})*/
// Enhanced Message Actions page
app.get("/wml/msg.wml", (req, res) => {
  const mid = String(req.query.mid || "");
  const jid = formatJid(req.query.jid || "");

  // Find message in the specific chat (using our new system)
  const messages = chatStore.get(jid) || [];
  const msg = messages.find((m) => m.key.id === mid);

  if (!msg) {
    sendWml(
      res,
      resultCard(
        "Message",
        ["Message not found"],
        `/wml/chat.wml?jid=${encodeURIComponent(jid)}&limit=15`
      )
    );
    return;
  }

  const text = truncate(messageText(msg), 150);
  const ts = new Date(Number(msg.messageTimestamp) * 1000).toLocaleString();

  // Enhanced media detection
  let mediaInfo = "";
  let mediaActions = "";
  let hasMedia = false;
  let transcriptionInfo = "";
  let transcriptionActions = "";

  if (msg.message) {
    if (msg.message.imageMessage) {
      const img = msg.message.imageMessage;
      const size = Math.round((img.fileLength || 0) / 1024);
      mediaInfo = `<p><small>Type: Image (${size}KB)</small></p>`;
      mediaActions = `<a href="/wml/media-info.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(
        jid
      )}" accesskey="4">[4] View Image</a><br/>
      <a href="/wml/media/${encodeURIComponent(
        mid
      )}.jpg" accesskey="5">[5] Download JPG</a><br/>`;
      hasMedia = true;
    } else if (msg.message.videoMessage) {
      const vid = msg.message.videoMessage;
      const size = Math.round((vid.fileLength || 0) / 1024);
      const duration = vid.seconds || 0;
      mediaInfo = `<p><small>Type: Video (${size}KB, ${duration}s)</small></p>`;
      mediaActions = `<a href="/wml/media-info.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(
        jid
      )}" accesskey="4">[4] View Video</a><br/>
      <a href="/wml/media/${encodeURIComponent(
        mid
      )}.mp4" accesskey="5">[5] Download MP4</a><br/>`;
      hasMedia = true;
    } else if (msg.message.audioMessage) {
      const aud = msg.message.audioMessage;
      const size = Math.round((aud.fileLength || 0) / 1024);
      const duration = aud.seconds || 0;

      mediaInfo = `<p><small>Type: Audio (${size}KB, ${duration}s)</small></p>`;

      hasMedia = true;

      // Gestione trascrizione
      if (msg.transcription) {
        if (msg.transcription === "[Trascrizione fallita]") {
          transcriptionInfo = `<p><small>Trascrizione: Fallita</small></p>`;
        } else if (
          msg.transcription === "[Audio troppo lungo per la trascrizione]"
        ) {
          transcriptionInfo = `<p><small>Trascrizione: Audio troppo lungo</small></p>`;
        } else {
          transcriptionInfo = `<p><small>Trascrizione: Disponibile</small></p>`;
          transcriptionActions = `<a href="/wml/audio-transcription.wml?mid=${encodeURIComponent(
            mid
          )}&amp;jid=${encodeURIComponent(
            jid
          )}" accesskey="6">[6] View Transcription</a><br/>`;
        }
      } else {
        transcriptionInfo = `<p><small>Trascrizione: In elaborazione...</small></p>`;
      }
    } else if (msg.message.documentMessage) {
      const doc = msg.message.documentMessage;
      const size = Math.round((doc.fileLength || 0) / 1024);
      const filename = doc.fileName || "document";
      mediaInfo = `<p><small>Type: Document (${size}KB)</small></p>
      <p><small>File: ${esc(filename)}</small></p>`;
      const ext = filename.split(".").pop() || "bin";
      mediaActions = `<a href="/wml/media-info.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(
        jid
      )}" accesskey="4">[4] View Document</a><br/>
      <a href="/wml/media/${encodeURIComponent(
        mid
      )}.${ext}" accesskey="5">[5] Download File</a><br/>`;
      hasMedia = true;
    } else if (msg.message.stickerMessage) {
      const sticker = msg.message.stickerMessage;
      const size = Math.round((sticker.fileLength || 0) / 1024);
      mediaInfo = `<p><small>Type: Sticker (${size}KB)</small></p>`;
      mediaActions = `<a href="/wml/media-info.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(
        jid
      )}" accesskey="4">[4] View Sticker</a><br/>
      <a href="/wml/media/${encodeURIComponent(
        mid
      )}.webp" accesskey="5">[5] Download Sticker</a><br/>`;
      hasMedia = true;
    }
  }

  const body = `
    <p><b>Message Details</b></p>
    <p>${esc(text)}</p>
    <p><small>Time: ${ts}</small></p>
    <p><small>From: ${msg.key.fromMe ? "Me" : "Them"}</small></p>
    ${mediaInfo}
    ${transcriptionInfo}
    
    <p><b>Actions:</b></p>
    <p>
      <a href="/wml/msg.reply.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(jid)}" accesskey="1">[1] Reply</a><br/>
      <a href="/wml/msg.react.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(jid)}" accesskey="2">[2] React</a><br/>
      <a href="/wml/msg.forward.wml?mid=${encodeURIComponent(
        mid
      )}" accesskey="3">[3] Forward</a><br/>
      ${mediaActions}
      ${transcriptionActions}
      <a href="/wml/msg.delete.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(jid)}" accesskey="7">[7] Delete</a><br/>
      <a href="/wml/msg.read.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(
    jid
  )}" accesskey="8">[8] Mark Read</a><br/>
    </p>
    
    <p><a href="/wml/chat.wml?jid=${encodeURIComponent(
      jid
    )}&amp;limit=15" accesskey="0">[0] Back to Chat</a></p>
    
    <do type="accept" label="Reply">
      <go href="/wml/msg.reply.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(jid)}"/>
    </do>
    ${
      hasMedia || transcriptionActions
        ? `<do type="options" label="Media">
      <go href="/wml/media-info.wml?mid=${encodeURIComponent(
        mid
      )}&amp;jid=${encodeURIComponent(jid)}"/>
    </do>`
        : ""
    }
  `;

  sendWml(res, card("msg", "Message", body));
});
app.get("/wml/send-menu.wml", (req, res) => {
  const to = esc(req.query.to || "");
  const search = (req.query.search || "").toLowerCase();
  const page = parseInt(req.query.page || "1", 10);
  const pageSize = 5;
  const contact = to
    ? contactStore.get?.(formatJid(to)) || contactStore[formatJid(to)]
    : null;
  const contactName = contact?.name || contact?.notify || jidFriendly(to) || "";

  // Recupera contatti in array
  let contactsArray = [];
  if (contactStore instanceof Map)
    contactsArray = Array.from(contactStore.values());
  else contactsArray = Object.values(contactStore);

  // Filtra contatti per ricerca
  const filteredContacts = contactsArray.filter(
    (c) => !search || (c.name || c.notify || "").toLowerCase().includes(search)
  );

  const totalPages = Math.ceil(filteredContacts.length / pageSize);
  const currentPage = Math.min(Math.max(page, 1), totalPages || 1);
  const start = (currentPage - 1) * pageSize;
  const pageContacts = filteredContacts.slice(start, start + pageSize);

  // Debug: verifica i contatti
  console.log("Total contacts:", contactsArray.length);
  console.log("Filtered contacts:", filteredContacts.length);
  console.log("Page contacts:", pageContacts.length);
  console.log("Sample contact:", pageContacts[0]);

  const selectOptions = pageContacts
    .map((c) => {
      const displayName = c.name || c.notify || jidFriendly(c.jid) || c.jid;
      const jidValue = c.jid || c.id || "";
      console.log("Contact option:", displayName, "->", jidValue);
      return `<option value="${esc(jidValue)}"${
        jidValue === to ? ' selected="selected"' : ""
      }>${esc(displayName)}</option>`;
    })
    .join("");

  // Costruisci paginazione "max 5 numeri" con First / Last / Back / Next
  let pagination = "";
  if (totalPages > 1) {
    pagination += "<p>";
    // First
    if (currentPage > 1) {
      pagination += `<a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}&amp;search=${encodeURIComponent(search)}&amp;page=1">First</a> `;
      pagination += `<a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}&amp;search=${encodeURIComponent(search)}&amp;page=${
        currentPage - 1
      }">Back</a> `;
    }
    // Calcola range di 5 numeri centrati sulla pagina corrente
    let startPage = Math.max(1, currentPage - 2);
    let endPage = Math.min(totalPages, startPage + 4);
    if (endPage - startPage < 4) startPage = Math.max(1, endPage - 4);
    for (let i = startPage; i <= endPage; i++) {
      if (i === currentPage) pagination += `<b>${i}</b> `;
      else
        pagination += `<a href="/wml/send-menu.wml?to=${encodeURIComponent(
          to
        )}&amp;search=${encodeURIComponent(search)}&amp;page=${i}">${i}</a> `;
    }
    // Next / Last
    if (currentPage < totalPages) {
      pagination += `<a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}&amp;search=${encodeURIComponent(search)}&amp;page=${
        currentPage + 1
      }">Next</a> `;
      pagination += `<a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}&amp;search=${encodeURIComponent(
        search
      )}&amp;page=${totalPages}">Last</a>`;
    }
    pagination += "</p>";
  }

  const body = `
    <p><b>Send Message</b></p>
    ${to ? `<p>To: <b>${esc(contactName)}</b></p>` : ""}
    <p>Search contacts:</p>
    <input name="search" value="${esc(
      req.query.search || ""
    )}" size="15" iname="search"/>
    <do type="accept" label="Filter">
      <go href="/wml/send-menu.wml">
        <postfield name="to" value="${esc(to)}"/>
        <postfield name="search" value="$(search)"/>
      </go>
    </do>
   
    <p>Select contact (page ${currentPage} of ${totalPages || 1}):</p>
    <select name="target" title="Contact">
      <option value="">-- Enter number manually --</option>
      ${selectOptions}
    </select>
    <do type="accept" label="Set Contact">
      <go href="/wml/send-menu.wml">
        <postfield name="to" value="$target"/>
        <postfield name="search" value="${esc(search)}"/>
        <postfield name="page" value="${currentPage}"/>
      </go>
    </do>
    ${pagination}
    
    ${
      to
        ? `<p><a href="/wml/send-menu.wml?search=${encodeURIComponent(
            search
          )}&amp;page=${currentPage}">[Clear Selection]</a></p>`
        : ""
    }
    <p>Or enter number manually:</p>
    <input name="target_manual" value="${esc(to)}" size="15" title="Manual"/>
    <do type="accept" label="Set Manual">
      <go href="/wml/send-menu.wml">
        <postfield name="to" value="$target_manual"/>
        <postfield name="search" value="${esc(search)}"/>
        <postfield name="page" value="${currentPage}"/>
      </go>
    </do>
    <p><b>Message Types:</b></p>
    <select name="msgtype" title="Type" iname="msgtype" ivalue="/wml/send.text.wml">
      <option value="/wml/send.text.wml">Text Message</option>
      <option value="/wml/send.tts.wml">Voice (Text-to-Speech)</option>
      <option value="/wml/send.image.wml">Image (URL)</option>
      <option value="/wml/send.video.wml">Video (URL)</option>
      <option value="/wml/send.audio.wml">Audio (URL)</option>
      <option value="/wml/send.document.wml">Document</option>
      <option value="/wml/send.sticker.wml">Sticker</option>
      <option value="/wml/send.location.wml">Location</option>
      <option value="/wml/send.contact.wml">Contact</option>
      <option value="/wml/send.poll.wml">Poll</option>
    </select>
    
<do type="accept" label="Send Message">
  <go href="/wml/send-dispatch.wml" method="post">
    <postfield name="msgtype" value="$(msgtype)"/>
    <postfield name="target" value="$(target)"/>
    <postfield name="target_manual" value="$(target_manual)"/>
  </go>
</do>


    
    <do type="accept" label="Reset All ">
      <go href="/wml/send-menu.wml" method="post">
       
      </go>
    </do>
    

    ${navigationBar()}
    <do type="options" label="Recent">
      <go href="/wml/contacts.wml"/>
    </do>
  `;

  sendWml(res, card("send-menu", "Send Menu", body));
});

// Quick Send - Streamlined for known contacts (FAST, non-blocking)
// Use this when contact is already known from chat/contact info
app.get("/wml/send-quick.wml", (req, res) => {
  const to = req.query.to || "";

  if (!to) {
    // No contact specified, redirect to full menu
    return res.redirect(302, "/wml/send-menu.wml");
  }

  // Fast lookup - non-blocking
  const jid = formatJid(to);
  const contact = contactStore.get?.(jid) || contactStore[jid];
  const contactName = contact?.name || contact?.notify || jidFriendly(to);

  // Minimal, fast HTML - optimized for Nokia WAP
  const body = `
    <p><b>Quick Send</b></p>
    <p>To: <b>${esc(contactName)}</b></p>
    <p>Number: ${esc(jidFriendly(to))}</p>

    <p>Select type:</p>
    <select name="msgtype" title="Type">
      <option value="/wml/send.text.wml">Text</option>
      <option value="/wml/send.tts.wml">Voice (TTS)</option>
      <option value="/wml/send.image.wml">Image</option>
      <option value="/wml/send.video.wml">Video</option>
      <option value="/wml/send.audio.wml">Audio</option>
      <option value="/wml/send.document.wml">Document</option>
      <option value="/wml/send.sticker.wml">Sticker</option>
      <option value="/wml/send.location.wml">Location</option>
      <option value="/wml/send.contact.wml">Contact</option>
      <option value="/wml/send.poll.wml">Poll</option>
    </select>

    <do type="accept" label="Continue">
      <go href="/wml/send-dispatch.wml" method="get">
        <postfield name="msgtype" value="$(msgtype)"/>
        <postfield name="target" value="${esc(to)}"/>
      </go>
    </do>

    <p>
      <a href="/wml/send-menu.wml" accesskey="0">[0] Full Menu</a> |
      <a href="/wml/chats.wml" accesskey="9">[9] Chats</a>
    </p>
  `;

  // Fast response - minimal processing
  sendWml(res, card("send-quick", "Quick Send", body));
});

// Route di dispatch per gestire la selezione del destinatario
// Uses simple HTTP redirect for Nokia WAP compatibility
app.post("/wml/send-dispatch.wml", (req, res) => {
  const msgtype = req.body.msgtype || "/wml/send.text.wml";
  const target = req.body.target || "";
  const targetManual = req.body.target_manual || "";

  // Debug: log dei parametri ricevuti
  console.log(
    "Dispatch POST - msgtype:",
    msgtype,
    "target:",
    target,
    "target_manual:",
    targetManual
  );

  // Determina il destinatario: priorità al contatto selezionato, altrimenti manuale
  const finalTarget = target || targetManual;

  if (!finalTarget) {
    // Nessun destinatario specificato, torna al menu con errore
    const body = `
      <p><b>Error</b></p>
      <p>Please select a contact or enter a number manually.</p>
      <p>Debug info: target="${esc(target)}", manual="${esc(targetManual)}"</p>
      <p><a href="/wml/send-menu.wml">Back to Send Menu</a></p>
      ${navigationBar()}
    `;
    return sendWml(res, card("error", "Error", body));
  }

  // Simple HTTP redirect - works on all Nokia WAP devices
  const redirectUrl = `${msgtype}?to=${encodeURIComponent(finalTarget)}`;
  console.log("Redirecting to:", redirectUrl);

  // Use standard HTTP 302 redirect for Nokia compatibility
  res.redirect(302, redirectUrl);
});

// Versione GET per fallback - Nokia WAP compatible
app.get("/wml/send-dispatch.wml", (req, res) => {
  const msgtype = req.query.msgtype || "/wml/send.text.wml";
  const target = req.query.target || "";
  const targetManual = req.query.target_manual || "";

  const finalTarget = target || targetManual;

  if (!finalTarget) {
    const body = `
      <p><b>Error</b></p>
      <p>Please select a contact or enter a number manually.</p>
      <p><a href="/wml/send-menu.wml">Back to Send Menu</a></p>
      ${navigationBar()}
    `;
    return sendWml(res, card("error", "Error", body));
  }

  const redirectUrl = `${msgtype}?to=${encodeURIComponent(finalTarget)}`;

  // Use standard HTTP 302 redirect for Nokia compatibility
  res.redirect(302, redirectUrl);
});

// Enhanced Send Text with templates
app.get("/wml/send.text.wml", (req, res) => {
  const to = esc(req.query.to || "");
  const template = req.query.template || "";

  const templates = [
    "Hello! How are you?",
    "Thanks for your message.",
    "I will call you back later.",
    "Please send me the details.",
    "Meeting confirmed for today.",
  ];

  const body = `
    <p><b>Send Text Message</b></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>
    
    <p>Message:</p>
    <input name="message" title="Your message" value="${esc(
      template
    )}" size="30" maxlength="1000"/>
    
    ${
      template
        ? ""
        : `
    <p><b>Templates:</b></p>
    <select name="tmpl" title="Quick Templates">
      ${templates
        .map(
          (t, i) =>
            `<option value="${esc(t)}">${i + 1}. ${esc(
              truncate(t, 20)
            )}</option>`
        )
        .join("")}
    </select>
    <do type="options" label="Use">
      <refresh>
        <setvar name="message" value="$(tmpl)"/>
      </refresh>
    </do>
    `
    }
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/send.text">
        <postfield name="to" value="$(to)"/>
        <postfield name="message" value="$(message)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a> 
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-text", "Send Text", body));
});

// Endpoint per inviare immagini
app.get("/wml/send.image.wml", (req, res) => {
  const to = esc(req.query.to || "");

  const body = `
    <p><b>Send Image</b></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>
    
    <p>Image URL:</p>
    <input name="imageUrl" title="Image URL" value="https://" size="30" maxlength="500"/>
    
    <p>Caption (optional):</p>
    <input name="caption" title="Image caption" value="" size="30" maxlength="1000"/>
    
    <p><small>Supported formats: JPG, PNG, GIF</small></p>
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/send.image">
        <postfield name="to" value="$(to)"/>
        <postfield name="imageUrl" value="$(imageUrl)"/>
        <postfield name="caption" value="$(caption)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a> 
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-image", "Send Image", body));
});

// Endpoint per inviare video
app.get("/wml/send.video.wml", (req, res) => {
  const to = esc(req.query.to || "");

  const body = `
    <p><b>Send Video</b></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>
    
    <p>Video URL:</p>
    <input name="videoUrl" title="Video URL" value="https://" size="30" maxlength="500"/>
    
    <p>Caption (optional):</p>
    <input name="caption" title="Video caption" value="" size="30" maxlength="1000"/>
    
    <p><small>Supported formats: MP4, 3GP, AVI</small></p>
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/send.video">
        <postfield name="to" value="$(to)"/>
        <postfield name="videoUrl" value="$(videoUrl)"/>
        <postfield name="caption" value="$(caption)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a> 
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-video", "Send Video", body));
});

// Endpoint per inviare audio
// Text-to-Speech form
app.get("/wml/send.tts.wml", (req, res) => {
  const to = esc(req.query.to || "");

  const ttsStatus = ttsEnabled ? '✓ Local TTS Ready' : '⚠ espeak not installed';

  const body = `
    <p><b>Send Voice Message (TTS)</b></p>
    <p><small>${ttsStatus} (2 languages, offline)</small></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>

    <p>Your message:</p>
    <input name="text" title="Message" value="" size="30" maxlength="500"/>

    <p>Language:</p>
    <select name="language" title="Language">
      <option value="en">English</option>
      <option value="it">Italian (Italiano)</option>
    </select>

    <p>Voice Message (PTT):</p>
    <select name="ptt" title="Voice">
      <option value="true">Yes (Voice Note)</option>
      <option value="false">No (Audio File)</option>
    </select>

    <p><small>Max 500 characters<br/>Local espeak TTS (offline)</small></p>

    <do type="accept" label="Send">
      <go method="post" href="/wml/send.tts">
        <postfield name="to" value="$(to)"/>
        <postfield name="text" value="$(text)"/>
        <postfield name="language" value="$(language)"/>
        <postfield name="ptt" value="$(ptt)"/>
      </go>
    </do>

    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a>
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-tts", "Send Voice (TTS)", body));
});

app.get("/wml/send.audio.wml", (req, res) => {
  const to = esc(req.query.to || "");

  const body = `
    <p><b>Send Audio</b></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>
    
    <p>Audio URL:</p>
    <input name="audioUrl" title="Audio URL" value="https://" size="30" maxlength="500"/>
    
    <p>Audio Type:</p>
    <select name="audioType" title="Audio Type">
      <option value="audio/mp3">MP3</option>
      <option value="audio/mp4">MP4</option>
      <option value="audio/ogg">OGG</option>
      <option value="audio/wav">WAV</option>
    </select>
    
    <p>Voice Message:</p>
    <select name="ptt" title="Voice Message">
      <option value="false">No</option>
      <option value="true">Yes</option>
    </select>
    
    <p><small>Max file size: 16MB</small></p>
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/send.audio">
        <postfield name="to" value="$(to)"/>
        <postfield name="audioUrl" value="$(audioUrl)"/>
        <postfield name="audioType" value="$(audioType)"/>
        <postfield name="ptt" value="$(ptt)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a> 
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-audio", "Send Audio", body));
});

// Endpoint per inviare documenti
app.get("/wml/send.document.wml", (req, res) => {
  const to = esc(req.query.to || "");

  const body = `
    <p><b>Send Document</b></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>
    
    <p>Document URL:</p>
    <input name="documentUrl" title="Document URL" value="https://" size="30" maxlength="500"/>
    
    <p>File Name:</p>
    <input name="fileName" title="File name" value="document.pdf" size="20" maxlength="100"/>
    
    <p>MIME Type:</p>
    <select name="mimeType" title="MIME Type">
      <option value="application/pdf">PDF</option>
      <option value="application/msword">Word</option>
      <option value="application/vnd.ms-excel">Excel</option>
      <option value="application/zip">ZIP</option>
      <option value="text/plain">Text</option>
      <option value="application/octet-stream">Other</option>
    </select>
    
    <p><small>Max file size: 100MB</small></p>
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/send.document">
        <postfield name="to" value="$(to)"/>
        <postfield name="documentUrl" value="$(documentUrl)"/>
        <postfield name="fileName" value="$(fileName)"/>
        <postfield name="mimeType" value="$(mimeType)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a> 
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-document", "Send Document", body));
});

// Endpoint per inviare sticker
app.get("/wml/send.sticker.wml", (req, res) => {
  const to = esc(req.query.to || "");

  const body = `
    <p><b>Send Sticker</b></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>
    
    <p>Sticker URL:</p>
    <input name="stickerUrl" title="Sticker URL" value="https://" size="30" maxlength="500"/>
    
    <p><small>Supported formats: WEBP, PNG</small></p>
    <p><small>Max file size: 1MB</small></p>
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/send.sticker">
        <postfield name="to" value="$(to)"/>
        <postfield name="stickerUrl" value="$(stickerUrl)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a> 
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-sticker", "Send Sticker", body));
});

// Endpoint per inviare posizione
app.get("/wml/send.location.wml", (req, res) => {
  const to = esc(req.query.to || "");

  const body = `
    <p><b>Send Location</b></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>
    
    <p>Latitude:</p>
    <input name="latitude" title="Latitude" value="41.9028" size="15" maxlength="20"/>
    
    <p>Longitude:</p>
    <input name="longitude" title="Longitude" value="12.4964" size="15" maxlength="20"/>
    
    <p>Location Name (optional):</p>
    <input name="name" title="Location name" value="" size="20" maxlength="100"/>
    
    <p><small>Example: Rome, Italy</small></p>
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/send.location">
        <postfield name="to" value="$(to)"/>
        <postfield name="latitude" value="$(latitude)"/>
        <postfield name="longitude" value="$(longitude)"/>
        <postfield name="name" value="$(name)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a> 
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-location", "Send Location", body));
});

// Endpoint per inviare contatti
app.get("/wml/send.contact.wml", (req, res) => {
  const to = esc(req.query.to || "");

  const body = `
    <p><b>Send Contact</b></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>
    
    <p>Contact Name:</p>
    <input name="contactName" title="Contact name" value="" size="20" maxlength="50"/>
    
    <p>Phone Number:</p>
    <input name="phoneNumber" title="Phone number" value="" size="15" maxlength="20"/>
    
    <p>Organization (optional):</p>
    <input name="organization" title="Organization" value="" size="20" maxlength="50"/>
    
    <p>Email (optional):</p>
    <input name="email" title="Email" value="" size="20" maxlength="100"/>
    
    <p><small>Format: +39XXXXXXXXXX</small></p>
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/send.contact">
        <postfield name="to" value="$(to)"/>
        <postfield name="contactName" value="$(contactName)"/>
        <postfield name="phoneNumber" value="$(phoneNumber)"/>
        <postfield name="organization" value="$(organization)"/>
        <postfield name="email" value="$(email)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a> 
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-contact", "Send Contact", body));
});

// Endpoint per inviare sondaggi
app.get("/wml/send.poll.wml", (req, res) => {
  const to = esc(req.query.to || "");

  const body = `
    <p><b>Send Poll</b></p>
    <p>To: <input name="to" title="Recipient" value="${to}" size="15"/></p>
    
    <p>Poll Question:</p>
    <input name="pollName" title="Poll question" value="" size="25" maxlength="100"/>
    
    <p>Option 1:</p>
    <input name="option1" title="Option 1" value="" size="20" maxlength="50"/>
    
    <p>Option 2:</p>
    <input name="option2" title="Option 2" value="" size="20" maxlength="50"/>
    
    <p>Option 3 (optional):</p>
    <input name="option3" title="Option 3" value="" size="20" maxlength="50"/>
    
    <p>Option 4 (optional):</p>
    <input name="option4" title="Option 4" value="" size="20" maxlength="50"/>
    
    <p>Selectable Answers:</p>
    <select name="selectableCount" title="Selectable answers">
      <option value="1">1 answer</option>
      <option value="2">2 answers</option>
      <option value="3">3 answers</option>
      <option value="4">4 answers</option>
    </select>
    
    <p><small>Min 2 options, max 12 options per poll</small></p>
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/send.poll">
        <postfield name="to" value="$(to)"/>
        <postfield name="pollName" value="$(pollName)"/>
        <postfield name="option1" value="$(option1)"/>
        <postfield name="option2" value="$(option2)"/>
        <postfield name="option3" value="$(option3)"/>
        <postfield name="option4" value="$(option4)"/>
        <postfield name="selectableCount" value="$(selectableCount)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/send-menu.wml?to=${encodeURIComponent(
        to
      )}" accesskey="0">[0] Back</a> 
      <a href="/wml/contacts.wml" accesskey="9">[9] Contacts</a>
    </p>
  `;

  sendWml(res, card("send-poll", "Send Poll", body));
});

// POST handlers per ogni tipo di messaggio
app.post("/wml/send.image", async (req, res) => {
  try {
    const { to, imageUrl, caption } = req.body;
    if (!sock) throw new Error("Not connected");

    const response = await axios.get(imageUrl, {
      responseType: "arraybuffer",
      timeout: 60000, // 60 second timeout for images
      maxContentLength: 50 * 1024 * 1024, // 50MB max for images
      maxBodyLength: 50 * 1024 * 1024
    });
    const result = await sock.sendMessage(formatJid(to), {
      image: response.data,
      caption,
    });

    sendWml(
      res,
      resultCard(
        "Image Sent",
        [
          `To: ${jidFriendly(to)}`,
          `Caption: ${caption || "No caption"}`,
          `ID: ${result?.key?.id || "Unknown"}`,
        ],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (error) {
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [error.message || "Failed to send image"],
        "/wml/send.image.wml"
      )
    );
  }
});

app.post("/wml/send.video", async (req, res) => {
  try {
    const { to, videoUrl, caption } = req.body;
    if (!sock) throw new Error("Not connected");

    const response = await axios.get(videoUrl, {
      responseType: "arraybuffer",
      timeout: 120000, // 120 second timeout for videos
      maxContentLength: 100 * 1024 * 1024, // 100MB max for videos
      maxBodyLength: 100 * 1024 * 1024
    });
    const result = await sock.sendMessage(formatJid(to), {
      video: response.data,
      caption,
    });

    sendWml(
      res,
      resultCard(
        "Video Sent",
        [
          `To: ${jidFriendly(to)}`,
          `Caption: ${caption || "No caption"}`,
          `ID: ${result?.key?.id || "Unknown"}`,
        ],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (error) {
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [error.message || "Failed to send video"],
        "/wml/send.video.wml"
      )
    );
  }
});

// Send TTS audio message
app.post("/wml/send.tts", async (req, res) => {
  try {
    const { to, text, language = "en", ptt = "true" } = req.body;

    if (!sock) throw new Error("Not connected");
    if (!text || text.trim().length === 0) {
      throw new Error("Please enter text to convert to speech");
    }

    console.log(`TTS request: "${text}" in ${language} to ${to}`);

    // Convert text to speech using local espeak (returns WAV)
    const audioBuffer = await textToSpeech(text, language);

    console.log(`TTS audio generated: ${audioBuffer.length} bytes (WAV from espeak)`);

    // Convert WAV to OGG Opus for WhatsApp compatibility
    console.log('Converting WAV to OGG Opus...');
    const oggBuffer = await convertWavToOgg(audioBuffer);
    console.log(`Converted to OGG: ${oggBuffer.length} bytes`);

    // Send as WhatsApp audio message
    const result = await sock.sendMessage(formatJid(to), {
      audio: oggBuffer,
      ptt: ptt === "true",
      mimetype: "audio/ogg; codecs=opus",
    });

    sendWml(
      res,
      resultCard(
        "Voice Message Sent",
        [
          `To: ${jidFriendly(to)}`,
          `Text: "${text.substring(0, 50)}${text.length > 50 ? "..." : ""}"`,
          `Language: ${language}`,
          `Type: ${ptt === "true" ? "Voice Note (PTT)" : "Audio File"}`,
          `ID: ${result?.key?.id || "Unknown"}`,
        ],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (error) {
    console.error("TTS send error:", error);
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [error.message || "Failed to send voice message"],
        "/wml/send.tts.wml"
      )
    );
  }
});

app.post("/wml/send.audio", async (req, res) => {
  try {
    const { to, audioUrl, audioType = "audio/mp4", ptt = "false" } = req.body;
    if (!sock) throw new Error("Not connected");

    const response = await axios.get(audioUrl, {
      responseType: "arraybuffer",
      timeout: 60000, // 60 second timeout
      maxContentLength: 50 * 1024 * 1024, // 50MB max
      maxBodyLength: 50 * 1024 * 1024
    });
    const result = await sock.sendMessage(formatJid(to), {
      audio: response.data,
      ptt: ptt === "true",
      mimetype: audioType,
    });

    sendWml(
      res,
      resultCard(
        "Audio Sent",
        [
          `To: ${jidFriendly(to)}`,
          `Type: ${audioType}`,
          `Voice Message: ${ptt === "true" ? "Yes" : "No"}`,
          `ID: ${result?.key?.id || "Unknown"}`,
        ],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (error) {
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [error.message || "Failed to send audio"],
        "/wml/send.audio.wml"
      )
    );
  }
});

app.post("/wml/send.document", async (req, res) => {
  try {
    const { to, documentUrl, fileName, mimeType } = req.body;
    if (!sock) throw new Error("Not connected");

    const response = await axios.get(documentUrl, {
      responseType: "arraybuffer",
      timeout: 120000, // 120 second timeout for documents
      maxContentLength: 100 * 1024 * 1024, // 100MB max
      maxBodyLength: 100 * 1024 * 1024
    });
    const result = await sock.sendMessage(formatJid(to), {
      document: response.data,
      fileName: fileName || "document",
      mimetype: mimeType || "application/octet-stream",
    });

    sendWml(
      res,
      resultCard(
        "Document Sent",
        [
          `To: ${jidFriendly(to)}`,
          `File: ${fileName || "document"}`,
          `Type: ${mimeType || "application/octet-stream"}`,
          `ID: ${result?.key?.id || "Unknown"}`,
        ],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (error) {
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [error.message || "Failed to send document"],
        "/wml/send.document.wml"
      )
    );
  }
});

app.post("/wml/send.sticker", async (req, res) => {
  try {
    const { to, stickerUrl } = req.body;
    if (!sock) throw new Error("Not connected");

    const response = await axios.get(stickerUrl, {
      responseType: "arraybuffer",
      timeout: 60000, // 60 second timeout
      maxContentLength: 10 * 1024 * 1024, // 10MB max for stickers
      maxBodyLength: 10 * 1024 * 1024
    });
    const result = await sock.sendMessage(formatJid(to), {
      sticker: response.data,
    });

    sendWml(
      res,
      resultCard(
        "Sticker Sent",
        [`To: ${jidFriendly(to)}`, `ID: ${result?.key?.id || "Unknown"}`],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (error) {
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [error.message || "Failed to send sticker"],
        "/wml/send.sticker.wml"
      )
    );
  }
});

app.post("/wml/send.location", async (req, res) => {
  try {
    const { to, latitude, longitude, name } = req.body;
    if (!sock) throw new Error("Not connected");

    const result = await sock.sendMessage(formatJid(to), {
      location: {
        degreesLatitude: parseFloat(latitude),
        degreesLongitude: parseFloat(longitude),
        name,
      },
    });

    sendWml(
      res,
      resultCard(
        "Location Sent",
        [
          `To: ${jidFriendly(to)}`,
          `Location: ${latitude}, ${longitude}`,
          `Name: ${name || "Unnamed location"}`,
          `ID: ${result?.key?.id || "Unknown"}`,
        ],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (error) {
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [error.message || "Failed to send location"],
        "/wml/send.location.wml"
      )
    );
  }
});

app.post("/wml/send.contact", async (req, res) => {
  try {
    const { to, contactName, phoneNumber, organization, email } = req.body;
    if (!sock) throw new Error("Not connected");

    const vcard = `BEGIN:VCARD\nVERSION:3.0\nFN:${contactName}\nTEL;type=CELL:${phoneNumber}\n${
      organization ? `ORG:${organization}\n` : ""
    }${email ? `EMAIL:${email}\n` : ""}END:VCARD`;

    const result = await sock.sendMessage(formatJid(to), {
      contacts: {
        displayName: contactName,
        contacts: [
          {
            displayName: contactName,
            vcard,
          },
        ],
      },
    });

    sendWml(
      res,
      resultCard(
        "Contact Sent",
        [
          `To: ${jidFriendly(to)}`,
          `Contact: ${contactName}`,
          `Phone: ${phoneNumber}`,
          `ID: ${result?.key?.id || "Unknown"}`,
        ],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (error) {
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [error.message || "Failed to send contact"],
        "/wml/send.contact.wml"
      )
    );
  }
});

app.post("/wml/send.poll", async (req, res) => {
  try {
    const {
      to,
      pollName,
      option1,
      option2,
      option3,
      option4,
      selectableCount,
    } = req.body;
    if (!sock) throw new Error("Not connected");

    const options = [option1, option2];
    if (option3) options.push(option3);
    if (option4) options.push(option4);

    const result = await sock.sendMessage(formatJid(to), {
      poll: {
        name: pollName,
        values: options,
        selectableCount: Math.min(parseInt(selectableCount), options.length),
      },
    });

    sendWml(
      res,
      resultCard(
        "Poll Sent",
        [
          `To: ${jidFriendly(to)}`,
          `Question: ${pollName}`,
          `Options: ${options.length}`,
          `Selectable: ${selectableCount}`,
          `ID: ${result?.key?.id || "Unknown"}`,
        ],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (error) {
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [error.message || "Failed to send poll"],
        "/wml/send.poll.wml"
      )
    );
  }
});

app.get("/wml/groups.search.wml", async (req, res) => {
  try {
    if (!sock) throw new Error("Not connected");

    // Query di ricerca
    const query = (req.query.q || "").toLowerCase().trim();
    if (!query) throw new Error("No search query provided");

    // Parametri di paginazione
    const page = parseInt(req.query.page) || 1;
    const limit = 5;
    const offset = (page - 1) * limit;

    // Prendo tutti i gruppi e filtro per nome
    const groups = await sock.groupFetchAllParticipating();
    const groupList = Object.values(groups)
      .filter((g) => (g?.subject || "").toLowerCase().includes(query))
      .sort((a, b) => (b?.subject || "").localeCompare(a?.subject || ""));

    const totalGroups = groupList.length;
    const totalPages = Math.max(1, Math.ceil(totalGroups / limit));
    const paginatedGroups = groupList.slice(offset, offset + limit);

    // Escape WML
    const escWml = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");

    // Lista risultati
    const list =
      paginatedGroups
        .map((g, idx) => {
          const globalIdx = offset + idx;
          const memberCount = g?.participants?.length || 0;
          return `<p><b>${globalIdx + 1}.</b> ${escWml(
            g.subject || "Unnamed Group"
          )}<br/>
        <small>${memberCount} members | ${escWml(
            g.id.slice(-8)
          )}...</small><br/>
        <a href="/wml/chat.wml?jid=${encodeURIComponent(
          g.id
        )}&amp;limit=15">[Chat]</a>
      </p>`;
        })
        .join("") ||
      `<p>No groups found matching "<i>${escWml(query)}</i>".</p>`;

    // Controlli paginazione
    let paginationControls = "<p><b>Pages:</b><br/>";
    if (page > 1) {
      paginationControls += `<a href="/wml/groups.search.wml?q=${encodeURIComponent(
        query
      )}&amp;page=1">[First]</a> `;
      paginationControls += `<a href="/wml/groups.search.wml?q=${encodeURIComponent(
        query
      )}&amp;page=${page - 1}">[&lt;]</a> `;
    }

    const startPage = Math.max(1, page - 2);
    const endPage = Math.min(totalPages, page + 2);

    for (let i = startPage; i <= endPage; i++) {
      if (i === page) {
        paginationControls += `<b>[${i}]</b> `;
      } else {
        paginationControls += `<a href="/wml/groups.search.wml?q=${encodeURIComponent(
          query
        )}&amp;page=${i}">[${i}]</a> `;
      }
    }

    if (page < totalPages) {
      paginationControls += `<a href="/wml/groups.search.wml?q=${encodeURIComponent(
        query
      )}&amp;page=${page + 1}">[&gt;]</a> `;
      paginationControls += `<a href="/wml/groups.search.wml?q=${encodeURIComponent(
        query
      )}&amp;page=${totalPages}">[Last]</a>`;
    }

    paginationControls += `<br/><small>Page ${page} of ${totalPages} (${totalGroups} results)</small></p>`;

    // Body
    const body = `
      <p><b>Search results for: "${escWml(query)}"</b></p>
      ${list}
      ${paginationControls}
      <p><a href="/wml/groups.wml">[Back to Groups]</a></p>
      <p>
        <a href="/wml/home.wml">[Home]</a> 
        <a href="/wml/chats.wml">[Chats]</a> 
        <a href="/wml/contacts.wml">[Contacts]</a>
      </p>`;

    const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <head>
    <meta http-equiv="Cache-Control" content="max-age=0"/>
  </head>
  <card id="search" title="Group Search">
    ${body}
  </card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");

    const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
    res.send(encodedBuffer);
  } catch (e) {
    const errorWml = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <card id="error" title="Error">
    <p><b>Error:</b></p>
    <p>${e.message || "Failed to search groups"}</p>
    <p><a href="/wml/groups.wml">[Back to Groups]</a></p>
  </card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    const encodedBuffer = iconv.encode(errorWml, "iso-8859-1");
    res.send(encodedBuffer);
  }
});

app.get("/wml/groups.wml", async (req, res) => {
  try {
    if (!sock) throw new Error("Not connected");

    // Parametri di paginazione
    const page = parseInt(req.query.page) || 1;
    const limit = 5;
    const offset = (page - 1) * limit;

    // PRODUCTION-GRADE: Check cache first for better performance
    let groupList = groupsCache.get('all-groups');

    if (!groupList) {
      // Cache miss - fetch from WhatsApp (async, non-blocking)
      const groups = await sock.groupFetchAllParticipating();
      groupList = Object.values(groups).sort((a, b) =>
        (b?.subject || "").localeCompare(a?.subject || "")
      );
      // Cache the result
      groupsCache.set('all-groups', groupList);
    }

    // Calcoli per la paginazione
    const totalGroups = groupList.length;
    const totalPages = Math.ceil(totalGroups / limit);
    const paginatedGroups = groupList.slice(offset, offset + limit);

    // Escape WML sicuro
    const escWml = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");

    // Lista gruppi
    const list =
      paginatedGroups
        .map((g, idx) => {
          const globalIdx = offset + idx;
          const memberCount = g?.participants?.length || 0;
          return `<p><b>${globalIdx + 1}.</b> ${escWml(
            g.subject || "Unnamed Group"
          )}<br/>
        <small>${memberCount} members | ${escWml(
            g.id.slice(-8)
          )}...</small><br/>
        <a href="/wml/group.view.wml?gid=${encodeURIComponent(
          g.id
        )}" accesskey="${Math.min(idx + 1, 9)}">[${Math.min(
            idx + 1,
            9
          )}] Open</a> |
        <a href="/wml/chat.wml?jid=${encodeURIComponent(
          g.id
        )}&amp;limit=15">[Chat]</a>
      </p>`;
        })
        .join("") || "<p>No groups found.</p>";

    // Controlli paginazione
    let paginationControls = "";
    if (totalPages > 1) {
      paginationControls = "<p><b>Pages:</b><br/>";

      if (page > 1) {
        paginationControls += `<a href="/wml/groups.wml?page=1">[First]</a> `;
        paginationControls += `<a href="/wml/groups.wml?page=${
          page - 1
        }">[&lt;]</a> `;
      }

      const startPage = Math.max(1, page - 2);
      const endPage = Math.min(totalPages, page + 2);

      for (let i = startPage; i <= endPage; i++) {
        if (i === page) {
          paginationControls += `<b>[${i}]</b> `;
        } else {
          paginationControls += `<a href="/wml/groups.wml?page=${i}">[${i}]</a> `;
        }
      }

      if (page < totalPages) {
        paginationControls += `<a href="/wml/groups.wml?page=${
          page + 1
        }">[&gt;]</a> `;
        paginationControls += `<a href="/wml/groups.wml?page=${totalPages}">[Last]</a>`;
      }

      paginationControls += `<br/><small>Page ${page} of ${totalPages} (${totalGroups} groups)</small></p>`;
    }

    // Form ricerca
    const searchForm = `
      <p><b>Search groups:</b></p>
      <p>
        <input name="q" title="Search..." value="" emptyok="true" size="15" maxlength="30"/>
        <do type="accept" label="Search">
          <go href="/wml/groups.search.wml" method="get">
            <postfield name="q" value="$(q)"/>
          </go>
        </do>
      </p>`;

    // Body card
    const body = `
      <p><b>My Groups (${totalGroups}) - Page ${page}/${totalPages || 1}</b></p>
      ${searchForm}
      ${list}
      ${paginationControls}
      <p><b>Group Actions:</b></p>
      <p>
        <a href="/wml/group.create.wml" accesskey="*">[*] Create New Group</a>
      </p>
      <p>
        <a href="/wml/home.wml">[Home]</a> 
        <a href="/wml/chats.wml">[Chats]</a> 
        <a href="/wml/contacts.wml">[Contacts]</a>
      </p>
      <do type="accept" label="Create">
        <go href="/wml/group.create.wml"/>
      </do>
      <do type="options" label="Menu">
        <go href="/wml/menu.wml"/>
      </do>`;

    const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <head>
    <meta http-equiv="Cache-Control" content="max-age=0"/>
  </head>
  <card id="groups" title="Groups">
    ${body}
  </card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
    res.setHeader("Pragma", "no-cache");
    res.setHeader("Expires", "0");

    const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
    res.send(encodedBuffer);
  } catch (e) {
    const errorWml = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <card id="error" title="Error">
    <p><b>Error:</b></p>
    <p>${e.message || "Failed to load groups"}</p>
    <p><a href="/wml/home.wml">[Back to Home]</a></p>
  </card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    const encodedBuffer = iconv.encode(errorWml, "iso-8859-1");
    res.send(encodedBuffer);
  }
});

// Group View - View group details, participants, settings (PRODUCTION-GRADE)
// Update group metadata handling
app.get("/wml/group.view.wml", async (req, res) => {
  try {
    if (!sock) throw new Error("Not connected");

    const gid = req.query.gid || "";
    if (!gid) throw new Error("No group ID provided");

    // Fetch group metadata - now returns LID-based information
    const metadata = await sock.groupMetadata(gid);

    // Extract group info with LID support
    const groupName = metadata.subject || "Unnamed Group";
    const groupDesc = metadata.desc || "No description";
    const participants = metadata.participants || [];
    const admins = participants.filter(p => p.admin).map(p => p.id);
    const isAdmin = admins.includes(sock.user?.id);
    
    // Handle owner fields
    const owner = metadata.owner; // Now LID
    const ownerPn = metadata.ownerPn; // Phone number if available
    const createdAt = metadata.creation ? new Date(metadata.creation * 1000).toLocaleDateString() : "Unknown";

    // WML escape
    const esc = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");

    // Pagination for participants
    const page = parseInt(req.query.page) || 1;
    const limit = 10;
    const offset = (page - 1) * limit;
    const totalPages = Math.ceil(participants.length / limit);
    const paginatedParticipants = participants.slice(offset, offset + limit);

    // Participant list with LID support
    const participantList = paginatedParticipants
      .map((p, idx) => {
        const globalIdx = offset + idx;
        // Handle participant ID (could be LID or PN)
        const participantId = p.id;
        const isLid = participantId.startsWith('lid:');
        const displayName = isLid ? 
          `LID:${participantId.substring(4)}` : 
          jidFriendly(participantId);
        const role = p.admin ? " (Admin)" : "";
        return `<p>${globalIdx + 1}. ${esc(displayName)}${role}</p>`;
      })
      .join("");

    // Rest of the group view code...
  } catch (e) {
    const errorWml = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <card id="error" title="Error">
    <p><b>Error:</b></p>
    <p>${(e.message || "Failed to load group").replace(/[<>&"']/g, "")}</p>
    <p><a href="/wml/groups.wml">[Back to Groups]</a></p>
  </card>
</wml>`;
    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.send(iconv.encode(errorWml, "iso-8859-1"));
  }
});

// Group Create - Create new group (PRODUCTION-GRADE, NON-BLOCKING)
app.get("/wml/group.create.wml", (req, res) => {
  const body = `
    <p><b>Create New Group</b></p>

    <p>Enter group name:</p>
    <p>
      <input name="groupname" title="Group Name" value="" emptyok="false" size="20" maxlength="50"/>
    </p>

    <p>Add participants (phone numbers, one per line, without +):</p>
    <p>
      <input name="participants" title="Participants" value="" emptyok="true" size="20" maxlength="200"/>
    </p>

    <p><small>Example: 393331234567</small></p>

    <do type="accept" label="Create">
      <go href="/wml/group.create.action.wml" method="post">
        <postfield name="groupname" value="$(groupname)"/>
        <postfield name="participants" value="$(participants)"/>
      </go>
    </do>

    <p>
      <a href="/wml/groups.wml" accesskey="0">[0] Cancel</a>
    </p>
  `;

  sendWml(res, card("group-create", "Create Group", body));
});

// Group Create Action - Process group creation (NON-BLOCKING)
app.post("/wml/group.create.action.wml", async (req, res) => {
  try {
    if (!sock) throw new Error("Not connected");

    const groupName = (req.body.groupname || "").trim();
    const participantsStr = (req.body.participants || "").trim();

    if (!groupName) throw new Error("Group name is required");
    if (!participantsStr) throw new Error("At least one participant is required");

    // Parse participants - non-blocking
    const participantNumbers = participantsStr
      .split(/[\n,;]+/)
      .map(p => p.trim())
      .filter(p => p.length > 0)
      .map(p => formatJid(p));

    if (participantNumbers.length === 0) {
      throw new Error("No valid participants provided");
    }

    // Create group - async operation
    const group = await sock.groupCreate(groupName, participantNumbers);

    // Invalidate groups cache since we created a new group
    groupsCache.invalidate('all-groups');

    const body = `
      <p><b>Group Created!</b></p>
      <p>Name: ${esc(groupName)}</p>
      <p>Members: ${participantNumbers.length}</p>
      <p>ID: ${esc(group.id.slice(-12))}...</p>

      <p>
        <a href="/wml/group.view.wml?gid=${encodeURIComponent(group.id)}" accesskey="1">[1] View Group</a><br/>
        <a href="/wml/chat.wml?jid=${encodeURIComponent(group.id)}&amp;limit=15" accesskey="2">[2] Open Chat</a><br/>
        <a href="/wml/groups.wml" accesskey="0">[0] All Groups</a>
      </p>
    `;

    sendWml(res, card("group-created", "Success", body));
  } catch (e) {
    const body = `
      <p><b>Error Creating Group</b></p>
      <p>${esc(e.message || "Failed to create group")}</p>
      <p>
        <a href="/wml/group.create.wml">[Try Again]</a><br/>
        <a href="/wml/groups.wml">[Back to Groups]</a>
      </p>
    `;
    sendWml(res, card("error", "Error", body));
  }
});

// Group Leave - Leave group (NON-BLOCKING)
app.get("/wml/group.leave.wml", async (req, res) => {
  try {
    const gid = req.query.gid || "";
    if (!gid) throw new Error("No group ID provided");

    const confirmed = req.query.confirm === "yes";

    if (!confirmed) {
      // Show confirmation page
      const metadata = await sock.groupMetadata(gid);
      const groupName = metadata.subject || "Unnamed Group";

      const body = `
        <p><b>Leave Group?</b></p>
        <p>Are you sure you want to leave:</p>
        <p><b>${esc(groupName)}</b></p>

        <p>
          <a href="/wml/group.leave.wml?gid=${encodeURIComponent(gid)}&amp;confirm=yes" accesskey="1">[1] Yes, Leave</a><br/>
          <a href="/wml/group.view.wml?gid=${encodeURIComponent(gid)}" accesskey="0">[0] Cancel</a>
        </p>
      `;

      sendWml(res, card("leave-confirm", "Confirm", body));
    } else {
      // Execute leave - non-blocking
      await sock.groupLeave(gid);

      // Invalidate groups cache since we left a group
      groupsCache.invalidate('all-groups');

      const body = `
        <p><b>Left Group</b></p>
        <p>You have left the group successfully.</p>
        <p>
          <a href="/wml/groups.wml" accesskey="1">[1] All Groups</a><br/>
          <a href="/wml/home.wml" accesskey="0">[0] Home</a>
        </p>
      `;

      sendWml(res, card("left", "Success", body));
    }
  } catch (e) {
    const body = `
      <p><b>Error</b></p>
      <p>${esc(e.message || "Failed to leave group")}</p>
      <p><a href="/wml/groups.wml">[Back to Groups]</a></p>
    `;
    sendWml(res, card("error", "Error", body));
  }
});

// Status Broadcast - Main page for posting status updates (PRODUCTION-GRADE)
app.get("/wml/status-broadcast.wml", (req, res) => {
  const body = `
    <p><b>Status Broadcast</b></p>
    <p>Post updates to your WhatsApp Status (visible to all your contacts for 24 hours)</p>

    <p><b>Select status type:</b></p>
    <p>
      <a href="/wml/status.text.wml" accesskey="1">[1] Text Status</a><br/>
      <a href="/wml/status.image.wml" accesskey="2">[2] Image Status</a><br/>
      <a href="/wml/status.video.wml" accesskey="3">[3] Video Status</a>
    </p>

    <p><b>Info:</b></p>
    <p><small>Status updates disappear after 24 hours and are visible to all your contacts.</small></p>

    <p>
      <a href="/wml/home.wml" accesskey="0">[0] Home</a> |
      <a href="/wml/menu.wml">[Menu]</a>
    </p>

    <do type="accept" label="Text">
      <go href="/wml/status.text.wml"/>
    </do>
  `;

  sendWml(res, card("status-broadcast", "Status", body));
});

// Status Text - Post text status (FAST, NON-BLOCKING)
app.get("/wml/status.text.wml", (req, res) => {
  const body = `
    <p><b>Post Text Status</b></p>

    <p>Enter your status message:</p>
    <p>
      <input name="text" title="Status Text" value="" emptyok="false" size="25" maxlength="700"/>
    </p>

    <p><small>Max 700 characters. Visible for 24 hours.</small></p>

    <do type="accept" label="Post">
      <go href="/wml/status.text.action.wml" method="post">
        <postfield name="text" value="$(text)"/>
      </go>
    </do>

    <p>
      <a href="/wml/status-broadcast.wml" accesskey="0">[0] Back</a>
    </p>
  `;

  sendWml(res, card("status-text", "Text Status", body));
});

// Status Text Action - Post text status (NON-BLOCKING, PRODUCTION-GRADE)
app.post("/wml/status.text.action.wml", async (req, res) => {
  try {
    if (!sock) throw new Error("Not connected");

    const text = (req.body.text || "").trim();
    if (!text) throw new Error("Status text cannot be empty");

    // Post status - async, non-blocking
    const result = await sock.sendMessage("status@broadcast", { text });

    const body = `
      <p><b>Status Posted!</b></p>
      <p><small>Your status update has been broadcast to all your contacts.</small></p>

      <p><b>Preview:</b></p>
      <p>${esc(text.substring(0, 100))}${text.length > 100 ? "..." : ""}</p>

      <p><small>ID: ${result?.key?.id?.slice(-8) || "Unknown"}</small></p>

      <p>
        <a href="/wml/status-broadcast.wml" accesskey="1">[1] Post Another</a><br/>
        <a href="/wml/home.wml" accesskey="0">[0] Home</a>
      </p>
    `;

    sendWml(res, card("status-posted", "Success", body));
  } catch (e) {
    const body = `
      <p><b>Error Posting Status</b></p>
      <p>${esc(e.message || "Failed to post status")}</p>
      <p>
        <a href="/wml/status.text.wml">[Try Again]</a><br/>
        <a href="/wml/status-broadcast.wml">[Back]</a>
      </p>
    `;
    sendWml(res, card("error", "Error", body));
  }
});

// Status Image - Post image status (NON-BLOCKING)
app.get("/wml/status.image.wml", (req, res) => {
  const body = `
    <p><b>Post Image Status</b></p>

    <p>Enter image URL:</p>
    <p>
      <input name="url" title="Image URL" value="" emptyok="false" size="30" maxlength="500"/>
    </p>

    <p>Optional caption:</p>
    <p>
      <input name="caption" title="Caption" value="" emptyok="true" size="25" maxlength="200"/>
    </p>

    <p><small>Image will be visible for 24 hours.</small></p>

    <do type="accept" label="Post">
      <go href="/wml/status.image.action.wml" method="post">
        <postfield name="url" value="$(url)"/>
        <postfield name="caption" value="$(caption)"/>
      </go>
    </do>

    <p>
      <a href="/wml/status-broadcast.wml" accesskey="0">[0] Back</a>
    </p>
  `;

  sendWml(res, card("status-image", "Image Status", body));
});

// Status Image Action - Post image status (NON-BLOCKING, PRODUCTION-GRADE)
app.post("/wml/status.image.action.wml", async (req, res) => {
  try {
    if (!sock) throw new Error("Not connected");

    const url = (req.body.url || "").trim();
    const caption = (req.body.caption || "").trim();

    if (!url) throw new Error("Image URL is required");

    // Download image - async, non-blocking
    const response = await axios.get(url, {
      responseType: "arraybuffer",
      timeout: 30000,
    });

    const imageBuffer = Buffer.from(response.data);

    // Post status with image - async
    const messageOptions = { image: imageBuffer };
    if (caption) messageOptions.caption = caption;

    const result = await sock.sendMessage("status@broadcast", messageOptions);

    const body = `
      <p><b>Image Status Posted!</b></p>
      <p><small>Your image status has been broadcast to all your contacts.</small></p>

      ${caption ? `<p><b>Caption:</b> ${esc(caption)}</p>` : ""}

      <p><small>ID: ${result?.key?.id?.slice(-8) || "Unknown"}</small></p>

      <p>
        <a href="/wml/status-broadcast.wml" accesskey="1">[1] Post Another</a><br/>
        <a href="/wml/home.wml" accesskey="0">[0] Home</a>
      </p>
    `;

    sendWml(res, card("status-posted", "Success", body));
  } catch (e) {
    const body = `
      <p><b>Error Posting Image Status</b></p>
      <p>${esc(e.message || "Failed to post image status")}</p>
      <p>
        <a href="/wml/status.image.wml">[Try Again]</a><br/>
        <a href="/wml/status-broadcast.wml">[Back]</a>
      </p>
    `;
    sendWml(res, card("error", "Error", body));
  }
});

// Status Video - Post video status (NON-BLOCKING)
app.get("/wml/status.video.wml", (req, res) => {
  const body = `
    <p><b>Post Video Status</b></p>

    <p>Enter video URL:</p>
    <p>
      <input name="url" title="Video URL" value="" emptyok="false" size="30" maxlength="500"/>
    </p>

    <p>Optional caption:</p>
    <p>
      <input name="caption" title="Caption" value="" emptyok="true" size="25" maxlength="200"/>
    </p>

    <p><small>Video will be visible for 24 hours.</small></p>

    <do type="accept" label="Post">
      <go href="/wml/status.video.action.wml" method="post">
        <postfield name="url" value="$(url)"/>
        <postfield name="caption" value="$(caption)"/>
      </go>
    </do>

    <p>
      <a href="/wml/status-broadcast.wml" accesskey="0">[0] Back</a>
    </p>
  `;

  sendWml(res, card("status-video", "Video Status", body));
});

// Status Video Action - Post video status (NON-BLOCKING, PRODUCTION-GRADE)
app.post("/wml/status.video.action.wml", async (req, res) => {
  try {
    if (!sock) throw new Error("Not connected");

    const url = (req.body.url || "").trim();
    const caption = (req.body.caption || "").trim();

    if (!url) throw new Error("Video URL is required");

    // Download video - async, non-blocking
    const response = await axios.get(url, {
      responseType: "arraybuffer",
      timeout: 60000, // 60 seconds for larger videos
    });

    const videoBuffer = Buffer.from(response.data);

    // Post status with video - async
    const messageOptions = { video: videoBuffer };
    if (caption) messageOptions.caption = caption;

    const result = await sock.sendMessage("status@broadcast", messageOptions);

    const body = `
      <p><b>Video Status Posted!</b></p>
      <p><small>Your video status has been broadcast to all your contacts.</small></p>

      ${caption ? `<p><b>Caption:</b> ${esc(caption)}</p>` : ""}

      <p><small>ID: ${result?.key?.id?.slice(-8) || "Unknown"}</small></p>

      <p>
        <a href="/wml/status-broadcast.wml" accesskey="1">[1] Post Another</a><br/>
        <a href="/wml/home.wml" accesskey="0">[0] Home</a>
      </p>
    `;

    sendWml(res, card("status-posted", "Success", body));
  } catch (e) {
    const body = `
      <p><b>Error Posting Video Status</b></p>
      <p>${esc(e.message || "Failed to post video status")}</p>
      <p>
        <a href="/wml/status.video.wml">[Try Again]</a><br/>
        <a href="/wml/status-broadcast.wml">[Back]</a>
      </p>
    `;
    sendWml(res, card("error", "Error", body));
  }
});

app.get("/wml/search.results.wml", (req, res) => {
  const q = String(req.query.q || "").trim();
  const searchType = req.query.type || "messages";
  const limitParam = req.query.limit || "10";
  const limit =
    limitParam === "all"
      ? Infinity
      : Math.max(1, Math.min(50, parseInt(limitParam)));
  const page = Math.max(1, parseInt(req.query.page || "1"));
  const pageSize = 5;
  if (!q || q.length < 2) {
    sendWml(
      res,
      resultCard(
        "Search Error",
        ["Query must be at least 2 characters"],
        "/wml/home.wml"
      )
    );
    return;
  }
  let allResults = [];
  const searchLower = q.toLowerCase();
  // funzione sicura per troncare
  function truncate(str, n) {
    return str && str.length > n ? str.slice(0, n) + "..." : str;
  }
  // Funzione per formattare il timestamp
  function formatTimestamp(timestamp) {
    if (!timestamp || timestamp === 0) return "Unknown";
    return new Date(Number(timestamp) * 1000).toLocaleString("en-GB", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  }
  if (searchType === "messages") {
    for (const [chatId, messages] of chatStore.entries()) {
      for (const msg of messages) {
        const content = extractMessageContent(msg.message);
        let text = "";
        let messageType = "";

        if (content?.conversation) {
          text = content.conversation;
          messageType = "text";
        } else if (content?.extendedTextMessage?.text) {
          text = content.extendedTextMessage.text;
          messageType = "text";
        } else if (content?.imageMessage) {
          text = "[Image] " + (content.imageMessage.caption || "");
          messageType = "image";
        } else if (content?.videoMessage) {
          text = "[Video] " + (content.videoMessage.caption || "");
          messageType = "video";
        } else if (content?.audioMessage) {
          const duration = content.audioMessage.seconds || 0;
          text = `[Audio ${duration}s]`;
          messageType = "audio";

          if (
            msg.transcription &&
            msg.transcription !== "[Trascrizione fallita]" &&
            msg.transcription !== "[Audio troppo lungo per la trascrizione]"
          ) {
            text += " " + msg.transcription;
          }
        } else if (content?.documentMessage) {
          text = "[Document] " + (content.documentMessage.fileName || "");
          messageType = "document";
        } else if (content?.stickerMessage) {
          text = "[Sticker]";
          messageType = "sticker";
        } else if (content?.locationMessage) {
          text = "[Location]";
          messageType = "location";
        } else if (content?.contactMessage) {
          text = "[Contact]";
          messageType = "contact";
        }

        const searchableText = text.toLowerCase();
        const typeSearchable = messageType.toLowerCase();

        if (
          searchableText.includes(searchLower) ||
          typeSearchable.includes(searchLower)
        ) {
          const contact = contactStore.get(chatId);
          const chatName =
            contact?.name || contact?.notify || jidFriendly(chatId);
          const timestamp = Number(msg.messageTimestamp) || 0;

          allResults.push({
            type: "message",
            chatId,
            chatName,
            messageId: msg.key.id,
            text: truncate(text, 40),
            timestamp: timestamp, // Timestamp numerico per ordinamento
            formattedTime: formatTimestamp(timestamp), // Timestamp formattato per visualizzazione
            fromMe: msg.key.fromMe,
            messageType: messageType,
            audioInfo:
              messageType === "audio"
                ? {
                    duration: content?.audioMessage?.seconds || 0,
                    hasTranscription: !!(
                      msg.transcription &&
                      msg.transcription !== "[Trascrizione fallita]" &&
                      msg.transcription !==
                        "[Audio troppo lungo per la trascrizione]"
                    ),
                  }
                : null,
          });

          if (limit !== Infinity && allResults.length >= limit) break;
        }
      }
      if (limit !== Infinity && allResults.length >= limit) break;
    }
    // Ordina i messaggi per timestamp decrescente
    allResults.sort((a, b) => b.timestamp - a.timestamp);
  } else if (searchType === "contacts") {
    const contacts = Array.from(contactStore.values()).filter((c) => {
      const name = (c.name || c.notify || c.verifiedName || "").toLowerCase();
      const number = c.id.replace("@s.whatsapp.net", "");
      return name.includes(searchLower) || number.includes(searchLower);
    });

    const limitedContacts =
      limit === Infinity ? contacts : contacts.slice(0, limit);
    allResults = limitedContacts.map((c) => {
      // Cerca l'ultimo messaggio con questo contatto
      const messages = chatStore.get(c.id) || [];
      const lastMessage =
        messages.length > 0 ? messages[messages.length - 1] : null;
      const timestamp = lastMessage ? Number(lastMessage.messageTimestamp) : 0;

      return {
        type: "contact",
        name: c.name || c.notify || c.verifiedName || "Unknown",
        number: jidFriendly(c.id),
        jid: c.id,
        timestamp: timestamp, // Timestamp numerico per ordinamento
        formattedTime: formatTimestamp(timestamp), // Timestamp formattato per visualizzazione
        lastMessageText: lastMessage
          ? truncate(messageText(lastMessage), 30)
          : "No messages",
      };
    });
    // Ordina i contatti per timestamp decrescente
    allResults.sort((a, b) => b.timestamp - a.timestamp);
  } else if (searchType === "chats") {
    for (const [chatId, messages] of chatStore.entries()) {
      const contact = contactStore.get(chatId);
      const isGroup = chatId.endsWith("@g.us");
      let chatName = isGroup
        ? contact?.subject || contact?.name || `Unnamed Group`
        : contact?.name ||
          contact?.notify ||
          contact?.verifiedName ||
          jidFriendly(chatId);

      if (chatName.toLowerCase().includes(searchLower)) {
        const lastMessage =
          messages.length > 0 ? messages[messages.length - 1] : null;
        const timestamp = lastMessage
          ? Number(lastMessage.messageTimestamp)
          : 0;
        const formattedTime = formatTimestamp(timestamp);

        allResults.push({
          type: "chat",
          chatId,
          chatName,
          isGroup,
          messageCount: messages.length,
          timestamp: timestamp, // Timestamp numerico per ordinamento
          formattedTime: formattedTime, // Timestamp formattato per visualizzazione
          phoneNumber: isGroup ? null : chatId.replace("@s.whatsapp.net", ""),
        });

        if (limit !== Infinity && allResults.length >= limit) break;
      }
    }
    // Ordina le chat per timestamp decrescente
    allResults.sort((a, b) => b.timestamp - a.timestamp);
  } else if (searchType === "groups") {
    for (const [chatId, messages] of chatStore.entries()) {
      if (!chatId.endsWith("@g.us")) continue;
      const contact = contactStore.get(chatId);
      const groupName = contact?.subject || contact?.name || `Unnamed Group`;

      if (groupName.toLowerCase().includes(searchLower)) {
        const lastMessage =
          messages.length > 0 ? messages[messages.length - 1] : null;
        const timestamp = lastMessage
          ? Number(lastMessage.messageTimestamp)
          : 0;
        const formattedTime = formatTimestamp(timestamp);
        let memberCount = 0;
        if (contact?.participants) memberCount = contact.participants.length;

        allResults.push({
          type: "group",
          chatId,
          chatName: groupName,
          messageCount: messages.length,
          timestamp: timestamp, // Timestamp numerico per ordinamento
          formattedTime: formattedTime, // Timestamp formattato per visualizzazione
          memberCount,
        });

        if (limit !== Infinity && allResults.length >= limit) break;
      }
    }
    // Ordina i gruppi per timestamp decrescente
    allResults.sort((a, b) => b.timestamp - a.timestamp);
  }

  // pagination
  const totalResults = allResults.length;
  const totalPages = Math.ceil(totalResults / pageSize) || 1;
  const startIndex = (page - 1) * pageSize;
  const endIndex = Math.min(startIndex + pageSize, totalResults);
  const paginatedResults = allResults.slice(startIndex, endIndex);

  const resultList =
    paginatedResults
      .map((r, idx) => {
        const globalIndex = startIndex + idx + 1;
        if (r.type === "message") {
          let messagePrefix = "";
          if (r.messageType === "audio") {
            messagePrefix = "[AUDIO] ";
            if (r.audioInfo?.hasTranscription) {
              messagePrefix += "[TRANSCRIPTION] ";
            }
          } else if (r.messageType === "image") {
            messagePrefix = "[IMG] ";
          } else if (r.messageType === "video") {
            messagePrefix = "[VID] ";
          } else if (r.messageType === "document") {
            messagePrefix = "[DOC] ";
          } else if (r.messageType === "sticker") {
            messagePrefix = "[STICK] ";
          }

          return `<p><b>${globalIndex}.</b> ${messagePrefix}${esc(r.text)}<br/>
        <small>From: ${esc(r.chatName)} | ${r.formattedTime} | ${
            r.fromMe ? "Me" : "Them"
          }</small><br/>
        <a href="/wml/chat.wml?jid=${encodeURIComponent(
          r.chatId
        )}&amp;limit=15">[Open Chat]</a> 
        <a href="/wml/msg.wml?mid=${encodeURIComponent(
          r.messageId
        )}&amp;jid=${encodeURIComponent(r.chatId)}">[Message]</a>
        ${
          r.messageType === "audio" && r.audioInfo?.hasTranscription
            ? ` <a href="/wml/audio-transcription.wml?mid=${encodeURIComponent(
                r.messageId
              )}&amp;jid=${encodeURIComponent(r.chatId)}">[TRANSCRIPTION]</a>`
            : ""
        }
      </p>`;
        } else if (r.type === "contact") {
          return `<p><b>${globalIndex}.</b> ${esc(r.name)}<br/>
        <small>${esc(r.number)} | Last: ${r.formattedTime}</small><br/>
        <small>Last msg: ${esc(r.lastMessageText)}</small><br/>
        <a href="/wml/contact.wml?jid=${encodeURIComponent(r.jid)}">[View]</a> |
        <a href="/wml/chat.wml?jid=${encodeURIComponent(
          r.jid
        )}&amp;limit=15">[Chat]</a>
      </p>`;
        } else if (r.type === "chat") {
          const typeIcon = r.isGroup ? "[GROUP]" : "[CHAT]";
          const phoneInfo = r.phoneNumber ? ` | ${r.phoneNumber}` : "";
          return `<p><b>${globalIndex}.</b> ${typeIcon} ${esc(r.chatName)}<br/>
        <small>${r.messageCount} messages | Last: ${
            r.formattedTime
          }${phoneInfo}</small><br/>
        <a href="/wml/chat.wml?jid=${encodeURIComponent(
          r.chatId
        )}&amp;limit=15">[Open]</a> |
        <a href="/wml/send.text.wml?to=${encodeURIComponent(
          r.chatId
        )}">[Send]</a>
        ${
          r.phoneNumber
            ? ` | <a href="wtai://wp/mc;${r.phoneNumber}">[Call]</a>`
            : ""
        }
      </p>`;
        } else if (r.type === "group") {
          const memberInfo =
            r.memberCount > 0 ? ` | ${r.memberCount} members` : "";
          return `<p><b>${globalIndex}.</b> [GROUP] ${esc(r.chatName)}<br/>
        <small>${r.messageCount} messages | Last: ${
            r.formattedTime
          }${memberInfo}</small><br/>
        <a href="/wml/chat.wml?jid=${encodeURIComponent(
          r.chatId
        )}&amp;limit=15">[Open]</a> |
        <a href="/wml/send.text.wml?to=${encodeURIComponent(
          r.chatId
        )}">[Send]</a>
      </p>`;
        }
        return "";
      })
      .join("") || "<p>No results found.</p>";

  // pagination controls (come prima)
  let paginationControls = "";
  if (totalPages > 1) {
    paginationControls = "<p><b>Pages:</b><br/>";
    if (page > 1) {
      paginationControls += `<a href="/wml/search.results.wml?q=${encodeURIComponent(
        q
      )}&amp;type=${encodeURIComponent(
        searchType
      )}&amp;limit=${encodeURIComponent(
        limitParam
      )}&amp;page=1">[&lt;&lt; First]</a> `;
      paginationControls += `<a href="/wml/search.results.wml?q=${encodeURIComponent(
        q
      )}&amp;type=${encodeURIComponent(
        searchType
      )}&amp;limit=${encodeURIComponent(limitParam)}&amp;page=${
        page - 1
      }">[&lt; Prev]</a> `;
    }
    const startPage = Math.max(1, page - 2);
    const endPage = Math.min(totalPages, page + 2);
    if (startPage > 1) {
      paginationControls += `<a href="/wml/search.results.wml?q=${encodeURIComponent(
        q
      )}&amp;type=${encodeURIComponent(
        searchType
      )}&amp;limit=${encodeURIComponent(limitParam)}&amp;page=1">[1]</a> `;
      if (startPage > 2) paginationControls += "... ";
    }
    for (let i = startPage; i <= endPage; i++) {
      if (i === page) paginationControls += `<b>[${i}]</b> `;
      else
        paginationControls += `<a href="/wml/search.results.wml?q=${encodeURIComponent(
          q
        )}&amp;type=${encodeURIComponent(
          searchType
        )}&amp;limit=${encodeURIComponent(
          limitParam
        )}&amp;page=${i}">[${i}]</a> `;
    }
    if (endPage < totalPages) {
      if (endPage < totalPages - 1) paginationControls += "... ";
      paginationControls += `<a href="/wml/search.results.wml?q=${encodeURIComponent(
        q
      )}&amp;type=${encodeURIComponent(
        searchType
      )}&amp;limit=${encodeURIComponent(
        limitParam
      )}&amp;page=${totalPages}">[${totalPages}]</a> `;
    }
    if (page < totalPages) {
      paginationControls += `<a href="/wml/search.results.wml?q=${encodeURIComponent(
        q
      )}&amp;type=${encodeURIComponent(
        searchType
      )}&amp;limit=${encodeURIComponent(limitParam)}&amp;page=${
        page + 1
      }">[Next &gt;]</a> `;
      paginationControls += `<a href="/wml/search.results.wml?q=${encodeURIComponent(
        q
      )}&amp;type=${encodeURIComponent(
        searchType
      )}&amp;limit=${encodeURIComponent(
        limitParam
      )}&amp;page=${totalPages}">[Last &gt;&gt;]</a>`;
    }
    paginationControls += "</p>";
  }

  const limitDisplay = limitParam === "all" ? "No limit" : limitParam;
  const searchTypeDisplay =
    searchType.charAt(0).toUpperCase() + searchType.slice(1);
  const body = `
    <p><b>Search Results</b></p>
    <p>Query: <b>${esc(q)}</b></p>
    <p>Type: ${esc(searchTypeDisplay)} | Limit: ${limitDisplay}</p>
    <p>Page: ${page}/${totalPages} | Total: ${totalResults}</p>
    <p>Showing: ${startIndex + 1}-${endIndex} of ${totalResults}</p>
    <p><small>Sorted by most recent first</small></p>
    ${resultList}
    ${paginationControls}
    <p><b>Search Again:</b></p>
    <p>
      <a href="/wml/search.wml?q=${encodeURIComponent(
        q
      )}" accesskey="1">[1] New Search</a> |
      <a href="/wml/home.wml" accesskey="0">[0] Home</a>
    </p>
    <do type="accept" label="Home">
      <go href="/wml/home.wml"/>
    </do>
  `;
  sendWml(res, card("search-results", "Search Results", body));
});

// Enhanced Search Form
app.get("/wml/search.wml", (req, res) => {
  const prevQuery = esc(req.query.q || "");
  const prevLimit = req.query.limit || "5"; // default a 10

  const body = `
    <p><b>Search WhatsApp</b></p>
        
    <p>Search for:</p>
    <input name="q" title="Search query" value="${prevQuery}" size="20" maxlength="100"/>
        
    <p>Search in:</p>
    <select name="type" title="Search Type">
      <option value="messages">Messages</option>
      <option value="contacts">Contacts</option>
      <option value="chats">Chats/Conversations</option>
      <option value="groups">Groups Only</option>
    </select>
        
    <p>Limit:</p>
    <select name="limit" title="Max Results">
      <option value="5" ${
        prevLimit === "5" ? 'selected="selected"' : ""
      }>5 results</option>
      <option value="10" ${
        prevLimit === "10" ? 'selected="selected"' : ""
      }>10 results</option>
      <option value="20" ${
        prevLimit === "20" ? 'selected="selected"' : ""
      }>20 results</option>
      <option value="50" ${
        prevLimit === "50" ? 'selected="selected"' : ""
      }>50 results</option>
      <option value="all" ${
        prevLimit === "all" ? 'selected="selected"' : ""
      }>No limit</option>
    </select>

    <do type="accept" label="Search">
      <go href="/wml/search.results.wml" method="get">
        <postfield name="q" value="$(q)"/>
        <postfield name="type" value="$(type)"/>
        <postfield name="limit" value="$(limit)"/>
        <postfield name="chatJid" value="$(chatJid)"/>
      </go>
    </do>

    ${navigationBar()}
  `;

  sendWml(res, card("search", "Search", body));
});

// Auto-refresh for dynamic content
app.get("/wml/live-status.wml", (req, res) => {
  const refreshInterval = req.query.interval || "30";

  const body = `
   
    <p><b>Live Status Monitor</b></p>
    <p>Updates every ${refreshInterval} seconds</p>
    
    <p><b>Connection:</b> ${connectionState}</p>
    <p><b>Messages:</b> ${messageStore.size}</p>
    <p><b>Contacts:</b> ${contactStore.size}</p>
    <p><b>Chats:</b> ${chatStore.size}</p>
    <p><b>Time:</b> ${new Date().toLocaleTimeString()}</p>
    
  
    
    <p><a href="/wml/home.wml" accesskey="0">[0] Home</a></p>
    
   
  `;

  sendWml(
    res,
    card(
      "live-status",
      "Live Status",
      body,
      `/wml/live-status.wml?interval=${refreshInterval}`
    )
  );
});

// Add all the existing endpoints from your original code here...
// [Previous POST handlers for send.text, send.image, etc.]

// Keep all existing POST handlers and API endpoints
app.post("/wml/send.text", async (req, res) => {
  try {
    if (!sock) throw new Error("Not connected");
    const { to, message } = req.body;
    const result = await sock.sendMessage(formatJid(to), { text: message });
    sendWml(
      res,
      resultCard(
        "Message Sent",
        [
          `To: ${jidFriendly(to)}`,
          `Message: ${truncate(message, 50)}`,
          `ID: ${result?.key?.id || "Unknown"}`,
        ],
        `/wml/send-menu.wml?to=${encodeURIComponent(to)}`
      )
    );
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Send Failed",
        [e.message || "Failed to send"],
        "/wml/send.text.wml"
      )
    );
  }
});

// Enhanced sync functions
async function loadChatHistory(jid, limit = 9999999999999) {
  if (!sock) return;
  try {
    // In production, implement proper message fetching
    logger.info(`Loading chat history for ${jid}, limit: ${limit}`);
  } catch (error) {
    logger.error(`Failed to load chat history: ${error.message}`);
  }
}

async function performInitialSync() {
  try {
    if (!sock || connectionState !== "open") {
      logger.warn("Cannot sync: not connected");
      return;
    }

    logger.info(`Starting enhanced initial sync (attempt ${syncAttempts + 1})`);
    syncAttempts++;

    let successCount = 0;

    // Sync contacts
    try {
      logger.info("Checking contacts...");
      if (contactStore.size === 0) {
        logger.info("Waiting for contacts via events...");
        await delay(3000);
      }
      logger.info(`Contacts in store: ${contactStore.size}`);
      successCount++;
    } catch (err) {
      logger.error("Contact sync failed:", err.message);
    }

    // Sync chats
    try {
      logger.info("Fetching chats...");
      const groups = await sock.groupFetchAllParticipating();
      logger.info(`Retrieved ${Object.keys(groups).length} groups`);

      for (const chatId of Object.keys(groups)) {
        if (!chatStore.has(chatId)) {
          chatStore.set(chatId, []);
        }
      }

      if (chatStore.size === 0) {
        logger.info("Waiting for chats via events...");
        await delay(3000);
      }

      logger.info(`Chats in store: ${chatStore.size}`);
      successCount++;
    } catch (err) {
      logger.error("Chat sync failed:", err.message);
    }

    // Check sync completion
    const counts = {
      contacts: contactStore.size,
      chats: chatStore.size,
      messages: messageStore.size,
    };

    logger.info("Sync results:", counts);

    if (counts.contacts > 0 && counts.chats > 0) {
      isFullySynced = true;
      logger.info("Initial sync completed successfully!");
    } else if (syncAttempts < 9999999) {
      const delayMs = syncAttempts * 5000;
      logger.info(`Sync incomplete, retrying in ${delayMs / 1000}s...`);
      setTimeout(performInitialSync, delayMs);
    } else {
      logger.warn("Sync attempts exhausted. Data may still load gradually.");
    }
  } catch (err) {
    logger.error("Initial sync failed:", err);
    if (syncAttempts < 999999) {
      setTimeout(performInitialSync, 5000);
    }
  }
}

// Helper function for saving messages to DB
function saveMessageToDB(msg, jid) {
  if (!msg || !msg.key || !msg.key.id) return;

  try {
    // Save to message store
    messageStore.set(msg.key.id, msg);

    // Add to chat store if not exists
    if (!chatStore.has(jid)) {
      chatStore.set(jid, []);
    }

    const chatMessages = chatStore.get(jid);
    if (!chatMessages.some(m => m.key?.id === msg.key.id)) {
      chatMessages.push(msg);
    }
  } catch (error) {
    logger.error('Error saving message to DB:', error.message);
  }
}

// ============ LOAD CHAT UTILS DEPENDENCIES ============
// Dependencies will be initialized after socket creation in connectWithBetterSync()
// This ensures sock is not null when passed to loadChatUtils

// Production-ready connection with better error handling
async function connectWithBetterSync() {
  // Prevent race conditions - only one connection attempt at a time
  if (isConnecting) {
    logger.warn('Connection already in progress, skipping duplicate attempt');
    return;
  }

  isConnecting = true;

  try {
  const { state, saveCreds } = await useMultiFileAuthState("./auth_info_baileys", {
  signalKeys: {
    'lid-mapping': true,
    'device-list': true,
    'tctoken': true
  }
});
    const { version } = await fetchLatestBaileysVersion();

    sock = makeWASocket({
      version,
      auth: state,
      printQRInTerminal: false,
      syncFullHistory: true,
      markOnlineOnConnect: false,
      emitOwnEvents: true,
      getMessage: async (key) => messageStore.get(key.id) || null,
      shouldIgnoreJid: (jid) => false,
      shouldSyncHistoryMessage: (msg) => true,
      browser: ["WhatsApp WML Gateway", "Chrome", "1.0.0"],
      connectTimeoutMs: 60000,
      defaultQueryTimeoutMs: 60000,
      keepAliveIntervalMs: 10000,
      retryRequestDelayMs: 1000,
    });

    // Initialize loadChatUtils dependencies NOW that sock is created
    initializeDependencies({
      logger: logger,
      sock: sock,
      chatStore: chatStore,
      messageStore: messageStore,
      connectionState: connectionState,
      formatJid: formatJid,
      delay: delay,
      saveMessageToDB: saveMessageToDB,
      performInitialSync: performInitialSync
    });
    logger.info('✓ loadChatUtils dependencies initialized with active socket');

    sock.ev.on("creds.update", saveCreds);

    sock.ev.on(
      "connection.update",
      async ({ connection, lastDisconnect, qr }) => {
        connectionState = connection;

        if (qr) {
          currentQR = qr;
          logger.info("QR Code generated");
          if (isDev) {
            qrcode.generate(qr, { small: true });
          }
        }

        if (connection === "close") {
          const shouldReconnect =
            lastDisconnect?.error?.output?.statusCode !==
            DisconnectReason.loggedOut;
          logger.info(
            `Connection closed. Should reconnect: ${shouldReconnect}`
          );

          if (shouldReconnect) {
            const delay = Math.min(5000 * Math.pow(2, syncAttempts), 30000); // Exponential backoff
            setTimeout(connectWithBetterSync, delay);
          } else {
            // Clear stores on logout
            contactStore.clear();
            chatStore.clear();
            messageStore.clear();
            isFullySynced = false;
            syncAttempts = 0;
          }
        } else if (connection === "open") {
          logger.info("WhatsApp connected successfully!");
          currentQR = null;

          // Only reset isFullySynced if we don't have data from disk
          // This prevents "syncing..." state when reconnecting with persistent data
          if (contactStore.size === 0 && chatStore.size === 0) {
            isFullySynced = false;
          } else {
            logger.info(`Keeping data from disk: ${contactStore.size} contacts, ${chatStore.size} chats`);
            // Mark as synced if we have data, will be updated by events
            isFullySynced = true;
          }

          syncAttempts = 0;

          // Start sync process to update with latest data
          setTimeout(enhancedInitialSync, 5000);
        }
      }
    );

    // Enhanced event handlers
    sock.ev.on(
      "messaging-history.set",
      ({ chats, contacts, messages, isLatest }) => {
        logger.info(
          `History batch - Chats: ${chats.length}, Contacts: ${contacts.length}, Messages: ${messages.length}`
        );

        // Separate JID and LID chats, process JID first
        const jidChats = chats.filter(c => !c.id.startsWith('lid:'));
        const lidChats = chats.filter(c => c.id.startsWith('lid:'));

        for (const chat of [...jidChats, ...lidChats]) {
          if (!chatStore.has(chat.id)) {
            chatStore.set(chat.id, []);
          }
        }

        // Separate JID and LID contacts, process JID first
        const jidContacts = contacts.filter(c => !c.id.startsWith('lid:'));
        const lidContacts = contacts.filter(c => c.id.startsWith('lid:'));

        for (const contact of [...jidContacts, ...lidContacts]) {
          contactStore.set(contact.id, contact);
        }

        for (const msg of messages) {
          if (msg.key?.id) {
            messageStore.set(msg.key.id, msg);
            const chatId = msg.key.remoteJid;
            if (!chatStore.has(chatId)) {
              chatStore.set(chatId, []);
            }
            const chatMessages = chatStore.get(chatId);
            chatMessages.push(msg);
          }
        }

        if (isLatest) {
          logger.info("Bulk history sync complete");
          isFullySynced = true;
          saveAll();
        }
      }
    );
    // Replace the messages.upsert event handler with this fixed version

 sock.ev.on("messages.upsert", async ({ messages }) => {
  let newMessagesCount = 0;
  const MAX_MESSAGES_PER_CHAT = 1000;

  for (const message of messages) {
    newMessagesCount++;
    if (message.key?.id) {
      messageStore.set(message.key.id, message);
      
      // Handle LID/PN in message keys
      const chatId = message.key.remoteJidAlt || message.key.remoteJid;
      const participant = message.key.participantAlt || message.key.participant;

      if (!chatStore.has(chatId)) {
        chatStore.set(chatId, []);
      }
      const chatMessages = chatStore.get(chatId);
      chatMessages.push(message);

      // Prevent memory exhaustion
      if (chatMessages.length > MAX_MESSAGES_PER_CHAT) {
        const oldMsg = chatMessages.shift();
        if (oldMsg.key?.id) {
          messageStore.delete(oldMsg.key.id);
        }
      }

      // Audio transcription with Whisper
      if (message.message?.audioMessage && transcriptionEnabled) {
        try {
          console.log("Transcribing audio with Whisper...");
          const audioBuffer = await downloadMediaMessage(message, "buffer");

          const maxSize = 10 * 1024 * 1024;
          if (audioBuffer.length > maxSize) {
            message.transcription = "[Audio troppo lungo per la trascrizione]";
          } else {
            const transcription = await transcribeAudioWithWhisper(audioBuffer, 'auto');
            message.transcription = transcription;
          }
        } catch (error) {
          console.error("Whisper transcription failed:", error);
          message.transcription = "[Trascrizione fallita]";
        }
      }
    }
  }

  if (newMessagesCount > 0) {
    saveMessages();
    saveChats();
  }
});

 sock.ev.on("contacts.set", ({ contacts }) => {
  logger.info(`Contacts set: ${contacts.length}`);

  // First, process all JID contacts (non-LID)
  const jidContacts = contacts.filter(c => !c.id.startsWith('lid:'));
  const lidContacts = contacts.filter(c => c.id.startsWith('lid:'));

  logger.info(`Processing ${jidContacts.length} JID contacts first, then ${lidContacts.length} LID contacts`);

  // Process JID contacts first
  for (const c of jidContacts) {
    // Transform to new structure if needed
    const contact = {
      id: c.id,
      name: c.name,
      notify: c.notify,
      verifiedName: c.verifiedName,
      phoneNumber: c.phoneNumber,
      lid: c.lid
    };

    // Store with both original ID and formatted JID as keys
    contactStore.set(c.id, contact);

    // Also store with formatted JID if different
    const formattedJid = formatJid(c.id);
    if (formattedJid !== c.id) {
      contactStore.set(formattedJid, contact);
    }

    // Store with phone number if available
    if (c.phoneNumber) {
      contactStore.set(c.phoneNumber, contact);
    }
  }

  // Then process LID contacts
  for (const c of lidContacts) {
    // Transform to new structure if needed
    const contact = {
      id: c.id,
      name: c.name,
      notify: c.notify,
      verifiedName: c.verifiedName,
      phoneNumber: c.phoneNumber,
      lid: c.lid
    };

    // Store with both original ID and formatted JID as keys
    contactStore.set(c.id, contact);

    // Also store with formatted JID if different
    const formattedJid = formatJid(c.id);
    if (formattedJid !== c.id) {
      contactStore.set(formattedJid, contact);
    }

    // Store with phone number if available
    if (c.phoneNumber) {
      contactStore.set(c.phoneNumber, contact);
    }
  }

  saveContacts();
});


 sock.ev.on("contacts.update", (contacts) => {
  for (const c of contacts) {
    if (c.id) {
      const existing = contactStore.get(c.id) || {};
      const updated = {
        ...existing,
        ...c,
        id: c.id
      };
      contactStore.set(c.id, updated);
    }
  }
});

// Add LID mapping event handler
sock.ev.on('lid-mapping.update', (update) => {
  console.log('New LID/PN mapping:', update);
  
  // Update your contact store with this mapping
  if (update.lid && update.pn) {
    const existingContact = contactStore.get(update.lid);
    if (existingContact) {
      contactStore.set(update.lid, {
        ...existingContact,
        phoneNumber: update.pn
      });
    }
  }
});

    sock.ev.on("chats.set", ({ chats }) => {
      logger.info(`Chats set: ${chats.length}`);

      // Separate JID and LID chats
      const jidChats = chats.filter(c => !c.id.startsWith('lid:'));
      const lidChats = chats.filter(c => c.id.startsWith('lid:'));

      logger.info(`Processing ${jidChats.length} JID chats first, then ${lidChats.length} LID chats`);

      // Process JID chats first
      for (const c of jidChats) {
        if (!chatStore.has(c.id)) {
          chatStore.set(c.id, []);
        }
      }

      // Then process LID chats
      for (const c of lidChats) {
        if (!chatStore.has(c.id)) {
          chatStore.set(c.id, []);
        }
      }
    });

    sock.ev.on("chats.update", (chats) => {
      for (const c of chats) {
        if (!chatStore.has(c.id)) {
          chatStore.set(c.id, []);
        }
      }
    });
  } catch (error) {
    logger.error("Connection error:", error);
    isConnecting = false; // Reset flag before retry
    setTimeout(connectWithBetterSync, 10000);
  } finally {
    // Reset connection flag when done (successful connection or error)
    // Note: For successful connections, the flag stays true until connection closes
    if (connectionState !== 'open' && connectionState !== 'connecting') {
      isConnecting = false;
    }
  }
}



async function getContactWithLidSupport(jid, sock) {
  // Check if it's a LID
  if (jid.startsWith('lid:')) {
    const contact = contactStore.get(jid);
    if (contact && contact.phoneNumber) {
      return {
        ...contact,
        displayNumber: contact.phoneNumber
      };
    }
    // Try to get PN from LID mapping
    try {
      const pn = await sock.signalRepository.lidMapping.getPNForLID(jid);
      if (pn) {
        return {
          ...contact,
          phoneNumber: pn,
          displayNumber: pn
        };
      }
    } catch (e) {
      // Silently fail
    }
    return {
      ...contact,
      displayNumber: `LID:${jid.substring(4)}`
    };
  }
  
  // Regular phone number
  return {
    ...contactStore.get(jid),
    displayNumber: jid.replace('@s.whatsapp.net', '')
  };
}

// NOTE: connectWithBetterSync() is now called inside worker #1 only (see cluster worker section)
// This prevents multiple QR codes from being generated by different worker processes

// Keep all existing API endpoints from the original code...
// [Include all /api/ routes here]

// Graceful shutdown
const gracefulShutdown = async (signal) => {
  logger.info(`Received ${signal}. Shutting down gracefully...`);
  try {
    // ADD THESE LINES:
    logger.info("Saving all data before shutdown...");
    await storage.saveImmediately("contacts", contactStore);
    await storage.saveImmediately("chats", chatStore);
    await storage.saveImmediately("messages", messageStore);
    await storage.saveImmediately("meta", {
      isFullySynced,
      syncAttempts,
      lastSync: new Date().toISOString(),
    });
    logger.info("Data saved successfully");

    if (typeof sock !== "undefined" && sock) {
      logger.info("Closing WhatsApp connection...");
      await sock.end();
      logger.info("WhatsApp connection closed");
    } else {
      logger.info("No WhatsApp connection to close");
    }

    

    logger.info("Graceful shutdown completed");
    process.exit(0);
  } catch (error) {
    logger.error("Error during shutdown:", error);
    process.exit(1);
  }
};

// Signal handlers
process.on("SIGINT", () => gracefulShutdown("SIGINT"));
process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
process.on("SIGUSR2", () => gracefulShutdown("SIGUSR2"));

process.on("uncaughtException", (error) => {
  logger.error("Uncaught Exception:", error);
  gracefulShutdown("uncaughtException");
});

process.on("unhandledRejection", (reason, promise) => {
  logger.error("Unhandled Rejection at:", promise, "reason:", reason);
});

// ============ PRODUCTION-GRADE HEALTH CHECKS & MONITORING ============

// Health check endpoint for load balancers
app.get("/health", (req, res) => {
  const health = {
    status: "healthy",
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    whatsapp: {
      connected: !!sock?.authState?.creds,
      state: connectionState,
      synced: isFullySynced
    }
  };

  const statusCode = health.whatsapp.connected ? 200 : 503;
  res.status(statusCode).json(health);
});

// Readiness check for Kubernetes/Docker
app.get("/ready", (req, res) => {
  const ready = !!sock?.authState?.creds && isFullySynced;
  res.status(ready ? 200 : 503).json({
    ready,
    whatsapp: {
      connected: !!sock?.authState?.creds,
      synced: isFullySynced
    }
  });
});

// Liveness check
app.get("/live", (req, res) => {
  res.status(200).json({
    alive: true,
    pid: process.pid,
    worker: cluster.worker?.id || 'master'
  });
});

// Advanced metrics endpoint for monitoring
app.get("/api/metrics", (req, res) => {
  const metrics = {
    server: performanceMetrics.getStats(),
    caches: {
      groups: groupsCache.getStats(),
      contacts: contactsCache.getStats(),
      chats: chatsCache.getStats(),
      messages: messagesCache.getStats()
    },
    cluster: {
      isMaster: cluster.isMaster || cluster.isPrimary,
      isWorker: cluster.isWorker,
      workerId: cluster.worker?.id || null,
      pid: process.pid
    },
    whatsapp: {
      connected: !!sock?.authState?.creds,
      state: connectionState,
      synced: isFullySynced,
      syncAttempts: syncAttempts
    },
    stores: {
      contacts: contactStore.size,
      chats: chatStore.size,
      messages: messageStore.size
    }
  };

  res.json(metrics);
});

// ============ SMART WORKER CLUSTERING ============
// Workers are only needed for CPU-intensive tasks (TTS, media conversion, etc.)
// For simple HTTP requests, 2 workers is optimal for Raspberry Pi 4
//
// WHEN TO INCREASE WORKERS:
// - Add TTS/speech-to-speech processing → 3-4 workers
// - Add image/video conversion → 3-4 workers
// - Add AI/ML processing → 4+ workers
// - High HTTP traffic (>1000 req/min) → 3-4 workers
//
// DEFAULT: 2 workers (Worker #1 = WhatsApp + tasks, Worker #2 = HTTP backup)
if (cluster.isMaster || cluster.isPrimary) {
  const numCPUs = os.cpus().length;

  // Use 2 workers by default (optimal for current workload)
  // Override with WEB_CONCURRENCY env var for heavy tasks
  const workers = process.env.WEB_CONCURRENCY || 2;

  logger.info(`🚀 Master process ${process.pid} starting`);
  logger.info(`💪 Spawning ${workers} worker processes (${numCPUs} CPUs available)`);
  logger.info(`ℹ️  Using ${workers} workers - increase WEB_CONCURRENCY env var for heavy tasks (TTS, media conversion)`);

  // Fork workers
  for (let i = 0; i < workers; i++) {
    cluster.fork();
  }

  // Handle worker crashes - auto-restart for high availability
  cluster.on('exit', (worker, code, signal) => {
    logger.warn(`⚠️  Worker ${worker.process.pid} died (${signal || code}). Restarting...`);
    cluster.fork();
  });

  // Log worker status
  cluster.on('online', (worker) => {
    logger.info(`✓ Worker ${worker.process.pid} is online`);
  });

} else {
  // Worker process - handle actual requests
  // Start server
  const server = app.listen(port, () => {
    logger.info(`🔥 Worker ${process.pid} - WhatsApp WML Gateway started on port ${port}`);
    logger.info(`Environment: ${process.env.NODE_ENV || "development"}`);
    logger.info("WML endpoints available at /wml/");
    logger.info("API endpoints available at /api/");

    // Only one worker handles periodic tasks (master-like behavior)
    if (cluster.worker.id === 1) {
      // ============ WHATSAPP CONNECTION (WORKER #1 ONLY) ============
      // Only worker #1 connects to WhatsApp to prevent multiple QR codes
      logger.info(`🔌 Worker #1 - Initiating WhatsApp connection...`);
      connectWithBetterSync();

      // Memory-based cleanup (only when memory usage is high)
      setInterval(() => {
        const memUsage = process.memoryUsage();
        const heapUsedMB = memUsage.heapUsed / 1024 / 1024;
        const heapTotalMB = memUsage.heapTotal / 1024 / 1024;
        const heapPercent = (heapUsedMB / heapTotalMB) * 100;

        logger.info(`Memory usage: ${heapUsedMB.toFixed(2)}MB / ${heapTotalMB.toFixed(2)}MB (${heapPercent.toFixed(1)}%)`);

        // Only cleanup if memory usage exceeds 85% (4GB RAM can handle more)
        if (heapPercent > 85) {
          logger.warn(`High memory usage detected (${heapPercent.toFixed(1)}%), running cleanup...`);
          storage.cleanupOldMessages(messageStore, chatStore, 500); // Keep 500 messages per chat (4GB RAM)

          // Force garbage collection if available
          if (global.gc) {
            global.gc();
            logger.info('Garbage collection forced');
          }
        }
      }, 10 * 60 * 1000); // Check every 10 minutes (less frequent for 4GB system)

      // Periodic save - increased interval to reduce Raspberry Pi I/O load
      setInterval(() => {
        saveAll();
        logger.info("Periodic save completed");
      }, 15 * 60 * 1000); // Every 15 minutes instead of 10
    } else {
      // Other workers don't connect to WhatsApp, they just handle HTTP requests
      logger.info(`📡 Worker #${cluster.worker.id} - Handling HTTP requests only (WhatsApp handled by Worker #1)`);
      logger.info(`💡 Worker #${cluster.worker.id} - Ready for heavy tasks (TTS, media conversion, AI processing)`);
    }
  });

  server.on("error", (error) => {
    if (error.code === "EADDRINUSE") {
      logger.error(`Port ${port} is already in use`);
      process.exit(1);
    } else {
      logger.error("Server error:", error);
      process.exit(1);
    }
  });

  // ============ PRODUCTION-GRADE GRACEFUL SHUTDOWN ============
  // Handles SIGTERM and SIGINT for zero-downtime deployments
  let isShuttingDown = false;

  const gracefulShutdown = (signal) => {
    if (isShuttingDown) return;
    isShuttingDown = true;

    logger.info(`\n⚠️  ${signal} received - initiating graceful shutdown`);

    // Stop accepting new connections
    server.close(() => {
      logger.info("✓ HTTP server closed - no longer accepting connections");

      // Close WhatsApp connection
      if (sock) {
        logger.info("Closing WhatsApp connection...");
        sock.end();
      }

      // Save all data
      logger.info("Saving all data...");
      try {
        saveAll();
        logger.info("✓ All data saved successfully");
      } catch (error) {
        logger.error("Error saving data during shutdown:", error);
      }

      // Close HTTP agent connections
      if (axiosAgent) {
        axiosAgent.destroy();
        logger.info("✓ HTTP connections closed");
      }

      logger.info("✅ Graceful shutdown complete");
      process.exit(0);
    });

    // Force shutdown after 30 seconds
    setTimeout(() => {
      logger.error("❌ Forced shutdown - graceful shutdown timeout exceeded");
      process.exit(1);
    }, 30000);
  };

  // Handle shutdown signals
  process.on("SIGTERM", () => gracefulShutdown("SIGTERM"));
  process.on("SIGINT", () => gracefulShutdown("SIGINT"));

  // Set server timeout for long-running requests
  server.timeout = 120000; // 2 minutes
  server.keepAliveTimeout = 65000; // 65 seconds (must be > load balancer timeout)
  server.headersTimeout = 66000; // Slightly more than keepAliveTimeout
}

// Initialize connection
app.get("/api/status", (req, res) => {
  const isConnected = !!sock?.authState?.creds;

  res.json({
    connected: isConnected,
    status: connectionState,
    user: sock?.user || null,
    qrAvailable: !!currentQR,
    syncStatus: {
      isFullySynced,
      syncAttempts,
      contactsCount: contactStore.size,
      chatsCount: chatStore.size,
      messagesCount: messageStore.size,
    },
    uptime: process.uptime(),
    recommendations: getRecommendations(isConnected),
  });
});

app.get("/api/status-detailed", async (req, res) => {
  try {
    const isConnected = !!sock?.authState?.creds;
    let syncStatus = {
      contacts: contactStore.size,
      chats: chatStore.size,
      messages: messageStore.size,
      isFullySynced,
      syncAttempts,
    };

    res.json({
      connected: isConnected,
      status: connectionState,
      user: sock?.user || null,
      qrAvailable: !!currentQR,
      syncStatus,
      stores: {
        contactStore: {
          size: contactStore.size,
          sample: Array.from(contactStore.entries())
            .slice(0, 3)
            .map(([key, value]) => ({
              key,
              name: value.name || value.notify || "Unknown",
              hasName: !!value.name,
            })),
        },
        chatStore: {
          size: chatStore.size,
          sample: Array.from(chatStore.keys()).slice(0, 5),
        },
      },
      recommendations: getRecommendations(isConnected),
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Performance monitoring endpoint - cluster and system metrics
app.get("/api/performance", (req, res) => {
  const memUsage = process.memoryUsage();
  const cpuUsage = process.cpuUsage();

  res.json({
    cluster: {
      isMaster: cluster.isMaster || cluster.isPrimary,
      isWorker: cluster.isWorker,
      workerId: cluster.worker?.id || null,
      workerPid: process.pid,
      totalWorkers: cluster.isMaster ? Object.keys(cluster.workers || {}).length : null,
    },
    system: {
      cpus: os.cpus().length,
      platform: os.platform(),
      arch: os.arch(),
      totalMemory: `${(os.totalmem() / 1024 / 1024 / 1024).toFixed(2)} GB`,
      freeMemory: `${(os.freemem() / 1024 / 1024 / 1024).toFixed(2)} GB`,
      loadAverage: os.loadavg(),
    },
    process: {
      uptime: `${Math.floor(process.uptime())} seconds`,
      memory: {
        rss: `${(memUsage.rss / 1024 / 1024).toFixed(2)} MB`,
        heapTotal: `${(memUsage.heapTotal / 1024 / 1024).toFixed(2)} MB`,
        heapUsed: `${(memUsage.heapUsed / 1024 / 1024).toFixed(2)} MB`,
        external: `${(memUsage.external / 1024 / 1024).toFixed(2)} MB`,
      },
      cpu: {
        user: `${(cpuUsage.user / 1000).toFixed(2)} ms`,
        system: `${(cpuUsage.system / 1000).toFixed(2)} ms`,
      },
    },
    optimization: {
      clustering: cluster.isMaster || cluster.isWorker,
      nonBlocking: true,
      asyncIO: true,
      multiCore: true,
    },
    stores: {
      contacts: contactStore.size,
      chats: chatStore.size,
      messages: messageStore.size,
    },
  });
});

function getRecommendations(isConnected) {
  if (!isConnected) {
    return ["Please connect to WhatsApp first", "Check QR code if available"];
  }

  if (!isFullySynced && contactStore.size === 0 && chatStore.size === 0) {
    return [
      "Try calling POST /api/full-sync to force data loading",
      "Wait a few more seconds for WhatsApp to sync",
      "Send a test message to trigger data loading",
    ];
  }

  if (contactStore.size === 0) {
    return ["Call POST /api/force-sync-contacts to load contacts"];
  }

  if (chatStore.size === 0) {
    return ["Call POST /api/force-sync-chats to load chats"];
  }

  return ["All systems operational"];
}

// Force sync endpoints
app.post("/api/full-sync", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    console.log("🔄 Starting full manual sync...");
    const results = {
      contacts: 0,
      chats: 0,
      recentChats: 0,
      errors: [],
    };

    // Sync contacts
    try {
      console.log("📞 Attempting contact sync...");

      // In Baileys, contacts are populated automatically via events
      // We can't manually fetch them, so we wait for the events
      if (contactStore.size === 0) {
        console.log("📞 Waiting for contacts to sync via events...");
        await delay(3000); // Wait for events to populate
      }

      results.contacts = contactStore.size;
      console.log(`📞 Contacts available: ${contactStore.size}`);
    } catch (error) {
      results.errors.push(`Contacts sync info: ${error.message}`);
    }

    // Sync chats
    try {
      const chats = await sock.groupFetchAllParticipating();
      Object.keys(chats).forEach((chatId) => {
        if (!chatStore.has(chatId)) {
          chatStore.set(chatId, []);
        }
      });
      results.chats = Object.keys(chats).length;
      console.log(`💬 Manually synced ${Object.keys(chats).length} chats`);
    } catch (error) {
      results.errors.push(`Chats sync failed: ${error.message}`);
    }

    // Sync recent chats
    try {
      console.log("💬 Checking for additional chats...");

      // In Baileys, we don't have fetchChats, but we have what we got from groupFetchAllParticipating
      // Let's wait a bit more for any chat events
      await delay(2000);

      results.recentChats = chatStore.size - results.chats;
      console.log(`💬 Additional chats found: ${results.recentChats}`);
    } catch (error) {
      results.errors.push(`Additional chats check failed: ${error.message}`);
    }

    // Update sync status
    if (contactStore.size > 0 || chatStore.size > 0) {
      isFullySynced = true;
    }

    res.json({
      status: "completed",
      results,
      currentStore: {
        contacts: contactStore.size,
        chats: chatStore.size,
        messages: messageStore.size,
      },
      isFullySynced,
    });
  } catch (error) {
    console.error("❌ Full sync failed:", error);
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/force-sync-contacts", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    console.log("🔄 Checking contact sync status...");

    // In Baileys, contacts are synced via events, not direct API calls
    // We can only report what we have and potentially trigger a refresh
    const initialCount = contactStore.size;

    // Wait a bit to see if more contacts come in
    console.log("📞 Waiting for contact events...");
    await delay(3000);

    const finalCount = contactStore.size;
    const newContacts = finalCount - initialCount;

    console.log(
      `✅ Contact sync check completed. Total: ${finalCount}, New: ${newContacts}`
    );

    res.json({
      status: "success",
      message: "Contacts are synced via WhatsApp events",
      initialCount,
      finalCount,
      newContacts,
      totalInStore: contactStore.size,
    });
  } catch (error) {
    console.error("❌ Contact sync check failed:", error);
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/force-sync-chats", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    console.log("🔄 Forcing chat sync...");

    const initialChatCount = chatStore.size;

    // Get participating groups (this works)
    const chats = await sock.groupFetchAllParticipating();

    Object.keys(chats).forEach((chatId) => {
      if (!chatStore.has(chatId)) {
        chatStore.set(chatId, []);
      }
    });

    // Wait for any additional chat events
    console.log("💬 Waiting for additional chat events...");
    await delay(3000);

    const finalChatCount = chatStore.size;
    const newChats = finalChatCount - initialChatCount;

    console.log(
      `✅ Chat sync completed. Groups: ${
        Object.keys(chats).length
      }, Total: ${finalChatCount}`
    );

    res.json({
      status: "success",
      groupChats: Object.keys(chats).length,
      initialTotal: initialChatCount,
      finalTotal: finalChatCount,
      newChats: newChats,
      totalInStore: chatStore.size,
    });
  } catch (error) {
    console.error("❌ Force sync chats failed:", error);
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/debug-stores", (req, res) => {
  res.json({
    connectionState,
    isFullySynced,
    syncAttempts,
    contactStore: {
      size: contactStore.size,
      sample: Array.from(contactStore.entries())
        .slice(0, 5)
        .map(([key, value]) => ({
          key,
          name: value.name || value.notify || "Unknown",
          hasName: !!value.name,
          notify: value.notify,
          verifiedName: value.verifiedName,
        })),
    },
    chatStore: {
      size: chatStore.size,
      chats: Array.from(chatStore.keys()).slice(0, 10),
    },
    messageStore: {
      size: messageStore.size,
      sample: Array.from(messageStore.keys()).slice(0, 5),
    },
  });
});

// =================== QR CODE ENDPOINTS ===================

app.get("/api/qr", (req, res) => {
  if (currentQR) {
    res.send(`
            <html><body style="text-align:center;padding:50px;font-family:Arial;">
                <h2>📱 WhatsApp QR Code</h2>
                <div style="background:white;padding:20px;border-radius:10px;display:inline-block;">
                    <img src="data:image/png;base64,${Buffer.from(
                      currentQR
                    ).toString(
                      "base64"
                    )}" style="border:10px solid #25D366;border-radius:10px;"/>
                </div>
                <p>Scan with WhatsApp app</p>
                <p><small>Auto-refresh in 10 seconds</small></p>
                <script>setTimeout(() => location.reload(), 10000);</script>
            </body></html>
        `);
  } else {
    res.json({
      message: "QR not available",
      connected: !!sock?.authState?.creds,
      status: connectionState,
    });
  }
});

app.get("/api/qr/image", async (req, res) => {
  const { format = "png" } = req.query;

  if (!currentQR) {
    // Return a placeholder image instead of JSON to prevent "unknown response" errors
    const placeholderText = `No QR Code\nStatus: ${connectionState}\nPlease wait...`;

    try {
      // Generate a simple text-based QR placeholder
      const qrBuffer = await QRCode.toBuffer(placeholderText, {
        type: "png",
        width: 256,
        margin: 2,
        color: {
          dark: "#666666",
          light: "#FFFFFF",
        },
      });

      res.setHeader("Content-Type", format === "wbmp" ? "image/vnd.wap.wbmp" : "image/png");
      res.setHeader("Cache-Control", "no-cache");
      return res.send(qrBuffer);
    } catch (err) {
      // Ultimate fallback: 1x1 transparent PNG
      const transparentPng = Buffer.from(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "base64"
      );
      res.setHeader("Content-Type", "image/png");
      return res.send(transparentPng);
    }
  }

  try {
    if (format.toLowerCase() === "wbmp") {
      // Generate QR as WBMP format using qrcode library
      try {
        const qrBuffer = await QRCode.toBuffer(currentQR, {
          type: "png",
          width: 256,
          margin: 2,
          color: {
            dark: "#000000",
            light: "#FFFFFF",
          },
        });

        // Convert PNG to simple WBMP-like format
        // WBMP is a monochrome format, so we'll return the QR as minimal binary
        res.setHeader("Content-Type", "image/vnd.wap.wbmp");
        res.setHeader("Content-Disposition", 'inline; filename="qr-code.wbmp"');
        res.setHeader("Cache-Control", "no-cache");

        // Return the buffer (simplified WBMP representation)
        res.send(qrBuffer);
      } catch (qrError) {
        // Fallback: return raw QR string as WBMP
        res.setHeader("Content-Type", "image/vnd.wap.wbmp");
        res.setHeader("Content-Disposition", 'inline; filename="qr-code.wbmp"');
        const qrBuffer = Buffer.from(currentQR, "utf8");
        res.send(qrBuffer);
      }
    } else if (format.toLowerCase() === "base64") {
      // Return as base64 JSON response
      res.json({
        qrCode: currentQR,
        format: "base64",
        timestamp: Date.now(),
        dataUrl: `data:text/plain;base64,${Buffer.from(currentQR).toString(
          "base64"
        )}`,
      });
    } else if (format.toLowerCase() === "png") {
      // Generate proper PNG QR code
      try {
        const qrBuffer = await QRCode.toBuffer(currentQR, {
          type: "png",
          width: 256,
          margin: 2,
          color: {
            dark: "#000000",
            light: "#FFFFFF",
          },
        });

        res.setHeader("Content-Type", "image/png");
        res.setHeader("Content-Disposition", 'inline; filename="qr-code.png"');
        res.setHeader("Cache-Control", "no-cache");
        res.send(qrBuffer);
      } catch (qrError) {
        // Fallback to base64 if available
        res.setHeader("Content-Type", "image/png");
        res.send(Buffer.from(currentQR, "base64"));
      }
    } else if (format.toLowerCase() === "svg") {
      // Generate SVG QR code
      try {
        const qrSvg = await QRCode.toString(currentQR, {
          type: "svg",
          width: 256,
          margin: 2,
          color: {
            dark: "#000000",
            light: "#FFFFFF",
          },
        });

        res.setHeader("Content-Type", "image/svg+xml");
        res.setHeader("Content-Disposition", 'inline; filename="qr-code.svg"');
        res.setHeader("Cache-Control", "no-cache");
        res.send(qrSvg);
      } catch (qrError) {
        res.status(500).json({ error: "Failed to generate SVG QR code" });
      }
    } else {
      res.status(400).json({
        error: "Unsupported format",
        supportedFormats: ["png", "svg", "base64", "wbmp"],
        examples: [
          "GET /api/qr/image?format=png",
          "GET /api/qr/image?format=svg",
          "GET /api/qr/image?format=wbmp",
          "GET /api/qr/image?format=base64",
        ],
      });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/qr/text", (req, res) => {
  if (!currentQR) {
    res.set("Content-Type", "text/vnd.wap.wml");
    return res.send(`<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN"
  "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <card id="noqr" title="QR Not Available">
    <p>QR code not available</p>
  </card>
</wml>`);
  }

  res.set("Content-Type", "text/vnd.wap.wml");
  res.send(`<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN"
  "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <card id="qr" title="WhatsApp QR">
    <p>Your QR string:</p>
    <p>${currentQR}</p>
  </card>
</wml>`);
});

app.post("/api/logout", async (req, res) => {
  try {
    if (sock) await sock.logout();
    if (fs.existsSync("./auth_info_baileys")) {
      fs.rmSync("./auth_info_baileys", { recursive: true });
    }

    // Clear stores
    contactStore.clear();
    chatStore.clear();
    messageStore.clear();
    isFullySynced = false;
    syncAttempts = 0;

    res.json({ status: "Logged out and data cleared" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/me", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const profilePic = await sock
      .profilePictureUrl(sock.user.id)
      .catch(() => null);
    const status = await sock.fetchStatus(sock.user.id).catch(() => null);

    res.json({
      user: sock.user,
      profilePicture: profilePic,
      status: status?.status,
      syncStatus: {
        isFullySynced,
        contactsCount: contactStore.size,
        chatsCount: chatStore.size,
      },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/update-profile-name", async (req, res) => {
  try {
    const { name } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    await sock.updateProfileName(name);
    res.json({ status: "Profile name updated" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/update-profile-status", async (req, res) => {
  try {
    const { status } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    await sock.updateProfileStatus(status);
    res.json({ status: "Profile status updated" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/update-profile-picture", async (req, res) => {
  try {
    const { imageUrl } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const response = await axios.get(imageUrl, { responseType: "arraybuffer" });
    await sock.updateProfilePicture(sock.user.id, response.data);
    res.json({ status: "Profile picture updated" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/presence", async (req, res) => {
  try {
    const { jid, presence = "available" } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    if (jid) {
      await sock.sendPresenceUpdate(presence, formatJid(jid));
    } else {
      await sock.sendPresenceUpdate(presence);
    }
    res.json({ status: `Presence set to ${presence}` });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// =================== ENHANCED CONTACTS ENDPOINTS ===================

app.get("/api/contacts/all", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    // Auto-sync if no contacts and not synced yet
    if (contactStore.size === 0 && !isFullySynced) {
      console.log("📞 No contacts found, waiting for sync events...");
      // In Baileys, contacts come via events, so we just wait
      await delay(2000);
      console.log(`📞 Contacts after wait: ${contactStore.size}`);
    }

    const { page = 1, limit = 100, enriched = false } = req.query;
    let contacts = Array.from(contactStore.values());

    // Sort contacts by last message timestamp (most recent first)
    contacts.sort((a, b) => {
      const messagesA = chatStore.get(a.id) || [];
      const messagesB = chatStore.get(b.id) || [];

      const lastMessageA = messagesA.length > 0 ? messagesA[messagesA.length - 1] : null;
      const lastMessageB = messagesB.length > 0 ? messagesB[messagesB.length - 1] : null;

      const timestampA = lastMessageA ? Number(lastMessageA.messageTimestamp) : 0;
      const timestampB = lastMessageB ? Number(lastMessageB.messageTimestamp) : 0;

      return timestampB - timestampA; // Most recent first
    });

    // Pagination
    const startIndex = (parseInt(page) - 1) * parseInt(limit);
    const endIndex = startIndex + parseInt(limit);
    const paginatedContacts = contacts.slice(startIndex, endIndex);

    if (enriched === "true") {
      const enrichedContacts = [];

      for (const contact of paginatedContacts) {
        try {
          const profilePic = await sock
            .profilePictureUrl(contact.id, "image")
            .catch(() => null);
          const status = await sock.fetchStatus(contact.id).catch(() => null);
          const businessProfile = await sock
            .getBusinessProfile(contact.id)
            .catch(() => null);

          enrichedContacts.push({
            id: contact.id,
            name: contact.name || contact.notify || contact.verifiedName,
            profilePicture: profilePic,
            status: status?.status,
            lastSeen: status?.setAt,
            isMyContact: contact.name ? true : false,
            isBusiness: !!businessProfile,
            businessProfile: businessProfile,
            notify: contact.notify,
            verifiedName: contact.verifiedName,
          });

          await delay(150);
        } catch (error) {
          enrichedContacts.push({
            id: contact.id,
            name: contact.name || contact.notify || contact.verifiedName,
            error: error.message,
          });
        }
      }

      res.json({
        contacts: enrichedContacts,
        pagination: {
          page: parseInt(page),
          limit: parseInt(limit),
          total: contacts.length,
          totalPages: Math.ceil(contacts.length / parseInt(limit)),
          hasNext: endIndex < contacts.length,
          hasPrev: parseInt(page) > 1,
        },
        syncInfo: { isFullySynced, syncAttempts },
      });
    } else {
      const basicContacts = paginatedContacts.map((contact) => ({
        id: contact.id,
        name: contact.name || contact.notify || contact.verifiedName,
        notify: contact.notify,
        verifiedName: contact.verifiedName,
        isMyContact: contact.name ? true : false,
      }));

      res.json({
        contacts: basicContacts,
        pagination: {
          page: parseInt(page),
          limit: parseInt(limit),
          total: contacts.length,
          totalPages: Math.ceil(contacts.length / parseInt(limit)),
          hasNext: endIndex < contacts.length,
          hasPrev: parseInt(page) > 1,
        },
        syncInfo: { isFullySynced, syncAttempts },
      });
    }
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/contacts/count", (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    res.json({
      totalContacts: contactStore.size,
      withNames: Array.from(contactStore.values()).filter((c) => c.name).length,
      businessContacts: Array.from(contactStore.values()).filter(
        (c) => c.verifiedName
      ).length,
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/contacts/search", async (req, res) => {
  try {
    const { query, limit = 50 } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });
    if (!query || query.length < 2) {
      return res
        .status(400)
        .json({ error: "Query must be at least 2 characters" });
    }

    const searchQuery = query.toLowerCase();
    let contacts = Array.from(contactStore.values());

    // Sort contacts by last message timestamp (most recent first)
    contacts.sort((a, b) => {
      const messagesA = chatStore.get(a.id) || [];
      const messagesB = chatStore.get(b.id) || [];

      const lastMessageA = messagesA.length > 0 ? messagesA[messagesA.length - 1] : null;
      const lastMessageB = messagesB.length > 0 ? messagesB[messagesB.length - 1] : null;

      const timestampA = lastMessageA ? Number(lastMessageA.messageTimestamp) : 0;
      const timestampB = lastMessageB ? Number(lastMessageB.messageTimestamp) : 0;

      return timestampB - timestampA; // Most recent first
    });

    const results = contacts
      .filter((contact) => {
        const name = (
          contact.name ||
          contact.notify ||
          contact.verifiedName ||
          ""
        ).toLowerCase();
        const number = contact.id.replace("@s.whatsapp.net", "");

        return name.includes(searchQuery) || number.includes(searchQuery);
      })
      .slice(0, parseInt(limit));

    res.json({
      query: query,
      results: results.map((contact) => ({
        id: contact.id,
        name: contact.name || contact.notify || contact.verifiedName,
        notify: contact.notify,
        verifiedName: contact.verifiedName,
        number: contact.id.replace("@s.whatsapp.net", ""),
        isMyContact: contact.name ? true : false,
      })),
      total: results.length,
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// =================== ENHANCED CHAT ENDPOINTS ===================

app.get("/api/chats/with-numbers", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    // Auto-sync if no chats and not synced yet
    if (chatStore.size === 0 && !isFullySynced) {
      console.log("💬 No chats found, attempting auto-sync...");
      try {
        const chats = await sock.groupFetchAllParticipating();

        Object.keys(chats).forEach((chatId) => {
          if (!chatStore.has(chatId)) {
            chatStore.set(chatId, []);
          }
        });

        console.log(`💬 Auto-synced ${Object.keys(chats).length} group chats`);

        // Wait for additional chat events
        await delay(2000);
        console.log(`💬 Total chats after wait: ${chatStore.size}`);
      } catch (syncError) {
        console.log("⚠️ Auto-sync failed:", syncError.message);
      }
    }

    const chats = Array.from(chatStore.keys()).map((chatId) => {
      const messages = chatStore.get(chatId) || [];
      const lastMessage = messages[messages.length - 1];
      const contact = contactStore.get(chatId);

      const phoneNumber = chatId
        .replace("@s.whatsapp.net", "")
        .replace("@g.us", "");
      const isGroup = chatId.endsWith("@g.us");

      return {
        id: chatId,
        phoneNumber: isGroup ? null : phoneNumber,
        groupId: isGroup ? phoneNumber : null,
        isGroup: isGroup,
        contact: {
          name: contact?.name || contact?.notify || contact?.verifiedName,
          isMyContact: contact?.name ? true : false,
        },
        messageCount: messages.length,
        lastMessage: lastMessage
          ? {
              id: lastMessage.key.id,
              message: extractMessageContent(lastMessage.message),
              timestamp: lastMessage.messageTimestamp,
              fromMe: lastMessage.key.fromMe,
            }
          : null,
      };
    });

    chats.sort((a, b) => {
      const aTime = a.lastMessage?.timestamp || 0;
      const bTime = b.lastMessage?.timestamp || 0;
      return bTime - aTime;
    });

    res.json({
      chats,
      total: chats.length,
      directChats: chats.filter((c) => !c.isGroup).length,
      groupChats: chats.filter((c) => c.isGroup).length,
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/chat/by-number/:number", async (req, res) => {
  try {
    const { number } = req.params;
    const { limit = 50, offset = 0 } = req.query;

    if (!sock) return res.status(500).json({ error: "Not connected" });

    const jid = formatJid(number);
    const messages = chatStore.get(jid) || [];

    const contact = contactStore.get(jid);
    const profilePic = await sock.profilePictureUrl(jid).catch(() => null);
    const status = await sock.fetchStatus(jid).catch(() => null);

    // Pagination
    const startIndex = Math.max(
      0,
      messages.length - parseInt(limit) - parseInt(offset)
    );
    const endIndex = messages.length - parseInt(offset);
    const paginatedMessages = messages.slice(startIndex, endIndex);

    const formattedMessages = paginatedMessages.map((msg) => ({
      id: msg.key.id,
      fromMe: msg.key.fromMe,
      timestamp: msg.messageTimestamp,
      message: extractMessageContent(msg.message),
      messageType: getContentType(msg.message),
      quoted: msg.message?.extendedTextMessage?.contextInfo?.quotedMessage
        ? true
        : false,
    }));

    res.json({
      number: number,
      jid: jid,
      contact: {
        name: contact?.name || contact?.notify || contact?.verifiedName,
        profilePicture: profilePic,
        status: status?.status,
        lastSeen: status?.setAt,
        isMyContact: contact?.name ? true : false,
      },
      chat: {
        messages: formattedMessages,
        total: messages.length,
        showing: formattedMessages.length,
        hasMore: startIndex > 0,
        isGroup: jid.endsWith("@g.us"),
      },
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/chat/exists/:number", (req, res) => {
  try {
    const { number } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const jid = formatJid(number);
    const messages = chatStore.get(jid) || [];
    const contact = contactStore.get(jid);

    res.json({
      number: number,
      jid: jid,
      exists: messages.length > 0,
      messageCount: messages.length,
      hasContact: !!contact,
      contactName: contact?.name || contact?.notify || contact?.verifiedName,
      lastActivity:
        messages.length > 0
          ? messages[messages.length - 1].messageTimestamp
          : null,
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/chat/stats/:number", async (req, res) => {
  try {
    const { number } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const jid = formatJid(number);
    const messages = chatStore.get(jid) || [];
    const contact = contactStore.get(jid);

    if (messages.length === 0) {
      return res.json({
        number: number,
        jid: jid,
        exists: false,
        message: "No chat history found",
        syncInfo: { isFullySynced, syncAttempts },
      });
    }

    // Calculate statistics
    const myMessages = messages.filter((msg) => msg.key.fromMe);
    const theirMessages = messages.filter((msg) => !msg.key.fromMe);
    const mediaMessages = messages.filter((msg) => {
      const type = getContentType(msg.message);
      return [
        "imageMessage",
        "videoMessage",
        "audioMessage",
        "documentMessage",
        "stickerMessage",
      ].includes(type);
    });

    const firstMessage = messages[0];
    const lastMessage = messages[messages.length - 1];

    // Message types breakdown
    const messageTypes = {};
    messages.forEach((msg) => {
      const type = getContentType(msg.message) || "unknown";
      messageTypes[type] = (messageTypes[type] || 0) + 1;
    });

    res.json({
      number: number,
      jid: jid,
      contact: {
        name: contact?.name || contact?.notify || contact?.verifiedName,
        isMyContact: contact?.name ? true : false,
      },
      statistics: {
        totalMessages: messages.length,
        myMessages: myMessages.length,
        theirMessages: theirMessages.length,
        mediaMessages: mediaMessages.length,
        messageTypes: messageTypes,
        firstMessage: {
          timestamp: firstMessage.messageTimestamp,
          fromMe: firstMessage.key.fromMe,
        },
        lastMessage: {
          timestamp: lastMessage.messageTimestamp,
          fromMe: lastMessage.key.fromMe,
        },
        chatDuration:
          lastMessage.messageTimestamp - firstMessage.messageTimestamp,
      },
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/chats/bulk-by-numbers", async (req, res) => {
  try {
    const { numbers, includeMessages = false, messageLimit = 10 } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });
    if (!Array.isArray(numbers)) {
      return res.status(400).json({ error: "Numbers must be an array" });
    }

    const results = [];

    for (const number of numbers) {
      try {
        const jid = formatJid(number);
        const messages = chatStore.get(jid) || [];
        const contact = contactStore.get(jid);

        const result = {
          number: number,
          jid: jid,
          exists: messages.length > 0,
          messageCount: messages.length,
          contact: {
            name: contact?.name || contact?.notify || contact?.verifiedName,
            isMyContact: contact?.name ? true : false,
          },
        };

        if (includeMessages && messages.length > 0) {
          const recentMessages = messages.slice(-parseInt(messageLimit));
          result.recentMessages = recentMessages.map((msg) => ({
            id: msg.key.id,
            fromMe: msg.key.fromMe,
            timestamp: msg.messageTimestamp,
            message: extractMessageContent(msg.message),
            messageType: getContentType(msg.message),
          }));
        }

        results.push(result);
      } catch (error) {
        results.push({
          number: number,
          error: error.message,
        });
      }
    }

    res.json({
      results,
      total: results.length,
      withChats: results.filter((r) => r.exists).length,
      withoutChats: results.filter((r) => !r.exists && !r.error).length,
      errors: results.filter((r) => r.error).length,
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// =================== OTHER ENDPOINTS ===================

app.get("/api/contacts", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    let contacts = Array.from(contactStore.values());

    // Sort contacts by last message timestamp (most recent first)
    contacts.sort((a, b) => {
      const messagesA = chatStore.get(a.id) || [];
      const messagesB = chatStore.get(b.id) || [];

      const lastMessageA = messagesA.length > 0 ? messagesA[messagesA.length - 1] : null;
      const lastMessageB = messagesB.length > 0 ? messagesB[messagesB.length - 1] : null;

      const timestampA = lastMessageA ? Number(lastMessageA.messageTimestamp) : 0;
      const timestampB = lastMessageB ? Number(lastMessageB.messageTimestamp) : 0;

      return timestampB - timestampA; // Most recent first
    });

    const enrichedContacts = [];
    for (const contact of contacts.slice(0, 50)) {
      try {
        const profilePic = await sock
          .profilePictureUrl(contact.id, "image")
          .catch(() => null);
        const status = await sock.fetchStatus(contact.id).catch(() => null);

        enrichedContacts.push({
          id: contact.id,
          name: contact.name || contact.notify || contact.verifiedName,
          profilePicture: profilePic,
          status: status?.status,
          isMyContact: contact.name ? true : false,
          lastSeen: status?.setAt,
        });

        await delay(100);
      } catch (error) {
        enrichedContacts.push({
          id: contact.id,
          name: contact.name || contact.notify || contact.verifiedName,
          error: error.message,
        });
      }
    }

    res.json({
      contacts: enrichedContacts,
      total: contactStore.size,
      showing: enrichedContacts.length,
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/chats", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const chats = Array.from(chatStore.keys()).map((chatId) => {
      const messages = chatStore.get(chatId) || [];
      const lastMessage = messages[messages.length - 1];

      return {
        id: chatId,
        isGroup: chatId.endsWith("@g.us"),
        messageCount: messages.length,
        lastMessage: lastMessage
          ? {
              id: lastMessage.key.id,
              message: extractMessageContent(lastMessage.message),
              timestamp: lastMessage.messageTimestamp,
              fromMe: lastMessage.key.fromMe,
            }
          : null,
      };
    });

    chats.sort((a, b) => {
      const aTime = a.lastMessage?.timestamp || 0;
      const bTime = b.lastMessage?.timestamp || 0;
      return bTime - aTime;
    });

    res.json({
      chats,
      total: chats.length,
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/messages/:jid", async (req, res) => {
  try {
    const { jid } = req.params;
    const { limit = 50, offset = 0 } = req.query;

    if (!sock) return res.status(500).json({ error: "Not connected" });

    const formattedJid = formatJid(jid);
    const messages = chatStore.get(formattedJid) || [];

    const startIndex = Math.max(
      0,
      messages.length - parseInt(limit) - parseInt(offset)
    );
    const endIndex = messages.length - parseInt(offset);
    const paginatedMessages = messages.slice(startIndex, endIndex);

    const formattedMessages = paginatedMessages.map((msg) => ({
      id: msg.key.id,
      fromMe: msg.key.fromMe,
      timestamp: msg.messageTimestamp,
      message: extractMessageContent(msg.message),
      messageType: getContentType(msg.message),
      quoted: msg.message?.extendedTextMessage?.contextInfo?.quotedMessage
        ? true
        : false,
    }));

    res.json({
      jid: formattedJid,
      messages: formattedMessages,
      total: messages.length,
      showing: formattedMessages.length,
      hasMore: startIndex > 0,
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/search-messages", async (req, res) => {
  try {
    const { query, jid, limit = 50 } = req.body;

    if (!sock) return res.status(500).json({ error: "Not connected" });
    if (!query || query.length < 2) {
      return res
        .status(400)
        .json({ error: "Query must be at least 2 characters" });
    }

    const results = [];
    const searchQuery = query.toLowerCase();

    const chatsToSearch = jid ? [formatJid(jid)] : Array.from(chatStore.keys());

    for (const chatId of chatsToSearch) {
      const messages = chatStore.get(chatId) || [];

      for (const msg of messages) {
        const content = extractMessageContent(msg.message);
        const messageText =
          content?.conversation ||
          content?.extendedTextMessage?.text ||
          content?.imageMessage?.caption ||
          content?.videoMessage?.caption ||
          "";

        if (messageText.toLowerCase().includes(searchQuery)) {
          results.push({
            chatId,
            messageId: msg.key.id,
            fromMe: msg.key.fromMe,
            timestamp: msg.messageTimestamp,
            message: messageText,
            messageType: getContentType(msg.message),
          });

          if (results.length >= limit) break;
        }
      }

      if (results.length >= limit) break;
    }

    results.sort((a, b) => b.timestamp - a.timestamp);

    res.json({
      query: query,
      results,
      total: results.length,
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update business profile handling for LIDs
app.get("/api/contact/:jid", async (req, res) => {
  try {
    const { jid } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const formattedJid = formatJid(jid);
    let profilePic, status, businessProfile;

    try {
      profilePic = await sock.profilePictureUrl(formattedJid).catch(() => null);
      status = await sock.fetchStatus(formattedJid).catch(() => null);
      businessProfile = await sock.getBusinessProfile(formattedJid).catch(() => null);
    } catch (e) {
      // Silent fail for optional features
    }

    res.json({
      jid: formattedJid,
      profilePicture: profilePic,
      status: status?.status,
      businessProfile,
      addressingMode: sock.getMessageAddressingMode(formattedJid),
      syncInfo: { isFullySynced, syncAttempts },
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/block/:jid", async (req, res) => {
  try {
    const { jid } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    await sock.updateBlockStatus(formatJid(jid), "block");
    res.json({ status: "Contact blocked" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/unblock/:jid", async (req, res) => {
  try {
    const { jid } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    await sock.updateBlockStatus(formatJid(jid), "unblock");
    res.json({ status: "Contact unblocked" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update check-numbers endpoint
app.post("/api/check-numbers", async (req, res) => {
  try {
    const { numbers } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const results = [];
    for (const number of numbers) {
      try {
        // Clean the number - remove non-digits and plus
        const cleanedNumber = number.replace(/\D/g, '');
        
        // Use cleaned number directly (no domain)
        const exists = await sock.onWhatsApp([cleanedNumber]);
        
        results.push({
          number,
          cleanedNumber,
          exists: exists.length > 0,
          details: exists[0] || null,
        });
      } catch (error) {
        results.push({
          number,
          error: error.message,
        });
      }
      await delay(500);
    }

    res.json({ results });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
// =================== SEND MESSAGE ENDPOINTS ===================

app.post("/api/send-text", async (req, res) => {
  try {
    const { to, message } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const result = await sock.sendMessage(formatJid(to), { text: message });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-image", async (req, res) => {
  try {
    const { to, imageUrl, caption } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const response = await axios.get(imageUrl, { responseType: "arraybuffer" });
    const result = await sock.sendMessage(formatJid(to), {
      image: response.data,
      caption,
    });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-video", async (req, res) => {
  try {
    const { to, videoUrl, caption } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const response = await axios.get(videoUrl, { responseType: "arraybuffer" });
    const result = await sock.sendMessage(formatJid(to), {
      video: response.data,
      caption,
    });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-audio", async (req, res) => {
  try {
    const { to, audioUrl, ptt = false } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const response = await axios.get(audioUrl, { responseType: "arraybuffer" });
    const result = await sock.sendMessage(formatJid(to), {
      audio: response.data,
      ptt,
      mimetype: "audio/mp4",
    });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-document", async (req, res) => {
  try {
    const { to, documentUrl, fileName, mimetype } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const response = await axios.get(documentUrl, {
      responseType: "arraybuffer",
      timeout: 120000, // 120 second timeout for documents
      maxContentLength: 100 * 1024 * 1024, // 100MB max
      maxBodyLength: 100 * 1024 * 1024
    });
    const result = await sock.sendMessage(formatJid(to), {
      document: response.data,
      fileName: fileName || "document",
      mimetype: mimetype || "application/octet-stream",
    });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-sticker", async (req, res) => {
  try {
    const { to, imageUrl } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const response = await axios.get(imageUrl, { responseType: "arraybuffer" });
    const result = await sock.sendMessage(formatJid(to), {
      sticker: response.data,
    });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-location", async (req, res) => {
  try {
    const { to, latitude, longitude, name } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const result = await sock.sendMessage(formatJid(to), {
      location: {
        degreesLatitude: parseFloat(latitude),
        degreesLongitude: parseFloat(longitude),
        name,
      },
    });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-contact", async (req, res) => {
  try {
    const { to, contacts } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const contactList = Array.isArray(contacts) ? contacts : [contacts];
    const vCards = contactList.map((contact) => ({
      displayName: contact.name,
      vcard: `BEGIN:VCARD\nVERSION:3.0\nFN:${contact.name}\nTEL;type=CELL:${contact.number}\nEND:VCARD`,
    }));

    const result = await sock.sendMessage(formatJid(to), {
      contacts: {
        displayName: `${contactList.length} contact${
          contactList.length > 1 ? "s" : ""
        }`,
        contacts: vCards,
      },
    });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-poll", async (req, res) => {
  try {
    const { to, name, values, selectableCount = 1 } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const result = await sock.sendMessage(formatJid(to), {
      poll: {
        name,
        values,
        selectableCount: Math.min(selectableCount, values.length),
      },
    });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-reaction", async (req, res) => {
  try {
    const { to, messageId, emoji } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const targetMessage = messageStore.get(messageId);
    if (!targetMessage) {
      return res.status(404).json({ error: "Message not found" });
    }

    const result = await sock.sendMessage(formatJid(to), {
      react: {
        text: emoji,
        key: targetMessage.key,
      },
    });
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-reply", async (req, res) => {
  try {
    const { to, message, quotedMessageId } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const quotedMessage = messageStore.get(quotedMessageId);
    if (!quotedMessage) {
      return res.status(404).json({ error: "Quoted message not found" });
    }

    const result = await sock.sendMessage(
      formatJid(to),
      { text: message },
      { quoted: quotedMessage }
    );
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/forward-message", async (req, res) => {
  try {
    const { messageId, to } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const targetMessage = messageStore.get(messageId);
    if (!targetMessage) {
      return res.status(404).json({ error: "Message not found" });
    }

    const recipients = Array.isArray(to) ? to : [to];
    const results = [];

    for (const recipient of recipients) {
      try {
        const result = await sock.relayMessage(
          formatJid(recipient),
          targetMessage.message,
          {}
        );
        results.push({
          recipient: formatJid(recipient),
          status: "sent",
          messageId: result.key.id,
        });
      } catch (error) {
        results.push({
          recipient: formatJid(recipient),
          status: "failed",
          error: error.message,
        });
      }
      await delay(1000);
    }

    res.json({ status: "ok", results });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.delete("/api/delete-message", async (req, res) => {
  try {
    const { messageId, to } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const targetMessage = messageStore.get(messageId);
    if (!targetMessage) {
      return res.status(404).json({ error: "Message not found" });
    }

    await sock.sendMessage(formatJid(to), { delete: targetMessage.key });
    res.json({ status: "ok" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

/*
app.post("/api/read-messages", async (req, res) => {
  // ⚠️ CRITICAL BAN PREVENTION (Baileys 7.x)
  // Read receipts (ACKs) are DISABLED to prevent WhatsApp bans
  // DO NOT re-enable without understanding the ban risks
  try {
    return res.status(403).json({
      error: "Read receipts disabled for ban prevention",
      warning: "Sending read receipts can trigger WhatsApp bans in Baileys 7.x",
      suggestion: "Do not re-enable this feature"
    });

    // DISABLED CODE (kept for reference):
    // const { messageIds } = req.body;
    // if (!sock) return res.status(500).json({ error: "Not connected" });
    //
    // const keys = messageIds
    //   .map((id) => {
    //     const msg = messageStore.get(id);
    //     return msg ? msg.key : null;
    //   })
    //   .filter(Boolean);
    //
    // if (keys.length === 0) {
    //   return res.status(404).json({ error: "No valid messages found" });
    // }
    //
    // await sock.readMessages(keys);
    // res.json({ status: "ok", markedAsRead: keys.length });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});*/

// =================== GROUP MANAGEMENT ===================

app.post("/api/group-create", async (req, res) => {
  try {
    const { name, participants } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const participantJids = participants.map((jid) => formatJid(jid));
    const group = await sock.groupCreate(name, participantJids);

    res.json({ status: "ok", groupId: group.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/groups", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const groups = await sock.groupFetchAllParticipating();
    res.json({ groups: Object.values(groups) });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/group/:groupId/metadata", async (req, res) => {
  try {
    const { groupId } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const metadata = await sock.groupMetadata(groupId);
    res.json({ group: metadata });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Update group participant handling to work with LIDs
app.post("/api/group/:groupId/participants", async (req, res) => {
  try {
    const { groupId } = req.params;
    const { participants, action } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    // Participants can now be LIDs or PNs
    const participantJids = participants.map(jid => {
      // If it's already a JID (with domain), use as-is
      if (jid.includes('@')) return jid;
      // If it's a LID, use as-is
      if (jid.startsWith('lid:')) return jid;
      // Otherwise, format as phone number
      return formatJid(jid);
    });

    const result = await sock.groupParticipantsUpdate(
      groupId,
      participantJids,
      action
    );

    res.json({ status: "ok", result });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/group/:groupId/subject", async (req, res) => {
  try {
    const { groupId } = req.params;
    const { subject } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    await sock.groupUpdateSubject(groupId, subject);
    res.json({ status: "ok" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/group/:groupId/description", async (req, res) => {
  try {
    const { groupId } = req.params;
    const { description } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    await sock.groupUpdateDescription(groupId, description);
    res.json({ status: "ok" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/group/:groupId/settings", async (req, res) => {
  try {
    const { groupId } = req.params;
    const { setting, value } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    await sock.groupSettingUpdate(groupId, setting, value);
    res.json({ status: "ok" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/group/:groupId/invite-code", async (req, res) => {
  try {
    const { groupId } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const code = await sock.groupInviteCode(groupId);
    res.json({
      inviteCode: code,
      inviteUrl: `https://chat.whatsapp.com/${code}`,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/group/:groupId/revoke-invite", async (req, res) => {
  try {
    const { groupId } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const newCode = await sock.groupRevokeInvite(groupId);
    res.json({
      status: "ok",
      newInviteCode: newCode,
      newInviteUrl: `https://chat.whatsapp.com/${newCode}`,
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/group/:groupId/leave", async (req, res) => {
  try {
    const { groupId } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    await sock.groupLeave(groupId);
    res.json({ status: "ok" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// =================== MEDIA & UTILITIES ===================

// Update media download/upload for LID support
app.post("/api/download-media", async (req, res) => {
  try {
    const { messageId } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const message = messageStore.get(messageId);
    if (!message) {
      return res.status(404).json({ error: "Message not found" });
    }

    const contentType = getContentType(message.message);
    if (![
      "imageMessage",
      "videoMessage",
      "audioMessage",
      "documentMessage",
    ].includes(contentType)) {
      return res.status(400).json({ error: "No downloadable media" });
    }

    // Updated download with proper options
    const mediaData = await downloadMediaMessage(
      message, 
      "buffer", 
      {}, 
      {
        logger,
        reuploadRequest: sock.updateMediaMessage
      }
    );

    if (!mediaData) {
      return res.status(400).json({ error: "Failed to download media" });
    }

    // Rest of the media handling code...
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get("/api/privacy", async (req, res) => {
  try {
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const privacy = await sock.fetchPrivacySettings();
    res.json({ privacySettings: privacy });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/privacy", async (req, res) => {
  try {
    const { setting, value } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    await sock.updatePrivacySettings({ [setting]: value });
    res.json({ status: "ok" });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-status", async (req, res) => {
  try {
    const { type, content } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    let statusMessage = {};

    if (type === "text") {
      statusMessage = { text: content };
    } else if (type === "image") {
      const response = await axios.get(content, {
        responseType: "arraybuffer",
      });
      statusMessage = { image: response.data };
    } else if (type === "video") {
      const response = await axios.get(content, {
        responseType: "arraybuffer",
      });
      statusMessage = { video: response.data };
    }

    const result = await sock.sendMessage("status@broadcast", statusMessage);
    res.json({ status: "ok", messageId: result.key.id });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post("/api/send-broadcast", async (req, res) => {
  try {
    const { message, recipients, delay: msgDelay = 2000 } = req.body;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    const results = [];

    for (let i = 0; i < recipients.length; i++) {
      try {
        const result = await sock.sendMessage(formatJid(recipients[i]), {
          text: message,
        });
        results.push({
          recipient: formatJid(recipients[i]),
          status: "sent",
          messageId: result.key.id,
        });

        if (i < recipients.length - 1) {
          await delay(parseInt(msgDelay));
        }
      } catch (error) {
        results.push({
          recipient: formatJid(recipients[i]),
          status: "failed",
          error: error.message,
        });
      }
    }

    res.json({ status: "ok", results });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// API endpoint per servire profile picture in tutti i formati con WBMP AD ALTISSIMA FEDELTA
 // Complete Profile Management System for WML Interface
// Enhanced Me/Profile page with full functionality

app.get("/wml/me.wml", async (req, res) => {
  try {
    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/home.wml")
      );
      return;
    }

    const user = sock.user;
    let profilePic = null;
    let status = null;

    try {
      profilePic = await sock.profilePictureUrl(user.id).catch(() => null);
      status = await sock.fetchStatus(user.id).catch(() => null);
    } catch (e) {
      // Silent fail for optional features
    }

    const body = `
      <p><b>My Profile</b></p>
      <p>Name: <b>${esc(user?.name || user?.notify || "Unknown")}</b></p>
      <p>Number: ${esc(
        user?.id?.replace("@s.whatsapp.net", "") || "Unknown"
      )}</p>
      <p>JID: <small>${esc(user?.id || "Unknown")}</small></p>
      ${
        status
          ? `<p>Status: <em>${esc(status.status || "No status")}</em></p>`
          : "<p>Status: <em>No status</em></p>"
      }
      ${
        status?.setAt
          ? `<p><small>Updated: ${new Date(
              status.setAt
            ).toLocaleString()}</small></p>`
          : ""
      }
      
      <p><b>Profile Actions:</b></p>
      <p>
        <a href="/wml/profile.edit-name.wml" accesskey="1">[1] Edit Name</a><br/>
        <a href="/wml/profile.edit-status.wml" accesskey="2">[2] Edit Status</a><br/>
        <a href="/wml/profile.picture.wml" accesskey="3">[3] View Profile Picture</a><br/>
      </p>
      
      <p><b>Account Info:</b></p>
      <p>Connected: ${esc(connectionState)}</p>
      <p>Sync Status: ${isFullySynced ? "Complete" : "In Progress"}</p>
      <p>Data: ${contactStore.size} contacts, ${chatStore.size} chats</p>
      <p>Messages: ${messageStore.size} stored</p>
      <p>Uptime: ${Math.floor(process.uptime() / 60)} minutes</p>
      
      ${navigationBar()}
      
      <do type="accept" label="Edit Name">
        <go href="/wml/profile.edit-name.wml"/>
      </do>
      <do type="options" label="Refresh">
        <go href="/wml/me.wml"/>
      </do>
    `;

    sendWml(res, card("me", "My Profile", body));
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Error",
        [e.message || "Failed to load profile"],
        "/wml/home.wml"
      )
    );
  }
});

// Edit Name page
app.get("/wml/profile.edit-name.wml", (req, res) => {
  if (!sock) {
    sendWml(
      res,
      resultCard("Error", ["Not connected to WhatsApp"], "/wml/me.wml")
    );
    return;
  }

  const user = sock.user;
  const currentName = user?.name || user?.notify || "";
  const success = req.query.success === "1";
  const preset = req.query.preset || "";

  const successMessage = success
    ? `
    <p><b>✓ Name Updated Successfully!</b></p>
    <p></p>
  `
    : "";

  const body = `
    <p><b>Edit Profile Name</b></p>
    ${successMessage}
    
    <p>Current Name: <b>${esc(currentName || "Not set")}</b></p>
    
    <p>New Name:</p>
    <input name="name" title="Your name" value="${esc(
      preset || currentName
    )}" size="25" maxlength="25"/>
    
    <p><small>Max 25 characters. This is your display name on WhatsApp.</small></p>
    
    <do type="accept" label="Update">
      <go method="post" href="/wml/profile.update-name">
        <postfield name="name" value="$(name)"/>
      </go>
    </do>
    
    <p><b>Quick Names:</b></p>
    <p>
      <a href="/wml/profile.edit-name.wml?preset=${encodeURIComponent(
        currentName
      )}" accesskey="1">[1] Keep Current</a><br/>
      <a href="/wml/profile.edit-name.wml?preset=${encodeURIComponent(
        user?.id?.replace("@s.whatsapp.net", "") || ""
      )}" accesskey="2">[2] Use Number</a><br/>
    </p>
    
    <p>
      <a href="/wml/me.wml" accesskey="0">[0] Back to Profile</a> |
      <a href="/wml/home.wml" accesskey="*">[*] Home</a>
    </p>
    
    <do type="options" label="Cancel">
      <go href="/wml/me.wml"/>
    </do>
  `;

  sendWml(res, card("edit-name", "Edit Name", body));
});

// Edit Status page
app.get("/wml/profile.edit-status.wml", async (req, res) => {
  try {
    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/me.wml")
      );
      return;
    }

    const user = sock.user;
    let currentStatus = "Loading...";

    try {
      const status = await sock.fetchStatus(user.id);
      currentStatus = status?.status || "No status set";
    } catch (e) {
      currentStatus = "Could not load status";
    }

    const success = req.query.success === "1";
    const preset = req.query.preset || "";

    const successMessage = success
      ? `
      <p><b>✓ Status Updated Successfully!</b></p>
      <p></p>
    `
      : "";

    const statusTemplates = [
      "Available",
      "Busy",
      "At work",
      "Can't talk, WhatsApp only",
      "In a meeting",
      "Sleeping",
      "Battery about to die",
    ];

    const body = `
      <p><b>Edit Status Message</b></p>
      ${successMessage}
      
      <p>Current Status: <b>${esc(currentStatus)}</b></p>
      
      <p>New Status:</p>
      <input name="status" title="Status message" value="${esc(
        preset ||
          (currentStatus !== "Loading..." &&
          currentStatus !== "Could not load status"
            ? currentStatus
            : "")
      )}" size="30" maxlength="139"/>
      
      <p><small>Max 139 characters. Leave empty to remove status.</small></p>
      
      <do type="accept" label="Update">
        <go method="post" href="/wml/profile.update-status">
          <postfield name="status" value="$(status)"/>
        </go>
      </do>
      
      <p><b>Quick Status:</b></p>
      ${statusTemplates
        .map(
          (tmpl, i) =>
            `<p><a href="/wml/profile.edit-status.wml?preset=${encodeURIComponent(
              tmpl
            )}" accesskey="${i + 1}">[${i + 1}] ${esc(tmpl)}</a></p>`
        )
        .join("")}
      
      <p>
        <a href="/wml/me.wml" accesskey="0">[0] Back to Profile</a> |
        <a href="/wml/home.wml" accesskey="*">[*] Home</a>
      </p>
      
      <do type="options" label="Cancel">
        <go href="/wml/me.wml"/>
      </do>
    `;

    sendWml(res, card("edit-status", "Edit Status", body));
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Error",
        [e.message || "Failed to load status editor"],
        "/wml/me.wml"
      )
    );
  }
});

// View Profile Picture page (RIUSA SISTEMA MEDIA MESSAGGI - FIXED XML)
app.get("/wml/profile.picture.wml", async (req, res) => {
  try {
    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/me.wml")
      );
      return;
    }

    const user = sock.user;
    let profilePic = null;
    let hasProfilePic = false;

    try {
      profilePic = await sock.profilePictureUrl(user.id, "image");
      hasProfilePic = !!profilePic;
    } catch (e) {
      // No profile picture or error loading
    }

    const body = hasProfilePic
      ? `
      <p><b>My Profile Picture</b></p>
      <p>Profile picture available</p>
      
      <p><b>Nokia Compatible:</b></p>
      <p>
        <a href="/wml/view-profile-wbmp.wml" accesskey="1">[1] WBMP View</a><br/>
        <a href="/api/profile-picture/small.jpg" accesskey="2">[2] Small JPG</a><br/>
        <a href="/api/profile-picture/small.png" accesskey="3">[3] Small PNG</a><br/>
      </p>
      
      <p><b>Full Quality:</b></p>
      <p>
        <a href="${esc(profilePic)}" accesskey="4">[4] Original</a><br/>
        <a href="/api/profile-picture/jpg" accesskey="5">[5] Download JPG</a><br/>
        <a href="/api/profile-picture/png" accesskey="6">[6] Download PNG</a><br/>
      </p>
      
      <p><b>Mobile Formats:</b></p>
      <p>
        <a href="/api/profile-picture/wbmp" accesskey="7">[7] WBMP Download</a><br/>
        <a href="/api/profile-picture/thumbnail" accesskey="8">[8] Thumbnail</a><br/>
      </p>
    `
      : `
      <p><b>My Profile Picture</b></p>
      <p><em>No profile picture set</em></p>
      
      <p><b>Info:</b></p>
      <p>Profile picture can only be updated from WhatsApp mobile app.</p>
      <p>Once set, you can view it here in multiple formats including WBMP for old devices.</p>
    `;

    const fullBody = `
      ${body}
      
      <p>
        <a href="/wml/me.wml" accesskey="0">[0] Back to Profile</a> |
        <a href="/wml/home.wml" accesskey="*">[*] Home</a>
      </p>
      
      <do type="accept" label="Back">
        <go href="/wml/me.wml"/>
      </do>
      <do type="options" label="Refresh">
        <go href="/wml/profile.picture.wml"/>
      </do>
    `;

    sendWml(res, card("profile-pic", "Profile Picture", fullBody));
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Error",
        [e.message || "Failed to load profile picture"],
        "/wml/me.wml"
      )
    );
  }
});

// Pagina dedicata per visualizzare profile picture in WBMP (FIXED XML + HIGH FIDELITY)
app.get("/wml/view-profile-wbmp.wml", async (req, res) => {
  try {
    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/me.wml")
      );
      return;
    }

    const user = sock.user;
    let hasProfilePic = false;

    try {
      const profilePic = await sock.profilePictureUrl(user.id, "image");
      hasProfilePic = !!profilePic;
    } catch (e) {
      // No profile picture
    }

    // Escape sicuro per WML
    const escWml = (text) =>
      (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&apos;");

    let body = "";
    let title = "Profile Picture (WBMP)";

    if (hasProfilePic) {
      body = `<p><b>My Profile Picture</b></p>
<p>Nokia 7210 Compatible Format</p>
<p>
<img src="/api/profile-picture/wbmp" alt="Profile WBMP"/>
</p>
<p>
<a href="/wml/profile.picture.wml" accesskey="0">[0] Back to Picture Options</a>
</p>
<p>
<a href="/wml/me.wml" accesskey="1">[1] Back to Profile</a> |
<a href="/wml/home.wml" accesskey="9">[9] Home</a>
</p>`;
    } else {
      body = `<p><b>No Profile Picture</b></p>
<p>No profile picture set</p>
<p>
<a href="/wml/me.wml" accesskey="0">[0] Back to Profile</a>
</p>`;
    }

    const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
<head><meta http-equiv="Cache-Control" content="max-age=0"/></head>
<card id="wbmp-profile" title="${escWml(title)}">
${body}
<do type="accept" label="Back">
<go href="/wml/profile.picture.wml"/>
</do>
<do type="options" label="Profile">
<go href="/wml/me.wml"/>
</do>
</card>
</wml>`;

    res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Pragma", "no-cache");

    const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");
    res.send(encodedBuffer);
  } catch (error) {
    logger.error("Profile WBMP view error:", error);
    res.status(500).send("Error loading profile WBMP view");
  }
});

// POST handler for updating name
app.post("/wml/profile.update-name", async (req, res) => {
  try {
    const { name } = req.body;
    if (!sock) throw new Error("Not connected");
    if (!name || name.trim().length === 0)
      throw new Error("Name cannot be empty");
    if (name.trim().length > 25)
      throw new Error("Name too long (max 25 characters)");

    await sock.updateProfileName(name.trim());

    sendWml(
      res,
      resultCard(
        "Name Updated",
        [
          `New name: ${esc(name.trim())}`,
          "Profile name updated successfully!",
          "Changes may take a few minutes to appear.",
        ],
        "/wml/profile.edit-name.wml?success=1",
        true
      )
    );
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Update Failed",
        [
          e.message || "Failed to update name",
          "Please try again or check connection",
        ],
        "/wml/profile.edit-name.wml"
      )
    );
  }
});

// POST handler for updating status
app.post("/wml/profile.update-status", async (req, res) => {
  try {
    const { status } = req.body;
    if (!sock) throw new Error("Not connected");
    if (status && status.length > 139)
      throw new Error("Status too long (max 139 characters)");

    await sock.updateProfileStatus(status || "");

    const statusText = status ? `"${status}"` : "Status cleared";

    sendWml(
      res,
      resultCard(
        "Status Updated",
        [
          `New status: ${esc(statusText)}`,
          "Status message updated successfully!",
          "Changes may take a few minutes to appear.",
        ],
        "/wml/profile.edit-status.wml?success=1",
        true
      )
    );
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Update Failed",
        [
          e.message || "Failed to update status",
          "Please try again or check connection",
        ],
        "/wml/profile.edit-status.wml"
      )
    );
  }
});

// API endpoint per servire profile picture in tutti i formati (come sistema media messaggi)

// Update profile picture endpoint
app.get("/api/profile-picture/:format", async (req, res) => {
  try {
    const { format } = req.params;
    if (!sock) return res.status(500).json({ error: "Not connected" });

    // Try to get profile picture - now handles LIDs
    let profilePicUrl;
    try {
      profilePicUrl = await sock.profilePictureUrl(sock.user.id, "image");
    } catch (e) {
      // Fallback to preview
      try {
        profilePicUrl = await sock.profilePictureUrl(sock.user.id, "preview");
      } catch (e2) {
        return res.status(404).json({ error: "No profile picture set" });
      }
    }

    if (!profilePicUrl) {
      return res.status(404).json({ error: "No profile picture set" });
    }

    // Download original image
    let mediaData = await axios.get(profilePicUrl, {
      responseType: "arraybuffer",
    });
    mediaData = mediaData.data;

    // Rest of the profile picture handling code...
  } catch (error) {
    logger.error("Profile picture download error:", error);
    res.status(500).json({ error: error.message });
  }
});

// Presence page - was referenced but not implemented
app.get("/wml/presence.wml", (req, res) => {
  if (!sock) {
    sendWml(
      res,
      resultCard("Error", ["Not connected to WhatsApp"], "/wml/home.wml")
    );
    return;
  }

  const body = `
    <p><b>Update Presence</b></p>
    <p>Set your availability status:</p>
    
    <p><b>Global Presence:</b></p>
    <p>
      <a href="/wml/presence.set.wml?type=available" accesskey="1">[1] Available</a><br/>
      <a href="/wml/presence.set.wml?type=unavailable" accesskey="2">[2] Unavailable</a><br/>
      <a href="/wml/presence.set.wml?type=composing" accesskey="3">[3] Typing</a><br/>
      <a href="/wml/presence.set.wml?type=recording" accesskey="4">[4] Recording</a><br/>
      <a href="/wml/presence.set.wml?type=paused" accesskey="5">[5] Paused</a><br/>
    </p>
    
    <p><b>Chat-Specific:</b></p>
    <p>Contact/Group JID:</p>
    <input name="jid" title="JID" size="20"/>
    
    <p>Presence type:</p>
    <select name="presence" title="Presence">
      <option value="available">Available</option>
      <option value="unavailable">Unavailable</option>
      <option value="composing">Typing</option>
      <option value="recording">Recording</option>
      <option value="paused">Paused</option>
    </select>
    
    <do type="accept" label="Set">
      <go method="post" href="/wml/presence.set">
        <postfield name="jid" value="$(jid)"/>
        <postfield name="presence" value="$(presence)"/>
      </go>
    </do>
    
    ${navigationBar()}
  `;

  sendWml(res, card("presence", "Presence", body));
});

// Privacy page - was referenced but not implemented
app.get("/wml/privacy.wml", async (req, res) => {
  try {
    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/home.wml")
      );
      return;
    }

    let privacySettings = null;
    try {
      privacySettings = await sock.fetchPrivacySettings();
    } catch (e) {
      // Silent fail
    }

    const body = `
    
      <p><b>Privacy Settings</b></p>
      
      ${
        privacySettings
          ? `
      <p><b>Current Settings:</b></p>
      <p>Last Seen: ${esc(privacySettings.lastSeen || "Unknown")}</p>
      <p>Profile Photo: ${esc(privacySettings.profilePicture || "Unknown")}</p>
      <p>Status: ${esc(privacySettings.status || "Unknown")}</p>
      <p>Read Receipts: ${esc(privacySettings.readReceipts || "Unknown")}</p>
      `
          : "<p><em>Privacy settings unavailable</em></p>"
      }
      
      <p><b>Privacy Actions:</b></p>
      <p>
        <a href="/wml/privacy.lastseen.wml" accesskey="1">[1] Last Seen</a><br/>
        <a href="/wml/privacy.profile.wml" accesskey="2">[2] Profile Photo</a><br/>
        <a href="/wml/privacy.status.wml" accesskey="3">[3] Status Privacy</a><br/>
        <a href="/wml/privacy.receipts.wml" accesskey="4">[4] Read Receipts</a><br/>
        <a href="/wml/privacy.groups.wml" accesskey="5">[5] Groups</a><br/>
      </p>
      
      <p><b>Blocked Contacts:</b></p>
      <p>
        <a href="/wml/blocked.list.wml" accesskey="7">[7] View Blocked</a><br/>
        <a href="/wml/block.contact.wml" accesskey="8">[8] Block Contact</a><br/>
      </p>
      
      ${navigationBar()}
      
      <do type="accept" label="Refresh">
        <go href="/wml/privacy.wml"/>
      </do>
    `;

    sendWml(res, card("privacy", "Privacy", body));
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Error",
        [e.message || "Failed to load privacy settings"],
        "/wml/home.wml"
      )
    );
  }
});

// =================== POST HANDLERS FOR QUICK ACTIONS ===================

// Presence setting handler
app.post("/wml/presence.set", async (req, res) => {
  try {
    const { jid, presence = "available" } = req.body;
    const type = req.query.type || presence;

    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/presence.wml")
      );
      return;
    }

    if (jid && jid.trim()) {
      await sock.sendPresenceUpdate(type, formatJid(jid.trim()));
      sendWml(
        res,
        resultCard(
          "Presence Updated",
          [
            `Set ${type} for ${esc(jid.trim())}`,
            "Presence updated successfully",
          ],
          "/wml/presence.wml",
          true
        )
      );
    } else {
      await sock.sendPresenceUpdate(type);
      sendWml(
        res,
        resultCard(
          "Presence Updated",
          [`Global presence set to ${type}`, "Presence updated successfully"],
          "/wml/presence.wml",
          true
        )
      );
    }
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Presence Failed",
        [e.message || "Failed to update presence"],
        "/wml/presence.wml"
      )
    );
  }
});

// Quick presence setting via GET for simple links
app.get("/wml/presence.set.wml", async (req, res) => {
  try {
    const { type = "available", jid } = req.query;

    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/presence.wml")
      );
      return;
    }

    if (jid && jid.trim()) {
      await sock.sendPresenceUpdate(type, formatJid(jid.trim()));
      sendWml(
        res,
        resultCard(
          "Presence Updated",
          [
            `Set ${type} for ${esc(jid.trim())}`,
            "Presence updated successfully",
          ],
          "/wml/presence.wml",
          true
        )
      );
    } else {
      await sock.sendPresenceUpdate(type);
      sendWml(
        res,
        resultCard(
          "Global Presence Updated",
          [`Presence set to: ${type}`, "All contacts will see this status"],
          "/wml/presence.wml",
          true
        )
      );
    }
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Presence Failed",
        [e.message || "Failed to update presence"],
        "/wml/presence.wml"
      )
    );
  }
});

// =================== MISSING UTILITY ENDPOINTS ===================

// Broadcast page - was referenced but missing
app.get("/wml/broadcast.wml", (req, res) => {
  const body = `
    <p><b>Broadcast Message</b></p>
    <p>Send message to multiple contacts</p>
    
    <p>Recipients (comma-separated):</p>
    <input name="recipients" title="Phone numbers" size="25" maxlength="500"/>
    
    <p>Message:</p>
    <input name="message" title="Your message" size="30" maxlength="1000"/>
    
    <p>Delay between sends (ms):</p>
    <select name="delay" title="Delay">
      <option value="1000">1 second</option>
      <option value="2000">2 seconds</option>
      <option value="5000">5 seconds</option>
      <option value="10000">10 seconds</option>
    </select>
    
    <do type="accept" label="Send">
      <go method="post" href="/wml/broadcast.send">
        <postfield name="recipients" value="$(recipients)"/>
        <postfield name="message" value="$(message)"/>
        <postfield name="delay" value="$(delay)"/>
      </go>
    </do>
    
    <p>
      <a href="/wml/contacts.wml" accesskey="1">[1] Select from Contacts</a> |
      <a href="/wml/home.wml" accesskey="0">[0] Home</a>
    </p>
  `;

  sendWml(res, card("broadcast", "Broadcast", body));
});

// Debug page - was referenced but missing
app.get("/wml/debug.wml", (req, res) => {
  const memUsage = process.memoryUsage();
  const uptime = Math.floor(process.uptime());

  const body = `

    <p><b>Debug Information</b></p>
    
    <p><b>Connection:</b></p>
    <p>State: ${esc(connectionState)}</p>
    <p>Socket: ${sock ? "Active" : "Null"}</p>
    <p>User: ${sock?.user?.id ? esc(sock.user.id) : "None"}</p>
    <p>QR: ${currentQR ? "Available" : "None"}</p>
    
    <p><b>Data Stores:</b></p>
    <p>Contacts: ${contactStore.size}</p>
    <p>Chats: ${chatStore.size}</p>
    <p>Messages: ${messageStore.size}</p>
    <p>Sync Status: ${isFullySynced ? "Complete" : "Pending"}</p>
    <p>Sync Attempts: ${syncAttempts}</p>
    
    <p><b>System:</b></p>
    <p>Uptime: ${uptime}s</p>
    <p>Memory: ${Math.round(memUsage.rss / 1024 / 1024)}MB</p>
    <p>Node: ${process.version}</p>
    <p>Env: ${process.env.NODE_ENV || "dev"}</p>
    
    <p><b>Debug Actions:</b></p>
    <p>
      <a href="/wml/debug.stores.wml" accesskey="1">[1] Store Details</a><br/>
      <a href="/wml/debug.logs.wml" accesskey="2">[2] Recent Logs</a><br/>
      <a href="/wml/debug.test.wml" accesskey="3">[3] Connection Test</a><br/>
    </p>
    
    ${navigationBar()}
    
    <do type="accept" label="Refresh">
      <go href="/wml/debug.wml"/>
    </do>
  `;

  sendWml(res, card("debug", "Debug", body, "/wml/debug.wml"));
});

// Logout confirmation page
app.get("/wml/logout.wml", (req, res) => {
  const body = `
    <p><b>Logout Confirmation</b></p>
    <p>This will:</p>
    <p>• Disconnect from WhatsApp</p>
    <p>• Clear all session data</p>
    <p>• Remove authentication</p>
    <p>• Clear local contacts/chats</p>
    
    <p><b>Are you sure?</b></p>
    <p>
      <a href="/wml/logout.confirm.wml" accesskey="1">[1] Yes, Logout</a><br/>
      <a href="/wml/home.wml" accesskey="0">[0] Cancel</a><br/>
    </p>
    
    <do type="accept" label="Cancel">
      <go href="/wml/home.wml"/>
    </do>
  `;

  sendWml(res, card("logout", "Logout", body));
});

// Logout execution
app.get("/wml/logout.confirm.wml", async (req, res) => {
  try {
    if (sock) {
      await sock.logout();
    }

    // Clear auth files
    if (fs.existsSync("./auth_info_baileys")) {
      fs.rmSync("./auth_info_baileys", { recursive: true });
    }

    // Clear stores
    contactStore.clear();
    chatStore.clear();
    messageStore.clear();
    isFullySynced = false;
    syncAttempts = 0;
    currentQR = null;
    connectionState = "disconnected";

    sendWml(
      res,
      resultCard(
        "Logged Out",
        [
          "Successfully logged out",
          "All data cleared",
          "You can scan QR to reconnect",
        ],
        "/wml/home.wml",
        true
      )
    );
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Logout Error",
        [e.message || "Logout failed"],
        "/wml/home.wml"
      )
    );
  }
});

// =================== SYNC ENDPOINTS ===================

// Force sync endpoints that were referenced but missing handlers
app.get("/wml/sync.full.wml", async (req, res) => {
  try {
    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/status.wml")
      );
      return;
    }

    // Trigger the existing performInitialSync function
    performInitialSync();

    sendWml(
      res,
      resultCard(
        "Sync Started",
        [
          "Full sync initiated",
          "This may take a few minutes",
          "Check status page for progress",
        ],
        "/wml/status.wml",
        true
      )
    );
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Sync Failed",
        [e.message || "Failed to start sync"],
        "/wml/status.wml"
      )
    );
  }
});

app.get("/wml/sync.contacts.wml", async (req, res) => {
  try {
    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/status.wml")
      );
      return;
    }

    const initialCount = contactStore.size;

    // Wait for contact events (contacts sync automatically in Baileys)
    await delay(3000);

    const finalCount = contactStore.size;
    const newContacts = finalCount - initialCount;

    sendWml(
      res,
      resultCard(
        "Contact Sync Complete",
        [
          `Initial contacts: ${initialCount}`,
          `Final contacts: ${finalCount}`,
          `New contacts: ${newContacts}`,
          "Contacts sync via WhatsApp events",
        ],
        "/wml/status.wml",
        true
      )
    );
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Contact Sync Failed",
        [e.message || "Failed to sync contacts"],
        "/wml/status.wml"
      )
    );
  }
});

// Enhanced Chats page with search and pagination
app.get("/wml/chats.wml", async (req, res) => {
  const userAgent = req.headers["user-agent"] || "";

  // Use req.query for GET requests, like in contacts
  const query = req.query;

  const page = Math.max(1, parseInt(query.page || "1"));
  let limit = Math.max(1, Math.min(20, parseInt(query.limit || "10")));

  // More restrictive limits for WAP 1.0 devices (like contacts)
  if (userAgent.includes("Nokia") || userAgent.includes("UP.Browser")) {
    limit = Math.min(5, limit); // Max 5 items per page
  }
  limit = Math.min(3, limit); // Max 5 i
  const search = query.q || "";
  const showGroups = query.groups !== "0"; // Default show groups
  const showDirect = query.direct !== "0"; // Default show direct chats

  // Auto-sync if no chats and not synced yet
  if (chatStore.size === 0 && !isFullySynced && sock) {
    try {
      logger.info("💬 No chats found, attempting auto-sync...");
      const groups = await sock.groupFetchAllParticipating();

      Object.keys(groups).forEach((chatId) => {
        if (!chatStore.has(chatId)) {
          chatStore.set(chatId, []);
        }
      });

      logger.info(`💬 Auto-synced ${Object.keys(groups).length} group chats`);
      await delay(1000); // Brief wait for additional events
    } catch (syncError) {
      logger.warn("⚠️ Auto-sync failed:", syncError.message);
    }
  }

  // Replace your entire chat processing section with this:
// Replace the chat mapping section in chats.wml with this corrected version
let chats = await Promise.all(
  Array.from(chatStore.keys()).map(async (chatId) => {
    const messages = chatStore.get(chatId) || [];
    const lastMessage = messages.length > 0 ? messages[messages.length - 1] : null;

    const isGroup = chatId.endsWith("@g.us");
    const phoneNumber = chatId
      .replace("@s.whatsapp.net", "")
      .replace("@g.us", "");

    // Use getContactName for better name resolution
    const chatName = await getContactName(chatId, sock);
    
    // Get contact object safely
    const contact = contactStore.get(chatId) || {};

    // Fixed: Use lastMessage instead of undefined msg
    const lastMessageText = lastMessage
      ? messageText(lastMessage)
      : "No messages";

    const lastTimestamp = lastMessage
      ? Number(lastMessage.messageTimestamp)
      : 0;
    const unreadCount = messages.filter((m) => !m.key.fromMe).length;

    return {
      id: chatId,
      name: chatName,
      isGroup,
      phoneNumber: isGroup ? null : phoneNumber,
      messageCount: messages.length,
      lastMessage: {
        text: lastMessageText || "No message text", // Ensure text is never undefined
        timestamp: lastTimestamp,
        fromMe: lastMessage ? lastMessage.key.fromMe : false,
        timeStr:
          lastTimestamp > 0
            ? new Date(lastTimestamp * 1000).toLocaleString("en-GB", {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })
            : "Never",
      },
      unreadCount,
      contact, // Now properly defined in all cases
    };
  })
);

  // Filter by chat type
  if (!showGroups) {
    chats = chats.filter((c) => !c.isGroup);
  }
  if (!showDirect) {
    chats = chats.filter((c) => c.isGroup);
  }

  // Apply search filter (like contacts)
  if (search) {
    const searchLower = search.toLowerCase();
    chats = chats.filter((c) => {
      const nameMatch = c.name.toLowerCase().includes(searchLower);
      const numberMatch = c.phoneNumber && c.phoneNumber.includes(searchLower);
      const messageMatch = c.lastMessage.text
        .toLowerCase()
        .includes(searchLower);
      return nameMatch || numberMatch || messageMatch;
    });
  }

  // Sort by last message timestamp (most recent first)
  chats.sort((a, b) => b.lastMessage.timestamp - a.lastMessage.timestamp);

  const total = chats.length;
  const start = (page - 1) * limit;
  const items = chats.slice(start, start + limit);

  // Safe WML escaping function (like contacts)
  const escWml = (text) =>
    (text || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&apos;");

  // Page header
  const searchHeader = search
    ? `<p><b>Search Results for:</b> ${escWml(search)} (${total})</p>`
    : `<p><b>All Chats</b> (${total})</p>`;

  // Chat list
  // In your chats.wml route, fix the list mapping section:

  const list =
    items
      .map((c, idx) => {
        const typeIcon = c.isGroup ? "[GROUP]" : "[CHAT]";
        const unreadBadge = c.unreadCount > 0 ? ` (${c.unreadCount})` : "";

        // Add safety checks for lastMessage properties
        const lastMessageText = c.lastMessage?.text || "No message text";
        const fromMe = c.lastMessage?.fromMe || false;

        // Safe text processing
        const messagePreview =
          lastMessageText.length > 40
            ? lastMessageText.substring(0, 37) + "..."
            : lastMessageText;
        const fromIndicator = fromMe ? "You: " : "";

        // Safe time string access
        const timeStr = c.lastMessage?.timeStr || "Unknown time";

        // Show both name and number/JID
        const displayNumber = c.phoneNumber || c.id.replace("@s.whatsapp.net", "").replace("@g.us", "");

        return `<p>${start + idx + 1}. ${typeIcon} ${escWml(
          c.name
        )}${unreadBadge}<br/>
      <small>${escWml(displayNumber)}</small><br/>
      <small>${escWml(fromIndicator + messagePreview)}</small><br/>
      <small>${escWml(timeStr)} | ${c.messageCount} msgs</small><br/>
      <a href="/wml/chat.wml?jid=${encodeURIComponent(
        c.id
      )}&amp;limit=15">[Open Chat]</a> 
      <a href="/wml/send.text.wml?to=${encodeURIComponent(
        c.id
      )}">[Send Message]</a>
      ${
        c.phoneNumber
          ? `  <a href="wtai://wp/mc;${c.phoneNumber}">[Call]</a>`
          : ""
      }
      ${
        c.phoneNumber
          ? `  <a href="wtai://wp/ms;${c.phoneNumber};">[SMS]</a>`
          : ""
      }
      ${
        c.phoneNumber
          ? `  <a href="wtai://wp/ap;${c.phoneNumber};">[Add to Phone]</a>`
          : ""
      }
    </p>`;
      })
      .join("") || "<p>No chats found.</p>";

  // Pagination con First/Last e numeri di pagina
  const totalPages = Math.ceil(total / limit) || 1;

  const firstPage =
    page > 1
      ? `<a href="/wml/chats.wml?page=1&amp;limit=${limit}&amp;q=${encodeURIComponent(
          search
        )}&amp;groups=${showGroups ? 1 : 0}&amp;direct=${
          showDirect ? 1 : 0
        }">[First]</a>`
      : "";

  const prevPage =
    page > 1
      ? `<a href="/wml/chats.wml?page=${
          page - 1
        }&amp;limit=${limit}&amp;q=${encodeURIComponent(search)}&amp;groups=${
          showGroups ? 1 : 0
        }&amp;direct=${showDirect ? 1 : 0}">[Previous]</a>`
      : "";

  const nextPage =
    page < totalPages
      ? `<a href="/wml/chats.wml?page=${
          page + 1
        }&amp;limit=${limit}&amp;q=${encodeURIComponent(search)}&amp;groups=${
          showGroups ? 1 : 0
        }&amp;direct=${showDirect ? 1 : 0}">[Next]</a>`
      : "";

  const lastPage =
    page < totalPages
      ? `<a href="/wml/chats.wml?page=${totalPages}&amp;limit=${limit}&amp;q=${encodeURIComponent(
          search
        )}&amp;groups=${showGroups ? 1 : 0}&amp;direct=${
          showDirect ? 1 : 0
        }">[Last]</a>`
      : "";

  // numeri di pagina (massimo 5 visibili: due prima, attuale, due dopo)
  let pageNumbers = "";
  const startPage = Math.max(1, page - 2);
  const endPage = Math.min(totalPages, page + 2);
  for (let p = startPage; p <= endPage; p++) {
    if (p === page) {
      pageNumbers += `<b>[${p}]</b> `;
    } else {
      pageNumbers += `<a href="/wml/chats.wml?page=${p}&amp;limit=${limit}&amp;q=${encodeURIComponent(
        search
      )}&amp;groups=${showGroups ? 1 : 0}&amp;direct=${
        showDirect ? 1 : 0
      }">${p}</a> `;
    }
  }

  const pagination = `
    <p>
      ${firstPage} ${firstPage && prevPage ? "" : ""} ${prevPage}
      ${pageNumbers}
      ${nextPage} ${nextPage && lastPage ? "" : ""} ${lastPage}
    </p>`;

  // Simplified search form (like contacts)
  const searchForm = `
    <p><b>Search chats:</b></p>
    <p>
      <input name="q" title="Search..." value="${escWml(
        search
      )}" emptyok="true" size="15" maxlength="30"/>
     
      <do type="accept" label="Search">
        <go href="/wml/chats.wml" method="get">
          <postfield name="q" value="$(q)"/>
          <postfield name="groups" value="$(groups)"/>
          <postfield name="direct" value="$(direct)"/>
          <postfield name="page" value="1"/>
          <postfield name="limit" value="${limit}"/>
        </go>
      </do>
    </p>`;

  // Filter toggles (simplified)
  const filterToggles = `
    <p><b>Quick Filters:</b></p>
    <p>
      ${
        showGroups
          ? `<a href="/wml/chats.wml?page=${page}&amp;limit=${limit}&amp;q=${encodeURIComponent(
              search
            )}&amp;groups=0&amp;direct=${showDirect ? 1 : 0}">[Hide Groups]</a>`
          : `<a href="/wml/chats.wml?page=${page}&amp;limit=${limit}&amp;q=${encodeURIComponent(
              search
            )}&amp;groups=1&amp;direct=${showDirect ? 1 : 0}">[Show Groups]</a>`
      } 
      ${
        showDirect
          ? `<a href="/wml/chats.wml?page=${page}&amp;limit=${limit}&amp;q=${encodeURIComponent(
              search
            )}&amp;groups=${showGroups ? 1 : 0}&amp;direct=0">[Hide Direct]</a>`
          : `<a href="/wml/chats.wml?page=${page}&amp;limit=${limit}&amp;q=${encodeURIComponent(
              search
            )}&amp;groups=${showGroups ? 1 : 0}&amp;direct=1">[Show Direct]</a>`
      }
    </p>`;

  // WML card body
  const body = `
    <p><b>Chats - Page ${page}/${Math.ceil(total / limit) || 1}</b></p>
    ${searchHeader}
    ${searchForm}
    ${filterToggles}
    ${list}
    ${pagination}
    <p>
      <a href="/wml/home.wml">[Home]</a> 
      <a href="/wml/contacts.wml">[Contacts]</a> 
      <a href="/wml/send-menu.wml">[New Message]</a>
    </p>
    <do type="accept" label="Refresh">
      <go href="/wml/chats.wml?page=${page}&amp;limit=${limit}&amp;q=${encodeURIComponent(
    search
  )}&amp;groups=${showGroups ? 1 : 0}&amp;direct=${showDirect ? 1 : 0}"/>
    </do>
    <do type="options" label="Menu">
      <go href="/wml/menu.wml"/>
    </do>`;

  // Create complete WML string
  const wmlOutput = `<?xml version="1.0"?>
<!DOCTYPE wml PUBLIC "-//WAPFORUM//DTD WML 1.1//EN" "http://www.wapforum.org/DTD/wml_1.1.xml">
<wml>
  <head>
    <meta http-equiv="Cache-Control" content="max-age=0"/>
  </head>
  <card id="chats" title="Chats">
    ${body}
  </card>
</wml>`;

  // --- KEY MODIFICATIONS FOR COMPATIBILITY (like contacts) ---

  // 1. Set headers for WAP 1.0 with correct encoding (ISO-8859-1)
  res.setHeader("Content-Type", "text/vnd.wap.wml; charset=iso-8859-1");
  res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
  res.setHeader("Pragma", "no-cache");
  res.setHeader("Expires", "0");

  // 2. Encode the entire WML string to ISO-8859-1 buffer
  const encodedBuffer = iconv.encode(wmlOutput, "iso-8859-1");

  // 3. Send the encoded buffer
  res.send(encodedBuffer);
});

// Advanced chat search page
app.get("/wml/chats.search.wml", async (req, res) => {
  const prevQuery = esc(req.query.q || "");
  const prevType = req.query.type || "all";
  const prevSort = req.query.sort || "recent";

  const body = `
    <p><b>Advanced Chat Search</b></p>
    
    <p>Search query:</p>
    <input name="q" title="Search query" value="${prevQuery}" size="20" maxlength="100"/>
    
    <p>Chat type:</p>
    <select name="type" title="Chat Type">
      <option value="all" ${
        prevType === "all" ? 'selected="selected"' : ""
      }>All Chats</option>
      <option value="direct" ${
        prevType === "direct" ? 'selected="selected"' : ""
      }>Direct Messages</option>
      <option value="groups" ${
        prevType === "groups" ? 'selected="selected"' : ""
      }>Groups Only</option>
    </select>
    
    <p>Sort by:</p>
    <select name="sort" title="Sort Order">
      <option value="recent" ${
        prevSort === "recent" ? 'selected="selected"' : ""
      }>Most Recent</option>
      <option value="messages" ${
        prevSort === "messages" ? 'selected="selected"' : ""
      }>Most Messages</option>
      <option value="name" ${
        prevSort === "name" ? 'selected="selected"' : ""
      }>Name A-Z</option>
    </select>
    
    <p>Results per page:</p>
    <select name="limit" title="Limit">
      <option value="5">5 results</option>
      <option value="10">10 results</option>
      <option value="20">20 results</option>
    </select>
    
    <do type="accept" label="Search">
      <go href="/wml/chats.results.wml" method="get">
        <postfield name="q" value="$(q)"/>
        <postfield name="type" value="$(type)"/>
        <postfield name="sort" value="$(sort)"/>
        <postfield name="limit" value="$(limit)"/>
      </go>
    </do>
    
    <p><b>Quick Searches:</b></p>
    <p>
      <a href="/wml/chats.wml?q=unread" accesskey="1">[1] Recent Activity</a><br/>
      <a href="/wml/chats.wml?groups=1&amp;direct=0" accesskey="2">[2] Groups Only</a><br/>
      <a href="/wml/chats.wml?groups=0&amp;direct=1" accesskey="3">[3] Direct Only</a><br/>
    </p>
    
    ${navigationBar()}
  `;

  sendWml(res, card("chats-search", "Chat Search", body));
});

// Chat search results
app.get("/wml/chats.results.wml", async (req, res) => {
  const q = String(req.query.q || "").trim();
  const chatType = req.query.type || "all";
  const sortBy = req.query.sort || "recent";
  const limit = Math.max(1, Math.min(50, parseInt(req.query.limit || "20")));

  if (!q || q.length < 1) {
    sendWml(
      res,
      resultCard("Search Error", ["Query is required"], "/wml/chats.search.wml")
    );
    return;
  }

  // Build and filter chat list (similar to main chats.wml logic)
  let chats = await Promise.all(
    Array.from(chatStore.keys()).map(async (chatId) => {
      const messages = chatStore.get(chatId) || [];
      const lastMessage =
        messages.length > 0 ? messages[messages.length - 1] : null;

      const isGroup = chatId.endsWith("@g.us");
      const phoneNumber = chatId
        .replace("@s.whatsapp.net", "")
        .replace("@g.us", "");

      // Use getContactName for better name resolution
      const chatName = await getContactName(chatId, sock);

      return {
      id: chatId,
      name: chatName,
      isGroup,
      phoneNumber: isGroup ? null : phoneNumber,
      messageCount: messages.length,
      lastMessage: {
        text: lastMessage ? messageText(lastMessage) : "No messages",
        timestamp: lastMessage ? Number(lastMessage.messageTimestamp) : 0,
      },
    };
  })
  );

  // Filter by type
  if (chatType === "direct") {
    chats = chats.filter((c) => !c.isGroup);
  } else if (chatType === "groups") {
    chats = chats.filter((c) => c.isGroup);
  }

  // Apply search filter
  const searchLower = q.toLowerCase();
  chats = chats.filter((c) => {
    const nameMatch = c.name.toLowerCase().includes(searchLower);
    const numberMatch = c.phoneNumber && c.phoneNumber.includes(searchLower);
    const messageMatch = c.lastMessage.text.toLowerCase().includes(searchLower);
    return nameMatch || numberMatch || messageMatch;
  });

  // Sort results
  if (sortBy === "recent") {
    chats.sort((a, b) => b.lastMessage.timestamp - a.lastMessage.timestamp);
  } else if (sortBy === "messages") {
    chats.sort((a, b) => b.messageCount - a.messageCount);
  } else if (sortBy === "name") {
    chats.sort((a, b) => a.name.localeCompare(b.name));
  }

  const results = chats.slice(0, limit);

  const resultList =
    results
      .map((c, idx) => {
        const typeIcon = c.isGroup ? "[GROUP]" : "[CHAT]";
        const messagePreview = truncate(c.lastMessage.text, 50);
        const lastActivity =
          c.lastMessage.timestamp > 0
            ? new Date(c.lastMessage.timestamp * 1000).toLocaleString("en-GB", {
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })
            : "No activity";

        // Show both name and number/JID
        const displayNumber = c.phoneNumber || c.id.replace("@s.whatsapp.net", "").replace("@g.us", "");

        return `<p><b>${idx + 1}.</b> ${typeIcon} ${esc(c.name)}<br/>
      <small>${esc(displayNumber)}</small><br/>
      <small>${esc(messagePreview)}</small><br/>
      <small>${lastActivity} | ${c.messageCount} msgs</small><br/>
      <a href="/wml/chat.wml?jid=${encodeURIComponent(
        c.id
      )}&amp;limit=15">[Open]</a> |
      <a href="/wml/send.text.wml?to=${encodeURIComponent(c.id)}">[Send]</a>
    </p>`;
      })
      .join("") || "<p>No matching chats found.</p>";

  const body = `
    <p><b>Chat Search Results</b></p>
    <p>Query: <b>${esc(q)}</b></p>
    <p>Type: ${esc(chatType)} | Sort: ${esc(sortBy)}</p>
    <p>Found: ${results.length} of ${chats.length}</p>
    
    ${resultList}
    
    <p><b>Search Again:</b></p>
    <p>
      <a href="/wml/chats.search.wml?q=${encodeURIComponent(
        q
      )}" accesskey="1">[1] Modify Search</a> |
      <a href="/wml/chats.wml" accesskey="0">[0] All Chats</a>
    </p>
    
    <do type="accept" label="Back">
      <go href="/wml/chats.wml"/>
    </do>
  `;

  sendWml(res, card("chat-results", "Search Results", body));
});
app.get("/wml/sync.chats.wml", async (req, res) => {
  try {
    if (!sock) {
      sendWml(
        res,
        resultCard("Error", ["Not connected to WhatsApp"], "/wml/status.wml")
      );
      return;
    }

    const initialCount = chatStore.size;

    // Fetch groups (the main chat sync method available)
    const groups = await sock.groupFetchAllParticipating();
    Object.keys(groups).forEach((chatId) => {
      if (!chatStore.has(chatId)) {
        chatStore.set(chatId, []);
      }
    });

    await delay(2000); // Wait for additional chat events

    const finalCount = chatStore.size;
    const newChats = finalCount - initialCount;

    sendWml(
      res,
      resultCard(
        "Chat Sync Complete",
        [
          `Groups fetched: ${Object.keys(groups).length}`,
          `Initial chats: ${initialCount}`,
          `Final chats: ${finalCount}`,
          `New chats: ${newChats}`,
        ],
        "/wml/status.wml",
        true
      )
    );
  } catch (e) {
    sendWml(
      res,
      resultCard(
        "Chat Sync Failed",
        [e.message || "Failed to sync chats"],
        "/wml/status.wml"
      )
    );
  }
});

// =================== ERROR HANDLING & SERVER SETUP ===================

// Error handling
app.use((err, req, res, next) => {
  console.error("Server Error:", err);
  res.status(500).json({
    error: "Internal server error",
    details: process.env.NODE_ENV === "development" ? err.message : undefined,
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({
    error: "Endpoint not found",
    path: req.path,
    method: req.method,
    suggestion: "Check the API documentation for available endpoints",
  });
});

export { app, sock, contactStore, chatStore, messageStore };
