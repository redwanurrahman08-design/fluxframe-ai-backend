import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import os from "os";
import crypto from "crypto";
import ffmpeg from "fluent-ffmpeg";
import ffmpegPath from "ffmpeg-static";

ffmpeg.setFfmpegPath(ffmpegPath);

const app = express();
app.use(express.json({ limit: "2mb" }));
app.use(cors({ origin: process.env.CORS_ORIGIN || "*" }));

const PORT = process.env.PORT || 3000;

const PUBLIC_BASE_URL = process.env.PUBLIC_BASE_URL || "";
const HF_TOKEN = process.env.HF_TOKEN || "";
const HF_MODEL = process.env.HF_MODEL || "runwayml/stable-diffusion-v1-5";

const VIDEO_DIR = path.join(process.cwd(), "videos");
fs.mkdirSync(VIDEO_DIR, { recursive: true });

// In-memory jobs store (demo). Production: database/redis.
const jobs = new Map();

function makeJobId() {
  return "ff_" + crypto.randomBytes(6).toString("hex");
}

function countWords(text) {
  const t = (text ?? "").trim();
  if (!t) return 0;
  return t.split(/\s+/).filter(Boolean).length;
}

function clamp(n, a, b) {
  return Math.max(a, Math.min(b, n));
}

function aspectToSize(aspectRatio) {
  if (aspectRatio === "16:9") return { W: 1280, H: 720 };
  if (aspectRatio === "1:1") return { W: 1024, H: 1024 };
  return { W: 720, H: 1280 }; // 9:16
}

// Generate smaller images (cheaper/faster) then ffmpeg scales/crops to final size.
function aspectToGenSize(aspectRatio) {
  if (aspectRatio === "16:9") return { w: 768, h: 512 };
  if (aspectRatio === "1:1") return { w: 768, h: 768 };
  return { w: 512, h: 768 }; // 9:16
}

function normalizeLines(prompt) {
  return (prompt || "")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
}

function guessScenesFromPrompt(prompt, durationSec) {
  const lines = normalizeLines(prompt);
  const content = lines
    .filter((l) => !/^(hook|main|proof|cta)\b/i.test(l))
    .filter((l) => l.length >= 6);

  const minScenes = 4;
  const maxScenes = 10;
  const targetScenes = clamp(Math.round(durationSec / 12), minScenes, maxScenes);

  let scenes = [];
  if (content.length >= minScenes) {
    const chunkSize = Math.ceil(content.length / targetScenes);
    for (let i = 0; i < content.length; i += chunkSize) {
      scenes.push(content.slice(i, i + chunkSize).join(" "));
      if (scenes.length >= targetScenes) break;
    }
  } else if (content.length > 0) {
    scenes = content.slice(0, minScenes);
  } else {
    scenes = [
      "Prompt → Storyboard → Scenes → Motion → Captions → Export.",
      "Scene-based generation with consistent styling.",
      "30–120 seconds. Up to 2500 words. Platform-ready formats.",
      "Generate your next video with FluxFrame AI.",
    ];
  }

  const titles = [
    "Hook",
    "Value",
    "Features",
    "Proof",
    "CTA",
    "Export",
    "Format",
    "Brand",
    "Launch",
    "Finish",
  ];

  const n = Math.max(minScenes, scenes.length);
  const segDur = durationSec / n;

  const timed = Array.from({ length: n }).map((_, i) => ({
    i,
    tStart: i * segDur,
    tEnd: (i + 1) * segDur,
    title: titles[i] || `Scene ${i + 1}`,
    text: scenes[i] || scenes[scenes.length - 1] || "",
  }));

  timed[0].title = "Hook";
  timed[timed.length - 1].title = "CTA";

  const storyboard = timed.map((s) => ({
    t: Math.floor(s.tStart),
    title: s.title,
    note: (s.text || "").slice(0, 120) + ((s.text || "").length > 120 ? "…" : ""),
  }));

  return { timed, storyboard };
}

function updateJob(jobId, patch) {
  const j = jobs.get(jobId);
  if (!j) return;
  jobs.set(jobId, { ...j, ...patch, updatedAt: Date.now() });
}

function escDrawtext(txt) {
  return (txt || "")
    .replaceAll("\\", "\\\\")
    .replaceAll(":", "\\:")
    .replaceAll("'", "\\'")
    .replaceAll("\n", " ")
    .slice(0, 160);
}

/**
 * FINAL HF IMAGE GENERATION (router.huggingface.co)
 */
async function hfGenerateImageToFile({ prompt, width, height, outPath }) {
  if (!HF_TOKEN) throw new Error("Missing HF_TOKEN env var (Hugging Face token).");

  // NEW Hugging Face router endpoint (api-inference is no longer supported)
  const endpoints = [
    `https://router.huggingface.co/models/${HF_MODEL}`,
    // fallback (some accounts/providers use this path)
    `https://router.huggingface.co/hf-inference/models/${HF_MODEL}`,
  ];

  const body = {
    inputs: prompt,
    parameters: {
      width,
      height,
      num_inference_steps: 25,
      guidance_scale: 7,
    },
    options: {
      wait_for_model: true,
    },
  };

  const maxAttempts = 10;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    for (const url of endpoints) {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${HF_TOKEN}`,
          "Content-Type": "application/json",
          Accept: "image/png",
        },
        body: JSON.stringify(body),
      });

      const ct = res.headers.get("content-type") || "";

      if (res.ok && ct.startsWith("image/")) {
        const buf = Buffer.from(await res.arrayBuffer());
        await fs.promises.writeFile(outPath, buf);
        return;
      }

      // Try to read JSON error (HF often returns JSON with loading info)
      let msg = `${res.status} ${res.statusText}`;
      try {
        if (ct.includes("application/json")) {
          const j = await res.json();
          msg = j.error || JSON.stringify(j);

          // Model loading case
          if (res.status === 503 && (j.estimated_time || j.estimated_time === 0)) {
            const waitMs = Math.max(2000, Math.ceil((j.estimated_time || 5) * 1000));
            await new Promise((r) => setTimeout(r, waitMs));
            continue; // retry
          }
        } else {
          msg = await res.text();
        }
      } catch (_) {}

      // Retry on rate limit / temporary errors
      if (res.status === 503 || res.status === 429) {
        await new Promise((r) => setTimeout(r, 2000 + attempt * 600));
        continue;
      }

      // If endpoint not found, try the next endpoint in endpoints[]
      if (res.status === 404) continue;

      // Hard failure
      throw new Error(`HF image generation failed: ${msg}`);
    }

    // If both endpoints failed this attempt, wait a bit and retry
    await new Promise((r) => setTimeout(r, 1500 + attempt * 400));
  }

  throw new Error("HF image generation failed after retries.");
}

function renderSegment({ imagePath, outPath, segDur, W, H, title, caption }) {
  const fps = 30;
  const frames = Math.max(1, Math.round(segDur * fps));
  const safeCaption = escDrawtext(caption);
  const safeTitle = escDrawtext(title);

  const titleSize = Math.round(H * 0.055);
  const capSize = Math.round(H * 0.034);

  const filter = [
    `scale=${W}:${H}:force_original_aspect_ratio=cover`,
    `crop=${W}:${H}`,
    `fps=${fps}`,
    `zoompan=z='min(zoom+0.0015,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=${frames}:s=${W}x${H}:fps=${fps}`,
    `fade=t=in:st=0:d=0.35`,
    `fade=t=out:st=${Math.max(0, segDur - 0.35)}:d=0.35`,
    `drawtext=text='${safeTitle}':fontcolor=white:fontsize=${titleSize}:x=48:y=40:shadowcolor=black:shadowx=2:shadowy=2`,
    `drawtext=text='${safeCaption}':fontcolor=white:fontsize=${capSize}:x=48:y=h-${capSize}-70:shadowcolor=black:shadowx=2:shadowy=2`,
  ].join(",");

  return new Promise((resolve, reject) => {
    ffmpeg()
      .input(imagePath)
      .inputOptions(["-loop 1"])
      .outputOptions([`-t ${segDur}`, `-r ${fps}`, "-pix_fmt yuv420p", "-movflags +faststart"])
      .videoFilters(filter)
      .on("end", resolve)
      .on("error", reject)
      .save(outPath);
  });
}

function concatSegments({ segments, outPath, durationSec, addBgAudio }) {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "ff_concat_"));
  const concatFile = path.join(tmpDir, "concat.txt");
  const lines = segments.map((p) => `file '${p.replaceAll("\\", "\\\\")}'`).join("\n");
  fs.writeFileSync(concatFile, lines, "utf8");

  return new Promise((resolve, reject) => {
    const cmd = ffmpeg()
      .input(concatFile)
      .inputOptions(["-f concat", "-safe 0"])
      .outputOptions(["-c:v libx264", "-preset veryfast", "-crf 20", "-pix_fmt yuv420p", "-movflags +faststart"]);

    if (addBgAudio) {
      cmd
        .input(`sine=frequency=220:duration=${durationSec}`)
        .inputOptions(["-f lavfi"])
        .outputOptions(["-c:a aac", "-b:a 128k", "-shortest"]);
    } else {
      cmd.outputOptions(["-an"]);
    }

    cmd.on("end", resolve).on("error", reject).save(outPath);
  });
}

async function runJob(jobId, payload) {
  try {
    if (!PUBLIC_BASE_URL) throw new Error("Missing PUBLIC_BASE_URL env var.");

    updateJob(jobId, { status: "rendering", progress01: 0.05, message: "Starting…" });

    const { W, H } = aspectToSize(payload.aspectRatio);
    const { w: genW, h: genH } = aspectToGenSize(payload.aspectRatio);

    const { timed, storyboard } = guessScenesFromPrompt(payload.prompt, payload.durationSec);
    updateJob(jobId, { storyboard, progress01: 0.10, message: "Building storyboard…" });

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "fluxframe_"));
    const imagePaths = [];

    for (let i = 0; i < timed.length; i++) {
      updateJob(jobId, {
        message: `Generating AI image ${i + 1}/${timed.length}…`,
        progress01: 0.10 + (i / timed.length) * 0.45,
      });

      const scene = timed[i];

      const scenePrompt =
        `${payload.prompt}\n\nSCENE: ${scene.title}\n${scene.text}\n` +
        `Style: neon cyber, cinematic lighting, high contrast, clean composition.`;

      const imgPath = path.join(tmpDir, `scene_${i}.png`);
      await hfGenerateImageToFile({ prompt: scenePrompt, width: genW, height: genH, outPath: imgPath });
      imagePaths.push(imgPath);
    }

    updateJob(jobId, { message: "Rendering motion scenes…", progress01: 0.60 });

    const segDur = payload.durationSec / timed.length;
    const segments = [];

    for (let i = 0; i < imagePaths.length; i++) {
      const segPath = path.join(tmpDir, `seg_${i}.mp4`);
      const scene = timed[i];

      await renderSegment({
        imagePath: imagePaths[i],
        outPath: segPath,
        segDur,
        W,
        H,
        title: scene.title,
        caption: scene.text,
      });

      segments.push(segPath);

      updateJob(jobId, {
        message: `Rendered scene ${i + 1}/${segments.length}…`,
        progress01: 0.60 + (i / segments.length) * 0.25,
      });
    }

    updateJob(jobId, { message: "Final render…", progress01: 0.90 });

    const finalPath = path.join(VIDEO_DIR, `${jobId}.mp4`);
    await concatSegments({
      segments,
      outPath: finalPath,
      durationSec: payload.durationSec,
      addBgAudio: !!payload.backgroundAudio,
    });

    const videoUrl = `${PUBLIC_BASE_URL}/api/video/${jobId}.mp4`;

    updateJob(jobId, {
      status: "complete",
      progress01: 1,
      message: "AI video complete.",
      videoUrl,
    });
  } catch (err) {
    updateJob(jobId, {
      status: "failed",
      progress01: 1,
      message: String(err?.message || err),
    });
  }
}

/* ------------------ ROUTES ------------------ */

app.post("/api/generate", (req, res) => {
  const body = req.body || {};
  const words = countWords(body.prompt);

  if (!body.prompt || words < 1) return res.status(400).send("Prompt is required");
  if (words > 2500) return res.status(400).send("Prompt exceeds 2500 words");
  if (typeof body.durationSec !== "number" || body.durationSec < 30 || body.durationSec > 120) {
    return res.status(400).send("Duration must be 30–120 seconds");
  }

  const jobId = makeJobId();

  jobs.set(jobId, {
    jobId,
    status: "queued",
    progress01: 0,
    message: "Queued",
    storyboard: [],
    videoUrl: null,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  });

  setTimeout(() => runJob(jobId, body), 10);
  res.json({ jobId });
});

app.get("/api/status", (req, res) => {
  const jobId = String(req.query.jobId || "");
  if (!jobId) return res.status(400).send("jobId required");

  const job = jobs.get(jobId);
  if (!job) return res.status(404).send("Job not found");

  res.json({
    jobId: job.jobId,
    status: job.status,
    progress01: job.progress01,
    message: job.message,
    storyboard: job.storyboard,
    videoUrl: job.videoUrl,
  });
});

app.get("/api/video/:file", (req, res) => {
  const file = (req.params.file || "").replaceAll("..", "");
  const full = path.join(VIDEO_DIR, file);
  if (!fs.existsSync(full)) return res.status(404).send("Video not found");
  res.setHeader("content-type", "video/mp4");
  res.sendFile(full);
});

app.get("/health", (_req, res) => res.send("ok"));

app.listen(PORT, () => console.log(`Backend running on port ${PORT}`));
