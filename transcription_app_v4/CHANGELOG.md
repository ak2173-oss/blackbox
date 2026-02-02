# Changelog

All notable changes to the BlackBox Transcription Hub are documented here.
Each entry includes what changed, why, technical details, sources of inspiration,
and any difficulties encountered during development.

---

## [v4.1] - 2026-02-02

### Added: Siri-style iOS 18 Border Glow Animation

**What:** Animated conic-gradient border glow around the browser viewport edges,
triggered when the USB audio recorder is detected.

**Why:** To give clear, visually striking feedback when the recorder device is
plugged in - matching the premium feel of PLAUD and iOS Siri animations.

**Inspiration/Reference:**
- iOS 18 Siri edge glow effect
- CodePen by firepanther: https://codepen.io/firepanther/pen/WNBZaEd
  - Uses conic-gradient with blur overlay technique
  - Colors: pink (#f652bb), blue (#0855ff), purple (#5f2bf6), orange (#ec882d)
  - We adopted the same 4-color palette

**Files changed:**
- `templates/index.html` - Added CSS, HTML overlay element, JS functions, and
  hook-in points to existing device detection flow

**Technical implementation:**
- CSS Houdini `@property --siri-angle` defines an animatable custom property (type: `<angle>`)
- `@keyframes siri-spin` rotates `--siri-angle` from 0deg to 360deg over 3 seconds
- Single `#siri-glow` div: `position: fixed; inset: 0; z-index: 9999; pointer-events: none`
- `border: 3px solid transparent` with `border-image: conic-gradient(from var(--siri-angle), ...) 1`
- `background: transparent` explicitly set so page content remains visible
- `::before` pseudo-element duplicates the border-image with `filter: blur(15px); opacity: 0.5` for soft glow
- Fade in/out via `opacity: 0` (default) -> `opacity: 1` (.active class), with `transition: opacity 0.6s ease`
- JS API: `showSiriAnimation()` adds `.active`, `hideSiriAnimation()` removes it

**Integration points (hooked into existing code):**
- `manualScan()` -> calls `showSiriAnimation()` when device found
- `checkDeviceStatus()` -> calls `showSiriAnimation()` on auto-detect
- `checkDeviceStatus()` -> calls `hideSiriAnimation()` on device disconnect
- `pollImportStatus()` -> calls `hideSiriAnimation()` when batch import completes
- `hideDeviceModal()` -> calls `hideSiriAnimation()` on modal close/cancel
- `resetModal()` -> calls `hideSiriAnimation()` as cleanup

**Difficulties and failed approaches:**

1. **Approach 1 - White mask overlay (FAILED)**
   - Two child divs inside a fixed full-viewport container
   - Each had `::before` (spinning conic-gradient) and `::after` (white background to mask center)
   - Result: `::after { background: white }` covered the entire page, making it a white screen
   - The gradient was visible at edges but all app content was hidden

2. **Approach 2 - CSS mask-composite (FAILED)**
   - Replaced white `::after` with `-webkit-mask-composite: xor` / `mask-composite: exclude`
   - Used padding trick: `padding: 3px` with mask on content-box vs full element
   - Result: mask-composite not rendering correctly in Chrome on user's system
   - Same white screen as approach 1

3. **Approach 3 - Four independent edge strips (FAILED)**
   - Four separate `position: fixed` divs (top/bottom/left/right), each 3px in size
   - No parent wrapper to avoid full-viewport element
   - Result: `overflow: hidden` on 3px containers clipped `filter: blur()` completely
   - Gradient was invisible - too thin to see without the blur glow

4. **Approach 4 - CSS Houdini border-image (SUCCESS)**
   - Single div with `border-image: conic-gradient(...)` and `background: transparent`
   - `@property` enables smooth CSS animation of the gradient angle
   - border-image only paints on the 3px border, center stays transparent
   - Blur glow via `::before` pseudo works because it's not clipped by overflow

**Other issues resolved:**
- Flask Jinja2 template caching served stale HTML with old white-mask CSS even after
  file was updated. Required full server kill (`pkill -9`) and restart. Multiple zombie
  processes on port 5000 needed cleanup.
- Animation played indefinitely because `hideDeviceModal()` didn't call `hideSiriAnimation()`.
  Fixed by adding the call to all modal dismiss paths.
- Tailwind CDN syntax issue: `bg-black/50` (newer syntax) didn't work, reverted to
  `bg-black bg-opacity-50` (compatible with CDN version in use).

---

## [v4.0] - 2026-01-28

### Added: USB Audio Recorder Detection & Batch Import

**What:** Auto-detect when USB audio recorder (D:/record) is plugged in, show popup
with file list, allow selecting files, choose Whisper model, and batch process them.

**Why:** Streamline the workflow of transferring recordings from a physical PLAUD-style
USB recorder into the transcription system without manual file copying.

**Inspiration:** PLAUD Note app's automatic device sync experience.

**Files changed:**
- `app.py` - Device monitoring thread, PowerShell detection, batch import processing
- `templates/index.html` - Device modal, file selection UI, progress tracking UI

**Technical implementation:**
- Background daemon thread polls every 3 seconds for device presence
- Two detection methods: direct WSL path (`/mnt/d/record`) with PowerShell fallback
- PowerShell commands from WSL: `Test-Path`, `Get-ChildItem`, `Copy-Item`
- Case-insensitive extension matching via `-imatch` in PowerShell
- Batch import runs in background thread, status polled via `/api/device/import/status`
- Per-file progress tracking: copying -> transcription -> complete

**Difficulties:**
- WSL2 doesn't auto-mount USB drives at `/mnt/d/`. Solved with PowerShell fallback.
- PowerShell `-match` was case-sensitive, missing .MP3 files. Fixed with `-imatch`.
- `status_callback` signature mismatch between pipeline and batch processor.
- Whisper model passed as string instead of object caused `'str' has no attribute 'transcribe'`.
  Fixed by temporarily setting `Config.WHISPER_MODEL` instead of passing model directly.
- Time estimates were wildly high (31 min). Changed from 60s/MB to 5s/MB.

### Added: Clickable Timestamps in Transcript

**What:** Click any transcript segment to jump audio playback to that exact time.

**Files changed:** `templates/project.html`

**Technical implementation:**
- Jinja2 server-side timestamp formatting (HH:MM:SS)
- `onclick="jumpToTime({{ segment.start }})"` on each transcript div
- `audioPlayer.currentTime = time; audioPlayer.play()` in JS

**Difficulties:**
- Initially tried JavaScript-only formatting but timestamps weren't rendering.
  Switched to Jinja2 server-side formatting.
- First implementation used a separate clickable button; changed to making the
  entire transcript segment div clickable for better UX.

### Changed: Time Formats to HH:MM:SS

**What:** All timestamps and durations display as `00:00:00` instead of `MM:SS` or raw seconds.

**Files changed:** `templates/project.html`, `templates/index.html`

### Changed: Date Format in Project Names

**What:** Changed from `20260128_162557` to `28-Jan-2026`.

**Files changed:** `pipeline.py` - `datetime.now().strftime("%d-%b-%Y")`

### Added: Automatic Whisper Model Unloading

**What:** After transcription completes, Whisper model is deleted from GPU memory
and `torch.cuda.empty_cache()` is called to free VRAM for Ollama.

**Files changed:** `pipeline.py`

### Added: Chat History Persistence

**What:** Q&A conversations saved per project and loaded on page revisit.

### Changed: UI Colors from Purple to Black (PLAUD-style)

**What:** Replaced purple accent colors with black/gray throughout all templates.

**Files changed:** `templates/index.html`, `templates/project.html`, `templates/upload.html`

**Technical implementation:**
- Custom Tailwind config overriding accent color palette:
  - accent-50: `#F9FAFB`, accent-100: `#F3F4F6`
  - accent-500: `#1F2937`, accent-600: `#111827`, accent-700: `#030712`

---

## [v2.0] - Initial Release

- GPU-accelerated transcription with faster-whisper
- Automatic speaker detection
- AI summarization powered by Phi-3 via Ollama
- Interactive Q&A chat
- Server-Sent Events for real-time progress
- Project search and filtering
- Centralized configuration with environment variables
