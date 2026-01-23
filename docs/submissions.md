# Submissions

> **⚠️ Your submission is a VIDEO** — recorded during live evaluation and uploaded to YouTube.
>
> **Required steps:**
> 1. Record video during live evaluation session
> 2. Upload to YouTube (Public or Unlisted — NOT Private)
> 3. Update `SUBMISSION.YAML` with video link
> 4. Push to `main` branch

## Live Evaluation Workflow

### Overview

| Phase | Data | Purpose |
|-------|------|---------|
| **Local Development** | Downloaded `hard` dataset | Build and test your solution |
| **Dev Server** (Day 2 early) | Simplified "toy" data | Connection practice only |
| **Final Server** (Day 2 later) | Hard difficulty | **Record your submission video** |

> **Connection instructions will be provided by end of Day 1.**

### Local Development

Develop using the downloaded `hard` dataset:

```bash
uv run brainstorm-stream --from-file data/hard/
uv run brainstorm-serve
```

### Dev Server — Connection Practice Only

> ⚠️ **The dev server streams simplified "toy" data.** Use it ONLY to verify your connection workflow works — NOT for developing your visualization.

```bash
# Terminal 1: Serve your web app
uv run brainstorm-serve

# Terminal 2: Send keyboard controls to server
uv run brainstorm-control --host <server-ip> --port 8765

# Update your app's WebSocket URL to: ws://<server-ip>:8765/stream
```

### Final Server — Record Your Video

The final server streams **hard-difficulty data** — same as local testing. This is where you record your submission video.

**Setup:**

```bash
# Terminal 1: Serve your web app
uv run brainstorm-serve

# Terminal 2: Send keyboard controls
uv run brainstorm-control --host <server-ip> --port 8765

# Update your app's WebSocket URL to: ws://<server-ip>:8765/stream
```

If you built a custom backend, update your backend's connection URL instead.

**Recording your video:**

1. Connect to our WiFi network at your designated time slot
2. Update your WebSocket URL to point to the server
3. Start the control client (Terminal 2)
4. Start screen recording (`Cmd+Shift+5` on Mac, `Win+G` on Windows)
5. Use arrow keys to interactively move the array
6. "Play the game" — optimize array placement using your visualization
7. Record 3-5 minutes demonstrating your solution
8. Add voiceover/edits afterward, then upload to YouTube

### Practice Sessions

- Connection instructions provided by end of Day 1
- Practice schedule posted on Day 2 (random team order)
- Each team gets an assigned time slot
- Limited attempts during practice (up to 5 tries)

## What to Submit

### 1. Video Demonstration (3-5 minutes)

**Requirements:**
- ⚠️ **Must be recorded during live evaluation** — connected to final server
- ⚠️ **Must show interactive array control** — you control array with arrow keys
- ⚠️ **Must be uploaded to YouTube** — Public or Unlisted (NOT Private)

**Content:**
- Show your app connected to the live server
- Demonstrate finding and tracking tuned regions
- Explain your approach and design decisions
- Include voice narration

You may edit the recording (trim, add annotations, voiceover), but core footage must be from live evaluation.

### 2. Code Repository

Your repository should include:
- Application source code
- README with setup instructions
- Dependencies file (`requirements.txt`, `package.json`, etc.)

### 3. Technical Summary (Optional)

A 1-2 page summary covering:
- Signal processing approach
- Visualization design rationale
- Performance characteristics
- OR design considerations

## Submitting Your Work

### Step 1: Upload Video to YouTube

Upload to **YouTube** with visibility set to **Public** or **Unlisted**.

> ❌ **Private videos will be rejected** — judges can't view them.

### Step 2: Update SUBMISSION.YAML

In your repository root, update `SUBMISSION.YAML`:

```yaml
submission_url: 'https://www.youtube.com/watch?v=YOUR_VIDEO_ID'
```

### Step 3: Push to Main Branch

```bash
git add SUBMISSION.YAML
git commit -m "Submit Track 2 solution"
git push origin main
```

You can update your submission anytime by pushing a new video link.

## Eligibility

To be **eligible for prizes**, your video must be recorded during **live evaluation** (final server with hard-difficulty data).

Teams using only downloaded data can submit for feedback, but only live-mode submissions are eligible for prizes.

## Common Errors

| Error | Problem | Solution |
|-------|---------|----------|
| Video not found | Private YouTube video | Change to Public or Unlisted |
| Invalid URL | Template URL still in file | Replace with your actual URL |
| Not recorded | Forgot to push | Run `git push origin main` |
| Wrong branch | Pushed to feature branch | Push to `main` |

## Judging Criteria

- **40% User Experience** — Intuitive, readable from distance, clear guidance
- **40% Technical Execution** — Accurate detection, low latency, handles noise
- **20% Innovation** — Novel visualization, creative signal processing

See [Overview](overview.md) for detailed criteria.

## Checklist

Before live evaluation:
- [ ] App works with `hard` difficulty locally
- [ ] Know how to change WebSocket URL for remote connection
- [ ] Provides clear directional guidance
- [ ] Readable from 6 feet away
- [ ] Has "found it" signal
- [ ] Screen recording software ready

During live evaluation:
- [ ] Connected to final server (not dev server)
- [ ] Control client running (`brainstorm-control`)
- [ ] Recording started
- [ ] Demonstrating interactive array movement
