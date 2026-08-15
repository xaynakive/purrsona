# ✿ PurrSona

**A tiny ML pet that reads your cat's personality and styles an aesthetic to match.**

Fill in your cat's traits on a little Tamagotchi-style device, press *hatch*, and a machine-learning model predicts their personality — then dresses them for it with a matching outfit, accessory, travel bag, and signature color.

---

## ✿ What it does

- You describe your cat across three short pages: looks, personality, and daily life.
- A gradient-boosting model classifies them into one of five personality types (Royal, Bossy, Chaotic, Dreamy, or Shy).
- The result *hatches* on the screen with a matching aesthetic and a short written reading.

## ✿ The interesting part: the ML runs in your browser

PurrSona has **no backend and no server to keep awake.** The trained model is exported
to **ONNX** and runs entirely client-side with `onnxruntime-web` — the prediction happens
on your own machine, in the browser, for free, instantly, forever. Nothing to deploy,
nothing to pay for, nothing that falls asleep.

The original **Python / FastAPI** version of the service lives in [`/api`](./api) — the
same model served the classic way (scikit-learn + FastAPI + Docker), kept for reference.

## ✿ Built with

- **Frontend** — plain HTML, CSS, and JavaScript. Pixel Tamagotchi interface
  (Press Start 2P + VT323). No framework, no build step.
- **Model** — scikit-learn gradient-boosting classifier, exported to ONNX.
- **In-browser inference** — onnxruntime-web.
- **Reference backend** — FastAPI + Docker (in `/api`).


## ✿ Files

- `index.html` — the app (loads the model and predicts in-browser)
- `purrsona_model.onnx` — the trained model
- `purrsona_encoders.json` — the label mappings used to encode inputs
- `api/` — the reference Python/FastAPI version
