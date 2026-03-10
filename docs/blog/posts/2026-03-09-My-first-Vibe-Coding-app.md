---
title: "I Built a Chess App in One Day Without Writing Code — Here's What I Learned"
description: My firsthand experience of developing an application using only vibe-coding
date: 2026-03-09
tags:
 - Vibecoding
 - Coding
 - LLM
 - AI
---

I've been obsessed with chess since I was a kid.

At first, I naively thought I could memorize all the possible opening lines — every trap, every blunder my opponents might stumble into. Then I learned that the number of possible chess games exceeds the number of atoms in the observable universe. That didn't kill my fascination; it deepened it.

Here's the thing: even grandmasters still brute-force memorize opening lines. So, 15 years ago, I bought my first chess book — [Mastering Chess Openings - Volume 1](https://m.media-amazon.com/images/I/71uYhDD1zBL._SY522_.jpg).

I couldn't even finish four chapters. The lines were hopelessly confusing. There are *transpositions* — the same board position can arise from completely different move orders. Between my day job and life, the book collected dust.

Years later, I discovered **Anki** and the concept of **Spaced Repetition**. A lightbulb moment: *What if someone built an app that converts opening lines into flashcards and lets you drill them?* That idea promptly sank to the bottom of my bucket list, filed under "someday."

AI brought "someday" to "today."

One fine Sunday, I cleared my desk and decided to vibe-code this chess app into existence. Now, this wasn't my first rodeo with AI-assisted development. I use coding assistants daily. But I'd handed over *complete project development* to agents twice before — an Android app and a webapp. Both times, despite Claude Opus being capable enough to [rewrite GCC from scratch](https://x.com/alexalbert__/status/1895535516064784400), it failed spectacularly on my modest little apps.

This time, I chose a different strategy: **active participation**.

> You can find the repo [here](https://github.com/0xba1a/learn-chess-openings). It's a completely offline app — no login required. Try it [here](https://eastrivervillage.com/learn-chess-openings/).

---

## Phase 1: Planning — The 90-Minute Debate

This phase was everything. Chess has unique intricacies that make naive data modeling a nightmare.

**The data model dilemma:** A document model struggles with transpositions. A relational model creates a tangled web of inbound and outbound edges, making it expensive to extract a single line. Neither option was clean.

**The alternative-move problem:** Imagine a user is practicing a puzzle and plays a *valid* chess move — just not the one the puzzle expects. If the app simply shows "Wrong!", the user is confused. They played a legitimate move! The system needs to recognize alternative continuations, gracefully redirect the user, and offer a path back to the intended line.

Beyond chess-specific headaches, there were questions about data growth, what to duplicate, what to normalize, and how to handle personalization (e.g., users naming the same line differently). Eventually, we — me and Claude — decided to go fully offline to keep things simple.

This discussion alone consumed over an hour. And critically, **I was reading every single word** the LLM responded with, counter-questioning relentlessly. The model did a solid job, but it had pitfalls. Those subtle mistakes? I would've completely missed them if I'd given the LLM full autonomy and waited for the final output.

After 90 minutes of intense back-and-forth, I asked the agent to consolidate everything into a high-level design document.

Then I took my first coffee break. I'd earned it.

---

## Phase 2: Design — The Fastest Document Review of My Life

The copilot produced a 20+ page design document. Here's where things got surreal.

In my day job, here's how this usually goes: after a brainstorming session, I'd tell the team, "Create the HLD capturing all our discussions and share it." Two to three days later, a first draft surfaces. I block two uninterrupted hours on my calendar to review it meticulously. A couple of rounds of back-and-forth in document comments follow. Maybe a live meeting. **The final document materializes in one to two weeks.**

Here? It was ready in 10 minutes.

*I* was the bottleneck. Reading a dense design document right after an intense brainstorming session is mentally brutal. And there was no "comment and wait for response" cycle — the AI fired answers back instantly. If I paused to check WhatsApp, the delay was entirely mine. The AI just sat there, patiently waiting.

Honestly? It was mildly guilt-tripping.

### Quality: Fast but Flawed

After about 45 minutes of reading and Q&A, the design document was finalized. Speed was the clear win. Quality? Just average. A good engineer would have done better.

The LLM missed some crucial points from our discussion. In some places, it chose approaches that were flat-out absurd — methods that wouldn't even work. When I pointed them out, it would shamelessly compliment my insight and correct its mistakes. I genuinely couldn't tell: was it an honest oversight, or was the LLM playing dumb to make me feel superior? *(If any Anthropic engineers read this — please answer me.)*

> **Final design document:** [design_document.md](https://github.com/0xba1a/learn-chess-openings/blob/main/docs/design_document.md)

### Resisting the Urge to Skip Ahead

After that exhausting review, every fiber of my being wanted to just say: *"Implement this."* But a wiser part of me knew — this design was complex enough that even a senior engineer would struggle to build it without proper breakdown. My previous failures had taught me not to rush.

So I asked the LLM to decompose the design into low-level modules. It identified five or six components. Then I asked for a detailed low-level design (LLD) for each one.

Here's where the HLD paid dividends. After designing Module 1 through multiple Q&A rounds, the context window was saturated with Module 1 details — earlier project-level discussions were lost to context compaction. **The design document served as persistent memory.** After completing each module's LLD, I'd ask the LLM to re-read the HLD and tackle the next module with the full picture in mind.

We ended up with eight modules. ([All module LLDs here.](https://github.com/0xba1a/learn-chess-openings/tree/main/docs/)) I reviewed every document thoroughly. It felt like one loooong meeting with an engineer who had instant answers to every question — where I was the only bottleneck.

This was the last phase where I held the steering wheel.

---

## Phase 3: Implementation — The Art of Letting Go

By afternoon, the HLD and all eight LLDs were ready. I told the copilot: *Go. Build it module by module.* I connected the Playwright MCP server so it could verify functionality after each module.

Then I made a deliberate, painful decision: **I stopped reading the code.**

Think about it — if I'm going to read and understand every line the LLM generates, why not just write the code myself? Reading kills the speed. Instead, I relied on the LLM for unit testing and asked it to pause at milestones for manual functional testing. If you write code for a living, you know how unnatural it feels to accept someone else's code without reviewing it yourself. This was an exercise in pure delegation.

### The Results: Mediocre Code, Magnificent Speed

The implementation was... average. Silly bugs cropped up: the undo button wasn't decrementing the move count; the board defaulted to white's turn even when the puzzle started with black. The LLM was a mediocre engineer. But it was a *fast* mediocre engineer.

My job in this phase boiled down to:

- Approving tool usage requests
- Clicking past "Copilot is running for longer. Do you want to continue?" dialogs
- Occasionally testing new features and catching regressions

**By evening, I had a working app** deployed on [GitHub Pages](https://eastrivervillage.com/learn-chess-openings/).

It felt... strange. An app that would've taken me two months as a side project materialized in a single day. The usual *"I built this with my own hands"* satisfaction was absent. In its place was something different — the satisfaction of *orchestration*.

---

## What I Learned

**AI isn't taking over. But conventional coding is evolving.**

The LLM was fast but dumb. I could've built this app without AI — but not in one day. And I seriously doubt AI could've built it from a one-line prompt like *"develop a webapp for learning chess openings."* The human in this loop was simultaneously **the bottleneck and the enabler**.

It was exhausting. A month's worth of development and design discussions compressed into a single day. It felt like running a marathon — not everyone can do it, and I definitely don't want to do it every day.

But this might become the norm. And if it does, the winners will be those with:

- **High endurance** — the stamina to sit through 4-hour AI sessions daily
- **Strong delegation instincts** — knowing when to intervene and when to let go
- **Deep domain knowledge** — only developers can converse with LLMs in their native language, catching subtle mistakes that non-technical users would miss

I used to wonder how PMs and executives trust project outcomes without reading a single line of code. Now I understand — it's a muscle. The muscle of trusting output from others and building systems to validate it.

**The shift for developers is clear:** focus on *what* to build, not *how*. Don't trust the LLM blindly. Don't micromanage every line of code. Find the balance. And above all — build the patience to read through every conversation and document the AI produces.

That's where the real skill lies now.