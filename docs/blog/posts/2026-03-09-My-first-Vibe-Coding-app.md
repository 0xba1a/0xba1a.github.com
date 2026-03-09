---
title: How to Vibe Code Effectively
description: My firsthand experience of developing an application using only vibe-coding
date: 2026-03-09
tags:
 - Vibecoding
 - Coding
 - LLM
 - AI
---

I had a great fascination for chess from my childhood. Earlier I thought of memorizing all the possible lines and mistakes the opponents would make on those lines (traps). But when I got to know that the number of possible lines is not even comprehensible compared to the number of atoms in the universe, my fascination grew even more. Still, I got to learn that even chess masters follow this method of learning chess opening lines. So, I purchased my first chess book 15 years back — ![Mastering Chess Openings - Volume 1](https://m.media-amazon.com/images/I/71uYhDD1zBL._SY522_.jpg)

But I couldn't even complete the first four chapters. Those lines were so confusing. There are transpositions — the same position can occur in different lines. With my day job and other responsibilities, dedicating so much time to chess was not very possible at that time.

Later I got to know about Anki, which introduced me to the "Spaced Repetition Algorithm". Then I thought, what if there is a good app that converts all the opening lines into flashcards and lets users practice? But that idea just stayed at the bottom of the bucket list for "someday".

The advent of AI and vibe-coding gave an awakening opportunity for all these "someday" projects lying in a coma. So, one fine Sunday, I cleared my desk and started this project. It's not actually my first-time vibe-coding experience. Though I use coding assistants regularly, I had only handed over complete project development to the agents twice before. Though Claude Opus has written GCC on its own, both times, it failed to write my applications — one Android app and another webapp.

Unlike those previous times, I decided to take active participation in the development this time. You can find the repo here — https://github.com/0xba1a/learn-chess-openings. It is a completely offline app that doesn't require any login. You can use it [here](https://eastrivervillage.com/learn-chess-openings/).

## Planning Phase

This was very important for me because chess has its own intricacies. The schema design was crucial. If we use a document data model, it would be hard to maintain transpositions. Using a relational model would create too many inbound and outbound edges. So, extracting a single line would be a heavy database operation.

A crucial problem was awareness of alternative lines. While practicing puzzles, the user may present a valid move but not the expected answer for the current puzzle. In that case, if the system returns an error, it would confuse the user — there is no way for them to identify which move the puzzle is expecting without revealing the actual move. So, the system should be aware of all the alternative moves and diverge to a different line based on the user's move. There should also be an option to alert the user and redirect them to the previous line.

Along with the chess-related nuances, I wanted to understand how much the database might grow, which information to duplicate and which not to. For example, different users may want to name the same line differently. We should honor that and at the same time, we shouldn't expose it to other users. Later, I (we — me and Claude) decided to use only offline storage to keep it simple.

This discussion alone took more than an hour. Importantly, I was actively reading every single word the LLM responded with in my VS Code Agent chat window and counter-questioning it back-to-back. Though the model did a great job, it had its own pitfalls. Those were the moments I would've missed if I had given complete autonomy to the LLM and waited for its final output — which would've been a total failure.

After the 90 minutes for detailed discussion, I asked the agent to write it down into a high-level-design document with all the discussion points captured in detail.

I took my first coffee break.