---
title: GPT-5 For Programmers
description: What is special with OpenAI's GPT-5 for programmers?
date: 2025-09-01
tags:
 - LLM
 - OpenAI
 - GPT
 - GPT-5
 - Coding Agent
---
# GPT-5 For Programmers

![OpenAI shifted it's focus towards programmers](images/gpt-5-for-programmers.png)
GPT models started as tools for creating content like text generation, summarization, and even image or video generation. But recently, the focus has shifted towards coding. After the success of Anthropic's Claude models among developers, many LLM companies, including OpenAI, began prioritizing coding tasks. This shift is largely driven by the potential for enterprise subscriptions, as coding models generate more revenue compared to individual creator tools.

OpenAI launched GPT-5 last month, aligning with this market trend. The new model places a strong emphasis on coding capabilities. The Cursor team, after testing it internally, called it the best OpenAI model yet. Our benchmarks also showed significant improvements in solving simple and moderate problems. However, for harder problems, the success rate remains similar to GPT-4.1.

> *We trained GPT-5 with developers in mind: we’ve focused on improving tool calling, instruction following, and long-context understanding to serve as the best foundation model for agentic applications.*
>
> <p align="right">- OpenAI </p>

## Bias to tools

It seems GPT-5 was fine-tuned extensively for coding tasks after its initial training. The model frequently uses the `sed -n` command to read specific portions of code, typically between 100 to 200 lines. Though the new model is having a larger context window instead of reading the entire file with `cat` it chooses to stick with the `sed` command. This approach helps keep the context focused. Interestingly, even when the agent has access to a more versatile tool like `edit_anthropic` for both reading and writing, the model still prefers `sed -n` in most cases. This strong bias towards a specific tool suggests that the fine-tuning process emphasized specific coding related fine-tuning, going beyond general training on publicly available data.

## Levers to adjust

OpenAI Cookbook has published a [GPT-5 Prompting Guide](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide), which dives into the model's personality, API parameters, and prompt techniques. One highlight from the guide is how the Cursor team optimized the model's thought responses. In agentic workflows, the model often only needs to perform actions, so it typically sends a function call without explaining its reasoning. While this saves tokens, it can make debugging or auditing difficult since the reasoning behind actions isn't clear. This explanation, when included, is called a `tool-preamble`.

The GPT-5 inference API allows developers to adjust the verbosity of these tool-preambles. OpenAI also introduced a `<tool_preambles>` tag in prompts to control this directly. By default, GPT-5 is more verbose, so the Cursor team experimented with reducing verbosity via the API. However, this made the model too silent. They eventually adopted a hybrid approach: setting minimal verbosity in the API while asking the model to provide concise explanations using the `<tool_preambles>` tag. This balance ensures just enough context is provided before tool calls, offering flexibility for developers to fine-tune the behavior to their needs.

OpenAI has made significant strides in **Instruction Following** with GPT-5, claiming the model now follows prompts with **surgical precision**. While this is a powerful improvement, it comes with a caveat: poorly crafted prompts can lead to unintended results. Essentially, the model prioritizes the given instructions over its own reasoning, putting the responsibility—and control—squarely in the hands of developers.

However, there’s a quirk. When given a list of tasks and instructed to handle them one at a time—completing the first task before moving to the next—the model tends to process all tasks together. This behavior can be problematic when tasks are independent and require a fresh context. To work around this, developers often spin up a new agent instance for each task, which adds some overhead but ensures clean execution. While not perfect, this approach highlights the importance of understanding and adapting to the model’s behavior.

Another big leap with GPT-5 is its **reasoning ability**. By default, the model is thorough—it reads multiple files to gather all the context it needs before making changes. We've seen this firsthand in our workflows, and it's a game-changer for solving simple to moderate problems. While this approach uses more tokens, the quality of results makes it worth it. OpenAI calls this behavior **Naturally Introspective**.

However, not all workflows need the model to handle context building. Some agent setups already prepare precise tasks, resolving all the dependencies beforehand. In such cases, GPT-5’s eagerness to gather context again can be unnecessary. To address this, OpenAI offers two ways to manage it: the `<context_gathering>` tag in prompts or the `reasoning_effort` parameter in the API. Both options let developers fine-tune how much reasoning the model applies, ensuring it fits seamlessly into different workflows.

## Conclusion

Taken together, the improvements — Natural awareness of reading code iteratively, ability to control reasoning, precise instruction following — are clearly aimed at coding. That focus is actually good for developers: the new parameters and prompting techniques give you the levers to control the model. The LLM becomes another tool in your automation workflow arsenal, making previously brittle tasks (like web scraping) automatable quickly and even non‑linearly.