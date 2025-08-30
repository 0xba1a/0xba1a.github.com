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

GPT models were introduced as creating models. Initial focus of all the LLMs were on text generation, summarization, image and video generation. But recently it got shifted towards coding. After heavy success of Anthropic Claude models among developers, all the LLM companies shifted their focus towards coding tasks. Major reason is it provides revenue option by enterprise subscription. The individually subscribed creator tools aren't generating as much enterprise subscriptions from the coding models.

OpenAI has released it's latest model GPT-5 last month. In line with the shift in the market, the new model has special importance to coding tasks. Cursor team has tested it internally before the release and vouched that it is the best among OpenAI models. In our benchmark also, we noticed substantial improvement in simple and moderate problem solving. But hard problems were failing in the same rate similar to Gpt-4.1.

> *We trained GPT-5 with developers in mind: we’ve focused on improving tool calling, instruction following, and long-context understanding to serve as the best foundation model for agentic applications.*
>
> <p align="right">- OpenAI </p>

It looks like the model was fine-tuned too much on coding related activities after the initial training. The model uses `sed -n` command heavily to read code. Though the context window is big, the model is not trying to read the entire code file using `cat`. Instead it prefers to read only a portion of the code between 100 to 200 lines to keep the context focused. Even though the agent has tool description of `edit_anthropic`, which is a usable for both reading and writing, the model chooses to use `sed -n` most of the times. This kind of bias towards cannot be achieved only by training on generally available text data. This is a result of heavy fine-tuning after the initial training.

OpenAI Cookbook has released a [GPT-5 prompting guide](https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide). It has a lot of interesting information about personality of the model, how to control its behaviors with API parameters and prompt heuristics. One interesting information mentioned in the guide was how cursor tuned the model for an optimal thought response. In most interaction in the agentic workflow, the model just need to perform the action. Understandably, the model sends only a function-call without any thoughts in the response. It is optimal for token cost. But it will make the debugging or auditing harder as sometimes we cannot understand why the model did that action. This explanation before the tool call is called `tool-preamble`.

The API provides an option to control the verbosity of the tool-preamble. With GPT-5, OpenAI provides an option to control the verbosity in the prompt itself with `<tool_preambles>` tag. OpenAI mentions that GPT-5 is more verbose by default. So, cursor team tried to reduce it using API but it become silent. So, they choose to use a combination approach. They call set the verbosity parameter in the API to minimal but asked the model to be expressive in the `<tool_preamble>` prompt. It provides optimum amount of explanation text before every tool calls.






 * Given a task list, I ask model to read and update one by one. It's still not following.
 * Cookbook is interesting
 * Thought edit_anthropic is provided, it was using `sed -n`
 * Cursor set verbosity low in the API and asked for verbose output in prompt
    * Write code for clarity first. Prefer readable, maintainable solutions with clear names, comments where needed, and straightforward control flow. Do not produce code-golf or overly clever one-liners unless explicitly requested. Use high verbosity for writing code and code tools.
 * Naturally intrspective. Asking to be more thorough in the prompt will be counter productive due to too many tool cals
 * Quote "Like GPT-4.1, GPT-5 follows prompt instructions with surgical precision"
 *