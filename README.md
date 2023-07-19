# fastllm

A collection of LLM services you can self host via docker or modal labs to support your applications development

## Objectives

The goal of this repo is to provide a series of docker containers, or modal labs deployments of common patterns when using LLMs and provide endpoints that allows you to intergrate easily with existing codebases that use the popular openai api.

## Roadmap

* Support GPT4all's embedding api and match it to `openai.com/v1/embedding`
* Support JSONFormer api to match it to chatcompletion with function_calls
* Support Cross Encoders based on sentence transformers for any huggingface model
* Provide great documentation and runbooks using MkDocs