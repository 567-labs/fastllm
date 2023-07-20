# FastLLM

A collection of LLM services you can self-host via Docker or modal labs to support your application development.

## Objectives

The goal of this repository is to provide a series of Docker containers, or modal labs deployments of common patterns when using LLMs and provide endpoints that allow you to integrate easily with existing codebases that use the popular OpenAI API.

## Roadmap

- [ ] Support [GPT4All's Embedding API](https://docs.gpt4all.io/gpt4all_python_embedding.html) and match it to [OpenAI Embeddings](https://openai.com/v1/embedding)
- [ ] Support JSONFormer API to match it to `ChatCompletion` with `function_calls`
- [ ] Support Cross Encoders based on sentence transformers for any Hugging Face model
- [ ] Provide great documentation and runbooks using MkDocs

## Contributing

Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).