# oci-generative-ai

## Usage

```shell
uvicorn facade:app --reload --host 0.0.0.0
```

## Notes

working:

- `meta.llama-3.1-70b-instruct`
- `meta.llama-3.2-90b-vision-instruct`
- `meta.llama-3.3-70b-instruct`
- `cohere.command-r-08-2024`
- `cohere.command-r-plus-08-2024`

not working:

- Entity with key `meta.llama-3-70b-instruct` not found
- Entity with key `meta.llama-3.1-405b-instruct` not found
- Entity with key `meta.llama-3.2-11b-vision-instruct` not found
- Entity with key `cohere.command-r-plus` not found
- Entity with key `cohere.command-r-16k` not found
