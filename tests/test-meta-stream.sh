#!/bin/sh
curl -v -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "stream": true,
  "model": "meta.llama-3.3-70b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue. describe in one sentence"
    }
  ]
}'