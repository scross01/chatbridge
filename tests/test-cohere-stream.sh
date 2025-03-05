#!/bin/sh
curl -v -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "stream": true,
  "model": "cohere.command-r-08-2024",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue. describe in one sentence"
    }
  ]
}'