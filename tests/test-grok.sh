#!/bin/sh
curl -v -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
  "stream": false,
  "model": "xai.grok-3",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue. describe in one sentence"
    }
  ]
}'