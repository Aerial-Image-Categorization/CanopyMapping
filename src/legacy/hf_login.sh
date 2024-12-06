#!/bin/bash
#read -p "Enter your Hugging Face API token: " hf_token
hf_token=$1
response="yes"

if huggingface-cli whoami &> /dev/null; then
  echo "A token is already saved on your machine."
  #read -p "Do you want to overwrite the existing token? (yes/no): " response
  if [[ "$response" != "yes" ]]; then
    echo "Keeping the existing token. Exiting."
    exit 0
  fi
  # Log out the current user
  huggingface-cli logout
fi

echo $hf_token | huggingface-cli login
