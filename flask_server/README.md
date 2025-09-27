# RadGraph Text Modification Server

This server provides an API for modifying generated text based on RadGraph medical entity extraction and corrections.

## Overview

The system works in the following way:

1. The vLLM rollout worker generates text responses
2. The text is sent to this Flask server
3. The server uses RadGraph to extract medical entities from both the generated text and ground truth
4. The server returns these annotations to the rollout worker
5. The rollout worker uses the `process_annotations` function to modify the text based on the annotations
6. The modified text replaces the original generated text before continuing with the pipeline

## Setup

1. Install required dependencies:
   ```
   pip install flask radgraph requests
   ```

2. Make the server script executable:
   ```
   chmod +x run_server.sh
   ```

## Running the Server

Start the server with:
```
./run_server.sh
```

This will start the server on port 5000, accessible at `http://localhost:5000/extract`.

## API Usage

The server accepts POST requests with the following JSON format:

```json
{
  "generations": ["generated text 1", "generated text 2", ...],
  "ground_truth": "reference ground truth text"
}
```

And returns:

```json
{
  "gen_annotations": {
    "0": { 
      "text": "generated text 1",
      "entities": { ... },
      "data_source": null,
      "data_split": "inference"
    },
    "1": { ... }
  },
  "gt_annotations": {
    "0": {
      "text": "reference ground truth text",
      "entities": { ... },
      "data_source": null,
      "data_split": "inference"
    }
  }
}
```

## Integration with vLLM Rollout

To use this server with the vLLM rollout pipeline:

1. Start the Flask server
2. Ensure the `extract_flag` is set to `True` in your prompts.meta_info
3. Set the `extractor_server` URL in prompts.meta_info
4. Run your generation pipeline

The extraction and modification will be automatically applied to the generated text before it is returned. 