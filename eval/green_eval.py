from green_score import GREEN
import json
import argparse
from tqdm import tqdm
import torch
import re

# refs = [
#     "Interstitial opacities without changes.",
#     "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
#     "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
# ]
# hyps = [
#     "Interstitial opacities at bases without changes.",
#     "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
#     "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
# ]

def clean_text(text):
    """ Remove excessive whitespace and replace empty text with '[EMPTY]' """
    text = text.strip()  # Remove leading/trailing spaces and newlines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'Indication:.*', '', text).strip()

    # Remove <|endoftext|>
    text = text.replace("<|endoftext|>", "")
    
    # Remove repeated sentences - enhanced version
    # Split on sentence endings, keeping the punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    unique_sentences = []
    seen_sentences = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Normalize sentence for comparison (remove punctuation and lowercase)
        normalized = re.sub(r'[.!?]+$', '', sentence).strip().lower()
        
        if normalized and normalized not in seen_sentences:
            unique_sentences.append(sentence)
            seen_sentences.add(normalized)
    
    # Join sentences with appropriate spacing
    text = ' '.join(unique_sentences)
    
    # Ensure proper sentence ending
    if text and not re.search(r'[.!?]$', text):
        text += '.'

    # Extract text within \boxed{} if it exists
    boxed_match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    if boxed_match:
        text_in_box = boxed_match.group(1).strip()
        text_in_box = re.sub(r'\s+', ' ', text_in_box) # Clean whitespace in the boxed text
        return text_in_box if text_in_box else "[EMPTY]" # Return cleaned boxed text or "[EMPTY]"
    
    return text if text else "[EMPTY]"  # Replace empty text with a placeholder

def main(data_path, save_path, model_name, use_vllm=False, batch_size=16, output_dir="."):
    # Load data
    print("Loading data...:{}".format(data_path))
    data = json.load(open(data_path))
    
    # Initialize GREEN scorer with vLLM and batch size support
    print(f"Initializing GREEN scorer with model: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Use vLLM: {use_vllm}")
    print(f"Batch size: {batch_size}")
    
    # Ensure fp32 precision for scoring to maintain accuracy
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Current device: {torch.cuda.current_device()}")
    
    try:
        green_scorer = GREEN(
            model_name=model_name,
            output_dir=output_dir,
            use_vllm=use_vllm,
            batch_size=batch_size
        )
        print(f"GREEN scorer initialized successfully with backend: {'vLLM' if use_vllm else 'HuggingFace'}")
    except Exception as e:
        print(f"Error initializing GREEN scorer: {str(e)}")
        raise e
    
    # Loop through the JSON data
    img_ids = []
    refs = []
    hyps = []

    # itemized data
    # count = 0
    for item in tqdm(data, desc="Loading data items"):
        if 'id' not in item:
            img_ids.append(item["dicom_id"])
        else:
            img_ids.append(item["id"])
        refs.append(clean_text(item["answer"]))
        hyps.append(clean_text(item["response"]))
        # count += 1
        # if count == 99:
        #     break

    print(f"Total items loaded: {len(refs)}")
    print(f"Computing GREEN scores...")
    
    # full data - compute GREEN scores with batching for efficiency
    mean, std, green_score_list, summary, result_df = green_scorer(refs, hyps)
    
    print(f"GREEN Score Statistics:")
    print(f"Mean: {mean:.6f}")
    print(f"Std: {std:.6f}")
    print(f"Number of scores: {len(green_score_list)}")
    print(f"Sample scores: {green_score_list[:5]}")  # Show first 5 scores
    print(summary)
    
    # for index, row in result_df.iterrows():
    #     print(f"Row {index}:\n")
    #     for col_name in result_df.columns:
    #         print(f"{col_name}: {row[col_name]}\n")
    #     print('-' * 80)
    
    # Save results with high precision
    print(f"Saving results to: {save_path}")
    result_df.to_csv(save_path, index=False, float_format='%.6f')
    print("Evaluation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for entire dataset using GREEN scorer")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input JSON data file')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save the output CSV file')
    parser.add_argument('--model_name_or_path', type=str, default="StanfordAIMI/GREEN-radllama2-7b", 
                        help='Name or path to the judge model. Default is GREEN-radllama2-7b')
    parser.add_argument('--use_vllm', action='store_true', 
                        help='Use vLLM backend for faster inference instead of HuggingFace')
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for GREEN scorer (default: 16)')
    parser.add_argument('--output_dir', type=str, default=".", 
                        help='Output directory for GREEN scorer (default: current directory)')
    args = parser.parse_args()
    
    main(args.data_path, args.save_path, args.model_name_or_path, 
         args.use_vllm, args.batch_size, args.output_dir)

