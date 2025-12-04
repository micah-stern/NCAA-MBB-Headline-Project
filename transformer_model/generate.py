# src/generate.py
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import argparse
from pathlib import Path


def generate_headline(model_path, input_text, max_length=50, num_return_sequences=1):
    """Generate headlines using the fine-tuned model."""
    # Load model and tokenizer
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = BartTokenizer.from_pretrained(model_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Generate output
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=3,
        num_return_sequences=num_return_sequences,
    )

    # Decode and return
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]


def main():
    parser = argparse.ArgumentParser(
        description="Generate headlines using fine-tuned BART"
    )
    parser.add_argument("--model_path", required=True, help="Path to fine-tuned model")
    parser.add_argument("--input_text", help="Input text for generation")
    parser.add_argument(
        "--input_file", help="File containing input texts (one per line)"
    )
    parser.add_argument("--output_file", help="File to save generated headlines")
    parser.add_argument(
        "--max_length", type=int, default=50, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to generate per input",
    )

    args = parser.parse_args()

    if args.input_text:
        headlines = generate_headline(
            args.model_path, args.input_text, args.max_length, args.num_return_sequences
        )
        print("\nGenerated Headlines:")
        for i, headline in enumerate(headlines, 1):
            print(f"{i}. {headline}")

    elif args.input_file and args.output_file:
        with open(args.input_file, "r", encoding="utf-8") as f_in, open(
            args.output_file, "w", encoding="utf-8"
        ) as f_out:

            for line in tqdm(f_in, desc="Generating headlines"):
                input_text = line.strip()
                if input_text:
                    headlines = generate_headline(
                        args.model_path,
                        input_text,
                        args.max_length,
                        args.num_return_sequences,
                    )
                    for headline in headlines:
                        f_out.write(f"{headline}\n")

        print(f"\nGenerated headlines saved to {args.output_file}")

    else:
        print(
            "Please provide either --input_text or both --input_file and --output_file"
        )


if __name__ == "__main__":
    main()
