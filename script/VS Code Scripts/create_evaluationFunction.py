from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import torch
import gc
from tqdm import tqdm

smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def eval_loop_fn(data_loader, model, processor, device):
    model.eval()
    running_loss = 0.0

    bleu1s, bleu2s, bleu3s, bleu4s, rougeLs = [], [], [], [], []

    tqdm_ob = tqdm(data_loader, total=len(data_loader), desc="Evaluating")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm_ob):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
            loss = outputs.loss

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)

            # Generate captions
            generated_ids = model.generate(pixel_values=pixel_values, max_length=128)
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            reference_texts = processor.batch_decode(labels, skip_special_tokens=True)

            for ref, hyp in zip(reference_texts, generated_texts):
                ref_tokens = nltk.word_tokenize(ref.lower())
                hyp_tokens = nltk.word_tokenize(hyp.lower())

                bleu1s.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
                bleu2s.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
                bleu3s.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
                bleu4s.append(sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

                rougeL = rouge.score(ref, hyp)["rougeL"].fmeasure
                rougeLs.append(rougeL)

            tqdm_ob.set_postfix({
                "val_loss": avg_loss,
                "bleu1": sum(bleu1s)/len(bleu1s),
                "bleu4": sum(bleu4s)/len(bleu4s),
                "rougeL": sum(rougeLs)/len(rougeLs),
            })

            del pixel_values, input_ids, labels
            torch.cuda.empty_cache()
            gc.collect()
            
    return {
        "val_loss": avg_loss,
        "bleu1": sum(bleu1s) / len(bleu1s),
        "bleu2": sum(bleu2s) / len(bleu2s),
        "bleu3": sum(bleu3s) / len(bleu3s),
        "bleu4": sum(bleu4s) / len(bleu4s),
        "rougeL": sum(rougeLs) / len(rougeLs),
    }