# requires transformers and sentencepiece installations
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Download fine-tuned model (t5 fine-tuned japanese model)
MODEL_NAME2 = "Huaibo/t5_dialog_jp"
PATH2 = "/dltk/app/model/data/t5_dialog_jp"   # change ~ to some existing path
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME2)
tokenizer.save_pretrained(PATH2)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME2)
model.save_pretrained(PATH2)


# Download base model (t5 japanese model)
MODEL_NAME1 = "sonoisa/t5-base-japanese"
PATH1 = "/dltk/app/model/data/t5_jp" 
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME1)
tokenizer.save_pretrained(PATH1)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME1)
model.save_pretrained(PATH1)


