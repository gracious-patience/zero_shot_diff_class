from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
import torch
import random
import string
import numpy as np
import argparse
import os
import shutil
from mybot import bot
parser = argparse.ArgumentParser()

## args parser
parser.add_argument(
        "--initializer",
        type=str,
        nargs="?",
        help="word to initialize invesion"
    )
parser.add_argument(
        "--token",
        type=str,
        nargs="?",
        help="word to initialize invesion"
    )
parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="list with dataset names"
    )
parser.add_argument(
        "--experiment",
        type=str,
        nargs="?",
        help="experiment name. Must coincide with the filename with pics paths"
    )
args = parser.parse_args()
##

## add learnt token to vocab
def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
  
  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embeds = loaded_learned_embeds[trained_token]
  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  embeds.to(dtype)
  # add the token in tokenizer
  token = token if token is not None else trained_token
  while True:
    try:
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
        break
    except:
        token = f"<sks{random.choice(string.ascii_letters)}{id.split('.')[0]}>"

  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds
  return token
## experiment's constants
if args.dataset == 'linney':
    names = [ 'berry', 'bird', 'dog', 'flower']
elif args.dataset == 'FlowerClassification10':
    names = ['Aster', 'Daisy', 'Iris', 'Lavender', 'Lily', 'Marigold', 'Orchid', 'Poppy', 'Rose', 'Sunflower']
elif args.dataset == 'wildanimals':
    names = ['cheetah', 'fox', 'hyena', 'lion', 'tiger', 'wolf']
elif args.dataset == 'oxford_pets':
    names = ['Abyssinian','American_Bulldog','American_Pit_Bull_Terrier','Basset_Hound','Beagle','Bengal_cat','Birman','Bombay_cat','Boxer','British_Shorthair','Chihuahua','Egyptian_Mau','English_Cocker_Spaniel','English_Setter','German_Shorthaired','Great_Pyrenees','Havanese','Japanese_Chin','Keeshond','Leonberger','Maine_Coon','Miniature_Pinscher','Newfoundland_dog','Persian_cat','Pomeranian','Pug','Ragdoll','Russian_Blue','Saint_Bernard','Samoyed','Scottish_Terrier','Shiba_Inu','Siamese','Sphynx_cat','Staffordshire_Bull_Terrier','Wheaten_Terrier','Yorkshire_Terrier']

dir_with_clip_class_embeddings = args.dataset + "_embeddings"
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"

tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)

path_to_file_with_pics_paths = '/home/sharfikeg/my_files/diffusion_staff/' + args.experiment
##

## save embeddings of classes
print("Preparing class embeddings...")
# embeddings are saved in format: (num_of_classes, 768)
if os.path.isdir(dir_with_clip_class_embeddings):
    class_embeds = torch.load(dir_with_clip_class_embeddings+"/"+dir_with_clip_class_embeddings+".pt")
else:
    os.mkdir(dir_with_clip_class_embeddings)
    with torch.no_grad():
        class_embeds = []
        for name in names:
            # when name is a single word 
            input = tokenizer(name, return_tensors="pt")
            class_embeds.append(text_encoder(**input)['last_hidden_state'][0][1])
    class_embeds = torch.stack(class_embeds, dim=0)
    torch.save(class_embeds, dir_with_clip_class_embeddings+"/"+dir_with_clip_class_embeddings+".pt")
print("Class embeddings are ready!")
##

## classify
stats = {
    200: {"accuracy":0, "hits":0},
    300: {"accuracy":0, "hits":0},
    400: {"accuracy":0, "hits":0},
    500: {"accuracy":0, "hits":0},
    600: {"accuracy":0, "hits":0},
    700: {"accuracy":0, "hits":0}
}
number = 1

with open(path_to_file_with_pics_paths, 'r') as f:
    for line in f:
        label, id = line.split('/')[-2], line.split('/')[-1]
        result_path = 'inversion_' + label+id.split('.')[0]
        pic_dir = label+id.split('.')[0]
        os.mkdir(pic_dir)
        shutil.copy("/home/sharfikeg/my_files/diffusion_staff/"+line.split('\n')[0], pic_dir)
        

        # lauch textual inversion
        if not args.token:
            token = f"<sks{random.choice(string.ascii_letters)}{id.split('.')[0]}>"
        else:
            token = args.token
        print(token)
        os.system(f'accelerate launch --config_file /home/sharfikeg/.cache/huggingface/accelerate/default_config1.yaml textual_inversion.py --pretrained_model_name_or_path="{pretrained_model_name_or_path}" --train_data_dir="{pic_dir}" --learnable_property="object" --placeholder_token="{token}" --initializer_token="{args.initializer}" --resolution=512 --train_batch_size=1 --gradient_accumulation_steps=4 --max_train_steps=700 --learning_rate=5.0e-04 --scale_lr --lr_scheduler="constant" --lr_warmup_steps=0 --output_dir="{result_path}" --save_steps 100')
        for epoch in stats:
            new_token = f"<sks{random.choice(string.ascii_letters)}{id.split('.')[0]}>"
            new_token = load_learned_embed_in_clip(result_path+'/learned_embeds-steps-'+str(epoch)+'.bin', text_encoder, tokenizer, token = new_token)
            input = tokenizer(new_token, return_tensors="pt")
            new_embed = text_encoder(**input)['last_hidden_state'][0][1].unsqueeze(0)
            dists = torch.linalg.norm(class_embeds-new_embed, dim=1)
            pred = names[torch.argmin(dists).item()]
            print(f'Model predicted: {pred}')
            if pred == label:
                stats[epoch]['hits'] += 1
        message = f"exp: {args.experiment}\npic: {label} {id}\naccs: \n"
        for epoch in stats:
            stats[epoch]['accuracy'] = stats[epoch]['hits']/number
            message += str(epoch) + " "+str(stats[epoch]['accuracy']) + "\n"
        print(message)
        bot.send_message(message)
        number += 1
        shutil.rmtree(pic_dir)
        shutil.rmtree(result_path)
##
