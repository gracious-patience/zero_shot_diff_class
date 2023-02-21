from diffusers import StableDiffusionPipeline
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
import torch
import numpy as np
import argparse
parser = argparse.ArgumentParser()

## args parser
parser.add_argument(
        "--checkpoint",
        type=str,
        nargs="?",
        help="path to learnt embedding"
    )
parser.add_argument(
        "--token",
        type=str,
        nargs="?",
        help="invented token"
    )
parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        help="dataset name"
    )
args = parser.parse_args()
##

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
learned_embeds_path = "/home/sharfikeg/my_files/inversion/" + args.checkpoint

tokenizer = CLIPTokenizer.from_pretrained(
    pretrained_model_name_or_path,
    subfolder="tokenizer",
)
text_encoder = CLIPTextModel.from_pretrained(
    pretrained_model_name_or_path, subfolder="text_encoder"
)


def load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer, token=None):
  loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
  
  # separate token and the embeds
  trained_token = list(loaded_learned_embeds.keys())[0]
  embeds = loaded_learned_embeds[trained_token]

  print(trained_token)

  # cast to dtype of text_encoder
  dtype = text_encoder.get_input_embeddings().weight.dtype
  embeds.to(dtype)
  new_embed = embeds
  # add the token in tokenizer
  token = token if token is not None else trained_token
  num_added_tokens = tokenizer.add_tokens(token)
  if num_added_tokens == 0:
    raise ValueError(f"The tokenizer already contains the token {token}. Please pass a different `token` that is not already in the tokenizer.")
  
  # resize the token embeddings
  text_encoder.resize_token_embeddings(len(tokenizer))
  
  # get the id for the token and assign the embeds
  token_id = tokenizer.convert_tokens_to_ids(token)
  print(token_id)
  text_encoder.get_input_embeddings().weight.data[token_id] = embeds
load_learned_embed_in_clip(learned_embeds_path, text_encoder, tokenizer)

with torch.no_grad():
    input = tokenizer(args.token, return_tensors="pt")
    print(input)
    new_embed = text_encoder(**input)['last_hidden_state'][0][1]

    dists = []
    ## choose list of names
    if args.dataset == 'linney':
        names = [ 'berry', 'bird', 'dog', 'flower']
    elif args.dataset == 'flowers10':
        names = ['Aster', 'Daisy', 'Iris', 'Lavender', 'Lily', 'Marigold', 'Orchid', 'Poppy', 'Rose', 'Sunflower']
    elif args.dataset == 'wildanimals':
        names = ['cheetah', 'fox', 'hyena', 'lion', 'tiger', 'wolf']
    elif args.dataset == 'oxford_pets':
        names = ['Abyssinian','American_Bulldog','American_Pit_Bull_Terrier','Basset_Hound','Beagle','Bengal_cat','Birman','Bombay_cat','Boxer','British_Shorthair','Chihuahua','Egyptian_Mau','English_Cocker_Spaniel','English_Setter','German_Shorthaired','Great_Pyrenees','Havanese','Japanese_Chin','Keeshond','Leonberger','Maine_Coon','Miniature_Pinscher','Newfoundland_dog','Persian_cat','Pomeranian','Pug','Ragdoll','Russian_Blue','Saint_Bernard','Samoyed','Scottish_Terrier','Shiba_Inu','Siamese','Sphynx_cat','Staffordshire_Bull_Terrier','Wheaten_Terrier','Yorkshire_Terrier']
    ##
    for name in names:
        input = tokenizer(name, return_tensors="pt")
        embed = text_encoder(**input)['last_hidden_state'][0][1]
        dists.append(torch.linalg.norm(new_embed - embed).item())
            
        print(f"dist between original clip embed and {name} reconstruction clip embed: {torch.linalg.norm(new_embed - embed).item()}")
        print('====')
    print(f"CLIP's decision:{names[np.argmin(dists)]}")
