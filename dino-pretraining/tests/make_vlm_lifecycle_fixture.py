import json,sys
from pathlib import Path
import numpy as np,torch
from transformers import BertConfig,BertModel
from mr_dino.model import DinoVisionTransformer3D
root=Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True)
release=root/'release';(release/'training').mkdir(parents=True,exist_ok=True);(release/'mr_dino').mkdir(exist_ok=True)
backbone_args={'volume_size':[4,32,32],'patch_size':[2,16,16],'voxel_spacing_mm':[1.,.5,.5],'in_chans':1,'n_storage_tokens':2,'drop_path_rate':0.}
config={'format':'test_teacher','arch':'tiny','source':'model.teacher.backbone','space':'atlas_space','step':4,'stage':'highres','weights':'teacher.pt','backbone':backbone_args,'training_crops':{'global':[4,32,32]}}
(release/'config.json').write_text(json.dumps(config))
args={'arch':'tiny','space':'atlas_space','target_shape':[4,32,32],'target_spacing':[1.,.5,.5],'posterior_shift_mm':0.}
(release/'training/checkpoint_metadata.json').write_text(json.dumps({'args':args}))
(release/'mr_dino/model.py').write_text('from mr_dino.model import DinoVisionTransformer3D\ndef vit_tiny_3d(**kwargs):\n return DinoVisionTransformer3D(embed_dim=32,depth=2,num_heads=4,**kwargs)\n')
torch.manual_seed(12);model=DinoVisionTransformer3D(embed_dim=32,depth=2,num_heads=4,**backbone_args);model.init_weights();model.eval();torch.save(model.state_dict(),release/'teacher.pt')
cache=root/'cache';cache.mkdir(exist_ok=True);rows=[];sentences=[]
for i in range(7):
 uid=f's{i}';split='train' if i<5 else 'val';rows.append(f'b,p{i},{uid},{split}')
 with torch.no_grad():
  x=torch.randn(1,1,4,32,32);tokens,grid=model.prepare_tokens_with_masks(x)
  rope=model.rope_embed(D=grid[0],H=grid[1],W=grid[2])
  tokens=model.blocks[0](tokens,rope)[0].numpy().astype(np.float16)
 (cache/(uid+'.f16')).write_bytes(tokens.tobytes());np.save(cache/(uid+'.fg.npy'),np.ones(grid,dtype=bool))
 index={'format':'mrdino_penultimate_tiles_v1','dim':32,'prefix':3,'latent_file':uid+'.f16','foreground_file':uid+'.fg.npy','token_grid':list(grid),'chunks':[{'offset':0,'length':len(tokens),'start':[0,0,0],'grid':list(grid)}]}
 (cache/(uid+'.cache.json')).write_text(json.dumps(index))
 (cache/(uid+'.study.json')).write_text(json.dumps({'study_uid':uid,'release_config':config,'sequences':[{'cache':uid+'.cache.json','source':uid+'_t1w.nii.gz'}]}))
 sentences.append({'volume_name':uid,'valid_json':True,'extracted_sentences':['alpha beta','gamma delta'] if i%2 else ['beta gamma','delta alpha']})
(root/'split.csv').write_text('batch_id,patient_uid,study_uid,split\n'+'\n'.join(rows)+'\n');(root/'sentences.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sentences))
text=root/'bert';text.mkdir(exist_ok=True)
vocab=['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','alpha','beta','gamma','delta'];(text/'vocab.txt').write_text('\n'.join(vocab)+'\n');(text/'tokenizer_config.json').write_text(json.dumps({'tokenizer_class':'BertTokenizer','do_lower_case':True}))
BertModel(BertConfig(vocab_size=len(vocab),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,max_position_embeddings=64)).save_pretrained(text)
recipe={'mode':'offline','release':str(release),'cache_dir':str(cache),'splits_csv':str(root/'split.csv'),'sentences':str(root/'sentences.jsonl'),'normalization':str(root/'normalization.pt'),'training':{'epochs':1,'seed':17,'warmup_steps':0,'max_sentences':2,'max_tokens':16,'save_every':1,'position_freqs':2,'model':{'dim':8,'text_model':str(text),'lr_block':1e-4,'lr_proj':1e-3,'lr_text':1e-4,'lr_temp':1e-3,'wd':.01}}}
(root/'config.json').write_text(json.dumps(recipe));print('Lifecycle fixture ready',flush=True)
