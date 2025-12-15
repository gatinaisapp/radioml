
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from transformers import AutoModel


class ImageTextModel(pl.LightningModule):
  def __init__(
    self,
    text_model="",
    image_model="vit_base_patch16_224"
    embed_dim=768,
    lr=1e-5,
    weight_decay=0.01
    ):
    
    
    super().__init__()
    
    self.save_hyperparameters()
    self.text_encorder=Automodel.from_pretrained(text_model)
    self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
    self.image_encoder = timm.create_model(image_model_name, pretrained=True, num_classes=embed_dim)

    self.lr=lr
    self.weight_decay=weight_decay
  def forward(self, images, input_ids, attention_mask):

    text_out= self.text_encoder(input_ids=input_ids, attention_mask= attention_mask)

    text_emb = F.normalize(self.text_proj(self.text_encoder(input_ids, attention_mask).last_hidden_state[:, 0]), dim=-1)
    image_emb= self.image_encoder(images) 
    image_emb= F.normalize(image_emb,dim=-1)
    return image_emb, text_emb

  def contrastive_loss(self, image_emb, text_emb):
    
    logits = torch.matmul(image_emb, text_emb.T) / self.temperature
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
   
  def training_step(self, batch, batch_idx):

    images, input_ids, attention_mask = batch
    image_emb, text_emb = self(images, input_ids, attention_mask)
    
    loss = self.contrastive_loss(image_emb, text_emb)
    self.log("train_loss", loss, prog_bar=True)
    
    return loss

  def validation_step(self, batch, batch_idx):
    
    images, input_ids, attention_mask = batch
    image_emb, text_emb = self(images, input_ids, attention_mask)
  
    loss = self.contrastive_loss(image_emb, text_emb)
    self.log("val_loss", loss, prog_bar=True)
    return loss

  def configure_optimizers(self):
      
    return torch.optim.AdamW(self.parameters(),lr=seflf.lr)



