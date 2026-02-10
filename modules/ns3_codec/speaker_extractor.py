import os
import torch
from .facodec import FACodecEncoder, FACodecDecoder


class SpeakerExtractor(torch.nn.Module):
    def __init__(self, ckpt_path, device, dtype=torch.bfloat16) -> None:
        super().__init__()
        fa_encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        
        fa_decoder = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )
        
        fa_encoder.load_state_dict(torch.load(os.path.join(ckpt_path, "ns3_facodec_encoder.bin")))
        fa_decoder.load_state_dict(torch.load(os.path.join(ckpt_path, "ns3_facodec_decoder.bin")))
        
        fa_encoder.eval().to(device=device, dtype=dtype)
        fa_decoder.eval().to(device=device, dtype=dtype)
        
        self.fa_encoder = fa_encoder
        self.fa_decoder = fa_decoder
        
    def forward(self, waveform):
        # waveform, B,C,L
        
        enc_out = self.fa_encoder(waveform)
        vq_post_emb, vq_id, _, quantized, spk_embs = self.fa_decoder(enc_out, eval_vq=False, vq=True)
        
        return spk_embs    # B,256
        