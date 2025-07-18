
import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
from efficientnet_pytorch.model import EfficientNet
from .mixstyle import MixStyle


# ==============================================================================
# ICAFusion CORE MODULES
# ==============================================================================
class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
    def forward(self, x):
        return x * self.bias

class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
    def forward(self, x1, x2):
        return x1 * self.w1 + x2 * self.w2

class CrossAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h
        self.que_proj_mod1 = nn.Linear(d_model, h * self.d_k)
        self.key_proj_mod1 = nn.Linear(d_model, h * self.d_k)
        self.val_proj_mod1 = nn.Linear(d_model, h * self.d_v)
        self.que_proj_mod2 = nn.Linear(d_model, h * self.d_k)
        self.key_proj_mod2 = nn.Linear(d_model, h * self.d_k)
        self.val_proj_mod2 = nn.Linear(d_model, h * self.d_v)
        self.out_proj_mod1 = nn.Linear(h * self.d_v, d_model)
        self.out_proj_mod2 = nn.Linear(h * self.d_v, d_model)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        self._init_weights()
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        mod1_fea_flat = x[0]
        mod2_fea_flat = x[1]
        b_s, nq = mod1_fea_flat.shape[:2]
        nk = mod1_fea_flat.shape[1]
        mod1_norm = self.LN1(mod1_fea_flat)
        mod2_norm = self.LN2(mod2_fea_flat)
        q_mod1 = self.que_proj_mod1(mod1_norm).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k_mod1 = self.key_proj_mod1(mod1_norm).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v_mod1 = self.val_proj_mod1(mod1_norm).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        q_mod2 = self.que_proj_mod2(mod2_norm).contiguous().view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)
        k_mod2 = self.key_proj_mod2(mod2_norm).contiguous().view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v_mod2 = self.val_proj_mod2(mod2_norm).contiguous().view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)
        att_mod1_by_mod2 = torch.matmul(q_mod1, k_mod2) / (self.d_k ** 0.5)
        att_mod1_by_mod2 = torch.softmax(att_mod1_by_mod2, -1)
        out_mod1_enhanced = torch.matmul(att_mod1_by_mod2, v_mod2).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_mod1_enhanced = self.out_proj_mod1(out_mod1_enhanced)
        att_mod2_by_mod1 = torch.matmul(q_mod2, k_mod1) / (self.d_k ** 0.5)
        att_mod2_by_mod1 = torch.softmax(att_mod2_by_mod1, -1)
        out_mod2_enhanced = torch.matmul(att_mod2_by_mod1, v_mod1).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out_mod2_enhanced = self.out_proj_mod2(out_mod2_enhanced)
        return [out_mod1_enhanced, out_mod2_enhanced]

class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop, loops_num=1):
        super(CrossTransformerBlock, self).__init__()
        self.loops = loops_num
        self.ln_before_mlp = nn.LayerNorm(d_model)
        self.crossatt = CrossAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp_mod1 = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.mlp_mod2 = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            nn.GELU(),
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )
        self.coeff_mod1_att_res = LearnableCoefficient()
        self.coeff_mod1_mlp_res = LearnableCoefficient()
        self.coeff_mod2_att_res = LearnableCoefficient()
        self.coeff_mod2_mlp_res = LearnableCoefficient()
    def forward(self, x):
        mod1_fea_flat = x[0]
        mod2_fea_flat = x[1]
        for loop in range(self.loops):
            out_mod1_att_enhanced, out_mod2_att_enhanced = self.crossatt([mod1_fea_flat, mod2_fea_flat])
            mod1_res_att = self.coeff_mod1_att_res(mod1_fea_flat) + out_mod1_att_enhanced
            mod2_res_att = self.coeff_mod2_att_res(mod2_fea_flat) + out_mod2_att_enhanced
            mod1_norm_for_mlp = self.ln_before_mlp(mod1_res_att)
            mod2_norm_for_mlp = self.ln_before_mlp(mod2_res_att)
            mod1_fea_flat = self.coeff_mod1_mlp_res(mod1_res_att) + self.mlp_mod1(mod1_norm_for_mlp)
            mod2_fea_flat = self.coeff_mod2_mlp_res(mod2_res_att) + self.mlp_mod2(mod2_norm_for_mlp)
        return [mod1_fea_flat, mod2_fea_flat]

class TransformerFusionBlock(nn.Module):
    def __init__(self, d_model, vert_anchors=7, horz_anchors=7, h=8, block_exp=4, n_layer=1):
        super(TransformerFusionBlock, self).__init__()
        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        d_k = d_model
        d_v = d_model
        self.pos_emb_mod1 = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb_mod2 = nn.Parameter(torch.zeros(1, vert_anchors * horz_anchors, self.n_embd))
        self.avgpool_mod1 = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.maxpool_mod1 = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
        self.weights_mod1_pool = LearnableWeights()
        self.avgpool_mod2 = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.maxpool_mod2 = nn.AdaptiveMaxPool2d((self.vert_anchors, self.horz_anchors))
        self.weights_mod2_pool = LearnableWeights()
        self.crosstransformer_blocks = nn.Sequential(*[
            CrossTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop=0.1, resid_pdrop=0.1, loops_num=1)
            for _ in range(n_layer)
        ])
        self.final_fusion_conv = nn.Sequential(
            nn.Conv2d(d_model * 2, d_model, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU()
        )
        self.coeff_skip_mod1 = LearnableCoefficient()
        self.coeff_skip_mod2 = LearnableCoefficient()
    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        street_view_fea = x[0]
        satellite_fea = x[1]
        bs_s, c_s, h_s_in, w_s_in = street_view_fea.shape
        bs_a, c_a, h_a_in, w_a_in = satellite_fea.shape
        street_view_pooled_avg = self.avgpool_mod1(street_view_fea)
        street_view_pooled_max = self.maxpool_mod1(street_view_fea)
        street_view_pooled = self.weights_mod1_pool(street_view_pooled_avg, street_view_pooled_max)
        satellite_pooled_avg = self.avgpool_mod2(satellite_fea)
        satellite_pooled_max = self.maxpool_mod2(satellite_fea)
        satellite_pooled = self.weights_mod2_pool(satellite_pooled_avg, satellite_pooled_max)
        street_view_flat = street_view_pooled.flatten(2).permute(0, 2, 1) + self.pos_emb_mod1
        satellite_flat = satellite_pooled.flatten(2).permute(0, 2, 1) + self.pos_emb_mod2
        street_view_enhanced_flat, satellite_enhanced_flat = self.crosstransformer_blocks([street_view_flat, satellite_flat])
        c_transformer_out = self.n_embd
        street_view_enhanced = street_view_enhanced_flat.permute(0, 2, 1).reshape(bs_s, c_transformer_out, self.vert_anchors, self.horz_anchors)
        satellite_enhanced = satellite_enhanced_flat.permute(0, 2, 1).reshape(bs_a, c_transformer_out, self.vert_anchors, self.horz_anchors)
        street_view_enhanced_upsampled = F.interpolate(street_view_enhanced, size=(h_s_in, w_s_in), mode='bilinear', align_corners=False)
        satellite_enhanced_upsampled = F.interpolate(satellite_enhanced, size=(h_a_in, w_a_in), mode='bilinear', align_corners=False)
        street_view_final = self.coeff_skip_mod1(street_view_fea) + street_view_enhanced_upsampled
        satellite_final = self.coeff_skip_mod2(satellite_fea) + satellite_enhanced_upsampled
        fusion_target_h, fusion_target_w = self.vert_anchors, self.horz_anchors
        if street_view_final.shape[2:] != (fusion_target_h, fusion_target_w):
            street_view_final = F.interpolate(street_view_final, size=(fusion_target_h, fusion_target_w), mode='bilinear', align_corners=False)
        if satellite_final.shape[2:] != (fusion_target_h, fusion_target_w):
            satellite_final = F.interpolate(satellite_final, size=(fusion_target_h, fusion_target_w), mode='bilinear', align_corners=False)
        fused_features = torch.cat([street_view_final, satellite_final], dim=1)
        fused_features = self.final_fusion_conv(fused_features)
        return fused_features

# ==============================================================================
# Helper functions
# ==============================================================================
def l2_normalize(x, dim=1, eps=1e-12):
    return x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps)

def cosine_similarity_map(feat_a, feat_g_adapted, dim=1):
    return torch.sum(feat_a * feat_g_adapted, dim=dim, keepdim=True)

def double_conv(in_channels, out_channels, dropout_rate=0.25):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate)
    )

# ==============================================================================
# EfficientNet Wrapper with MixStyle Injection
# ==============================================================================
class EfficientNetWithMixStyle(EfficientNet):
    def extract_features_multiscale(self, inputs, mixstyle_layers=None, mixstyle_injection_points=None):
        multiscale_features = {}
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        multiscale_features[0] = x
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if mixstyle_layers and mixstyle_injection_points and idx+1 in mixstyle_injection_points:
                ms_layer_idx = mixstyle_injection_points.index(idx+1)
                if ms_layer_idx < len(mixstyle_layers):
                    x = mixstyle_layers[ms_layer_idx](x)
            multiscale_features[idx + 1] = x
        return x, multiscale_features

# ==============================================================================
# CVM_VIGOR Model (FINAL CORRECTED VERSION)
# ==============================================================================
class CVM_VIGOR(nn.Module):
    def __init__(self, c_in_grd=3, c_in_sat=3, d_model=320, n_layer=2,
                 vert_anchors=16, horz_anchors=16, h=8, block_exp=4, FoV=360, dropout_rate=0.3,
                 use_mixstyle=False, mixstyle_p=0.5, mixstyle_alpha=0.1, mixstyle_mix='random'):
        super(CVM_VIGOR, self).__init__()
        
        circular_padding = (FoV == 360)
        self.grd_efficientnet = EfficientNetWithMixStyle.from_pretrained('efficientnet-b0', circular=circular_padding)
        self.sat_efficientnet = EfficientNetWithMixStyle.from_pretrained('efficientnet-b0', circular=False)
        
        self.use_mixstyle = use_mixstyle
        if self.use_mixstyle:
            # Inject after block 4 (key 4) and block 9 (key 9)
            self.mixstyle_injection_points = [4,9] 
            self.mixstyle_layers = nn.ModuleList([
                MixStyle(p=mixstyle_p, alpha=mixstyle_alpha, mix=mixstyle_mix)
                for _ in self.mixstyle_injection_points
            ])
            print(f"MixStyle enabled with '{mixstyle_mix}' strategy. Injecting after EfficientNet blocks: {self.mixstyle_injection_points}")

        # The final block of B0 (before the head) has 320 channels.
        self.grd_adapt_conv = nn.Conv2d(320, d_model, 1)
        self.sat_adapt_conv = nn.Conv2d(320, d_model, 1)

        self.transformer_fusion_block = TransformerFusionBlock(
            d_model=d_model, vert_anchors=vert_anchors, horz_anchors=horz_anchors,
            h=h, block_exp=block_exp, n_layer=n_layer
        )
        
        # Based on your debug output:
        # Key 16: 320 ch, Key 9: 112 ch, Key 4: 40 ch, Key 2: 24 ch, Key 0: 32 ch
        self.adapt_grd_feat_for_cos_sim_16 = nn.Conv2d(320, 512, 1)
        self.adapt_grd_feat_for_cos_sim_9  = nn.Conv2d(112, 320, 1)
        self.adapt_grd_feat_for_cos_sim_4  = nn.Conv2d(40, 160, 1)
        self.adapt_grd_feat_for_cos_sim_2  = nn.Conv2d(24, 80, 1)
        self.adapt_grd_feat_for_cos_sim_0  = nn.Conv2d(32, 40, 1)

        self.adapt_skip_sat_16 = nn.Conv2d(320, 320, 1)
        self.adapt_skip_sat_9  = nn.Conv2d(112, 112, 1)
        self.adapt_skip_sat_4  = nn.Conv2d(40,  40, 1)
        self.adapt_skip_sat_2  = nn.Conv2d(24,  24, 1)
        self.adapt_skip_sat_0  = nn.Conv2d(32,  32, 1)

        self.deconv6 = double_conv(d_model, 512, dropout_rate=dropout_rate)
        self.conv6 = double_conv(512 + 1 + 320 + 1, 640, dropout_rate=dropout_rate)
        
        self.deconv5 = nn.ConvTranspose2d(640, 320, 2, 2)
        self.conv5 = double_conv(320 + 1 + 112 + 1, 320, dropout_rate=dropout_rate)

        self.deconv4 = nn.ConvTranspose2d(320, 160, 2, 2)
        self.conv4 = double_conv(160 + 1 + 40 + 1, 160, dropout_rate=dropout_rate)

        self.deconv3 = nn.ConvTranspose2d(160, 80, 2, 2)
        self.conv3 = double_conv(80 + 1 + 24 + 1, 80, dropout_rate=dropout_rate)

        self.deconv2 = nn.ConvTranspose2d(80, 40, 2, 2)
        self.conv2 = double_conv(40 + 1 + 32 + 1, 40, dropout_rate=dropout_rate)

        self.deconv1 = nn.ConvTranspose2d(40, 16, 2, 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, stride=1, padding=1)
        )

    def forward(self, grd, sat, semantic_mask):
        mixstyle_layers_to_pass = self.mixstyle_layers if self.use_mixstyle else None
        mixstyle_points_to_pass = self.mixstyle_injection_points if self.use_mixstyle else None
        
        grd_feature_volume, multiscale_grd_raw = self.grd_efficientnet.extract_features_multiscale(
            grd, mixstyle_layers=mixstyle_layers_to_pass, mixstyle_injection_points=mixstyle_points_to_pass
        )
        sat_feature_volume, multiscale_sat_raw = self.sat_efficientnet.extract_features_multiscale(
            sat, mixstyle_layers=mixstyle_layers_to_pass, mixstyle_injection_points=mixstyle_points_to_pass
        )

        grd_features_adapted_for_transformer = self.grd_adapt_conv(grd_feature_volume)
        sat_features_adapted_for_transformer = self.sat_adapt_conv(sat_feature_volume)

        fused_features = self.transformer_fusion_block([grd_features_adapted_for_transformer, sat_features_adapted_for_transformer])
        
        decoder_stages = [
            {'deconv': self.deconv6, 'conv': self.conv6, 'adapt_grd': self.adapt_grd_feat_for_cos_sim_16, 'grd_raw': multiscale_grd_raw[16], 'adapt_sat': self.adapt_skip_sat_16, 'sat_raw': multiscale_sat_raw[16]},
            {'deconv': self.deconv5, 'conv': self.conv5, 'adapt_grd': self.adapt_grd_feat_for_cos_sim_9,  'grd_raw': multiscale_grd_raw[9],  'adapt_sat': self.adapt_skip_sat_9,  'sat_raw': multiscale_sat_raw[9]},
            {'deconv': self.deconv4, 'conv': self.conv4, 'adapt_grd': self.adapt_grd_feat_for_cos_sim_4,  'grd_raw': multiscale_grd_raw[4],  'adapt_sat': self.adapt_skip_sat_4,  'sat_raw': multiscale_sat_raw[4]},
            {'deconv': self.deconv3, 'conv': self.conv3, 'adapt_grd': self.adapt_grd_feat_for_cos_sim_2,  'grd_raw': multiscale_grd_raw[2],  'adapt_sat': self.adapt_skip_sat_2,  'sat_raw': multiscale_sat_raw[2]},
            {'deconv': self.deconv2, 'conv': self.conv2, 'adapt_grd': self.adapt_grd_feat_for_cos_sim_0,  'grd_raw': multiscale_grd_raw[0],  'adapt_sat': self.adapt_skip_sat_0,  'sat_raw': multiscale_sat_raw[0]}
        ]

        x = fused_features
        for stage_info in decoder_stages:
            a_k_prime = stage_info['deconv'](x)
            g_k_prime_adapted_channels = stage_info['adapt_grd'](stage_info['grd_raw'])
            target_h_w_cos = (a_k_prime.shape[2], a_k_prime.shape[3])
            
            if g_k_prime_adapted_channels.shape[2:] != target_h_w_cos:
                g_k_prime_for_cos_sim = F.interpolate(g_k_prime_adapted_channels, size=target_h_w_cos, mode='bilinear', align_corners=False)
            else:
                g_k_prime_for_cos_sim = g_k_prime_adapted_channels
            
            a_k_prime_norm = l2_normalize(a_k_prime)
            g_k_prime_for_cos_sim_norm = l2_normalize(g_k_prime_for_cos_sim)
            m_k_prime = cosine_similarity_map(a_k_prime_norm, g_k_prime_for_cos_sim_norm)
            
            skip_sat_adapted_channels = stage_info['adapt_sat'](stage_info['sat_raw'])
            if skip_sat_adapted_channels.shape[2:] != target_h_w_cos:
                skip_sat_for_concat = F.interpolate(skip_sat_adapted_channels, size=target_h_w_cos, mode='bilinear', align_corners=False)
            else:
                skip_sat_for_concat = skip_sat_adapted_channels
            
            mask_k = F.interpolate(semantic_mask, size=target_h_w_cos, mode='bilinear', align_corners=False)
            x = torch.cat([a_k_prime, m_k_prime, skip_sat_for_concat, mask_k], dim=1)
            
            x = stage_info['conv'](x)

        x = self.deconv1(x)
        x = self.conv1(x)

        logits_flattened = torch.flatten(x, start_dim=1)
        heatmap = torch.reshape(F.softmax(logits_flattened, dim=-1), x.size())
        
        return logits_flattened, heatmap