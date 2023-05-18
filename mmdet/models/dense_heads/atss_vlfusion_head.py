# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import Scale
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from transformers import BertConfig

from mmdet.registry import MODELS
from mmdet.structures.bbox import cat_boxes
from mmdet.utils import InstanceList
from ..utils import filter_scores_and_topk, select_single_mlvl, VLFuse, permute_and_flatten, BertEncoderLayer
from .atss_head import ATSSHead


class Conv3x3Norm(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=1,
                 use_dcn=False,
                 norm_type=None):
        super(Conv3x3Norm, self).__init__()

        if use_dcn:
            self.conv = ModulatedDeformConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups)
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups)

        if isinstance(norm_type, (list, tuple)):
            assert len(norm_type) == 2
            assert norm_type[0] == 'gn'
            gn_group = norm_type[1]
            norm_type = norm_type[0]

        if norm_type == 'bn':
            bn_op = nn.BatchNorm2d(out_channels)
        elif norm_type == 'gn':
            bn_op = nn.GroupNorm(
                num_groups=gn_group, num_channels=out_channels)
        if norm_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    def forward(self, x, **kwargs):
        x = self.conv(x, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DYReLU(nn.Module):

    def __init__(self,
                 inp,
                 oup,
                 reduction=4,
                 lambda_a=1.0,
                 K2=True,
                 use_bias=True,
                 use_spatial=False,
                 init_a=[1.0, 0.0],
                 init_b=[0.0, 0.0]):
        super(DYReLU, self).__init__()
        self.oup = oup
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        # print('init_a: {}, init_b: {}'.format(self.init_a, self.init_b))

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze), nn.ReLU(inplace=True),
            nn.Linear(squeeze, oup * self.exp), h_sigmoid())
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:  # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)

        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True) / 3
            out = out * ys

        return out


def permute_and_flatten(layer, N, A, C, H, W):
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


class DyConv(torch.nn.Module):

    def __init__(self,
                 conv_func,
                 in_channels=256,
                 out_channels=256,
                 use_dyfuse=True,
                 use_dyrelu=False,
                 use_dcn=False):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse:
            self.AttnConv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = h_sigmoid()
        else:
            self.AttnConv = None

        if use_dyrelu:
            self.relu = DYReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_dcn:
            self.offset = nn.Conv2d(
                in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.AttnConv is not None:
            for m in self.AttnConv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs):
        visual_feats = inputs['visual']
        language_dict_features = inputs['lang']

        next_x = []
        for level, feature in enumerate(visual_feats):

            conv_args = dict()
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]

            if level > 0:
                temp_fea.append(self.DyConv[2](visual_feats[level - 1],
                                               **conv_args))
            if level < len(visual_feats) - 1:
                temp_fea.append(
                    F.upsample_bilinear(
                        self.DyConv[0](visual_feats[level + 1], **conv_args),
                        size=[feature.size(2),
                              feature.size(3)]))
            mean_fea = torch.mean(torch.stack(temp_fea), dim=0, keepdim=False)

            if self.AttnConv is not None:
                attn_fea = []
                res_fea = []
                for fea in temp_fea:
                    res_fea.append(fea)
                    attn_fea.append(self.AttnConv(fea))

                res_fea = torch.stack(res_fea)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))

                mean_fea = torch.mean(
                    res_fea * spa_pyr_attn, dim=0, keepdim=False)

            next_x.append(mean_fea)

        next_x = [self.relu(item) for item in next_x]

        features_dict = {'visual': next_x, 'lang': language_dict_features}

        return features_dict


from mmengine.model import BaseModel

class VLFusionModule(BaseModel):

    def __init__(self,
                 in_channels,
                 feat_channels,
                 num_base_priors,
                 early_fuse=False,
                 num_dyhead_blocks=6,
                 lang_model_name='bert-base-uncased',
                 use_dyrelu=True,
                 use_dyfuse=True,
                 use_dcn=True,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_base_priors = num_base_priors
        self.early_fuse = early_fuse
        self.num_dyhead_blocks = num_dyhead_blocks
        self.use_dyrelu = use_dyrelu
        self.use_dyfuse = use_dyfuse
        self.use_dcn = use_dcn
        self.lang_cfg = BertConfig.from_pretrained(lang_model_name)
        self.lang_dim = self.lang_cfg.hidden_size
        self._init_layers()

    def _init_layers(self) -> None:
        bias_value = -math.log((1 - 0.01) / 0.01)
        num_dyhead_blocks = self.num_dyhead_blocks

        conv_func = lambda i, o, s: Conv3x3Norm(
            i, o, s, use_dcn=self.use_dcn, norm_type=['gn', 16])

        dyhead_tower = []
        for i in range(num_dyhead_blocks):
            if self.early_fuse:
                # cross-modality fusion
                dyhead_tower.append(VLFuse())
                # lang branch
                dyhead_tower.append(
                    BertEncoderLayer(
                        self.lang_cfg,
                        clamp_min_for_underflow=True,
                        clamp_max_for_overflow=True))

            # vision branch
            dyhead_tower.append(
                DyConv(
                    conv_func,
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    use_dyrelu=(self.use_dyrelu
                                and self.in_channels == self.feat_channels)
                    if i == 0 else self.use_dyrelu,
                    use_dyfuse=(self.use_dyfuse
                                and self.in_channels == self.feat_channels)
                    if i == 0 else self.use_dyfuse,
                    use_dcn=(self.use_dcn
                             and self.in_channels == self.feat_channels)
                    if i == 0 else self.use_dcn,
                ))

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self.bbox_pred = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 1, kernel_size=1)
        self.dot_product_projection_text = nn.Linear(
            self.lang_dim,
            self.num_base_priors * self.feat_channels,
            bias=True)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.bias_lang = nn.Parameter(
            torch.zeros(self.lang_dim), requires_grad=True)
        self.bias0 = nn.Parameter(
            torch.Tensor([bias_value]), requires_grad=True)
        self.scales = nn.ModuleList([Scale(1.0) for _ in range(5)])

    def forward(self, visual_feats: Tuple[Tensor], language_feats: dict):
        bbox_reg = []
        centerness = []

        feat_inputs = {'visual': visual_feats, 'lang': language_feats}

        dyhead_tower = self.dyhead_tower(feat_inputs)

        cls_logits = []

        if self.early_fuse:
            embedding = dyhead_tower['lang']['hidden']
        else:
            embedding = language_feats['embedded']

        embedding = F.normalize(embedding, p=2, dim=-1)
        dot_product_proj_tokens = self.dot_product_projection_text(embedding /
                                                                   2.0)
        dot_product_proj_tokens_bias = torch.matmul(
            embedding, self.bias_lang) + self.bias0

        for l, feature in enumerate(visual_feats):
            visual = dyhead_tower['visual'][l]
            B, C, H, W = visual.shape

            bbox_pred = self.scales[l](self.bbox_pred(visual))
            bbox_reg.append(bbox_pred)
            centerness.append(self.centerness(visual))

            dot_product_proj_queries = permute_and_flatten(
                visual, B, -1, C, H, W)

            A = dot_product_proj_queries.shape[1]
            bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)
            dot_product_logit = (
                torch.matmul(dot_product_proj_queries,
                             dot_product_proj_tokens.transpose(-1, -2)) /
                self.log_scale.exp()) + bias
            dot_product_logit = torch.clamp(dot_product_logit, max=50000)
            dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
            cls_logits.append(dot_product_logit)

        return bbox_reg, centerness, cls_logits


def convert_grounding_to_cls_scores(logits, positive_maps):
    assert len(positive_maps) == logits.shape[0]  # batch size

    scores = torch.zeros(logits.shape[0], logits.shape[1],
                         len(positive_maps[0])).to(logits.device)
    if positive_maps is not None:
        if all(x == positive_maps[0] for x in positive_maps):
            # only need to compute once
            positive_map = positive_maps[0]
            for label_j in positive_map:
                scores[:, :, label_j -
                       1] = logits[:, :,
                                   torch.LongTensor(positive_map[label_j]
                                                    )].mean(-1)
        else:
            for i, positive_map in enumerate(positive_maps):
                for label_j in positive_map:
                    scores[i, :, label_j - 1] = logits[
                        i, :, torch.LongTensor(positive_map[label_j])].mean(-1)
    return scores


@MODELS.register_module()
class ATSSVLFusionHead(ATSSHead):

    def __init__(self,
                 *args,
                 early_fuse=False,
                 num_dyhead_blocks=6,
                 lang_model_name='bert-base-uncased',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.head = VLFusionModule(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            num_base_priors=self.num_base_priors,
            early_fuse=early_fuse,
            num_dyhead_blocks=num_dyhead_blocks,
            lang_model_name=lang_model_name)

    def _init_layers(self) -> None:
        """No need to initialize the ATSS head layer."""
        pass

    def forward(self, visual_feats: Tuple[Tensor],
                language_feats: dict) -> Tuple[Tensor]:
        bbox_preds, centerness, cls_logits = self.head(visual_feats,
                                                       language_feats)
        return bbox_preds, centerness, cls_logits

    def predict(self,
                visual_feats: Tuple[Tensor],
                language_feats: dict,
                batch_data_samples,
                rescale: bool = True):
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        batch_token_positive_maps = [
            data_samples.token_positive_map
            for data_samples in batch_data_samples
        ]
        outs = self(visual_feats, language_feats)

        predictions = self.predict_by_feat(
            *outs,
            batch_img_metas=batch_img_metas,
            batch_token_positive_maps=batch_token_positive_maps,
            rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        bbox_preds: List[Tensor],
                        score_factors: List[Tensor],
                        cls_logits: List[Tensor],
                        batch_img_metas: Optional[List[dict]] = None,
                        batch_token_positive_maps: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        with_nms: bool = True) -> InstanceList:
        assert len(bbox_preds) == len(score_factors)
        num_levels = len(bbox_preds)

        featmap_sizes = [bbox_preds[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)

        result_list = []

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            token_positive_maps = batch_token_positive_maps[img_id]
            bbox_pred_list = select_single_mlvl(
                bbox_preds, img_id, detach=True)
            score_factor_list = select_single_mlvl(
                score_factors, img_id, detach=True)
            cls_logit_list = select_single_mlvl(
                cls_logits, img_id, detach=True)

            results = self._predict_by_feat_single(
                bbox_pred_list=bbox_pred_list,
                score_factor_list=score_factor_list,
                cls_logit_list=cls_logit_list,
                mlvl_priors=mlvl_priors,
                token_positive_maps=token_positive_maps,
                img_meta=img_meta,
                cfg=cfg,
                rescale=rescale,
                with_nms=with_nms)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                cls_logit_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                token_positive_maps: dict,
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = True,
                                with_nms: bool = True) -> InstanceData:
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        score_thr = cfg.get('score_thr', 0)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []

        for level_idx, (bbox_pred, score_factor, cls_logit, priors) in \
                enumerate(zip(bbox_pred_list,
                              score_factor_list, cls_logit_list, mlvl_priors)):
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(
                -1, self.bbox_coder.encode_size)
            score_factor = score_factor.permute(1, 2, 0).reshape(-1).sigmoid()

            scores = convert_grounding_to_cls_scores(
                logits=cls_logit.sigmoid()[None],
                positive_maps=[token_positive_maps])[0]

            results = filter_scores_and_topk(
                scores, score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))

            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            score_factor = score_factor[keep_idxs]
            scores = torch.sqrt(scores * score_factor)

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        predictions = self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

        if len(predictions) > 0:
            # Note: GLIP adopts a very strange bbox decoder logic,
            # and if 1 is not added here, it will not align with
            # the official mAP.
            predictions.bboxes[:, 2:] = predictions.bboxes[:, 2:] + 1
        return predictions
