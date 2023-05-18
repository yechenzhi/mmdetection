# GLIP: Grounded Language-Image Pre-training  

> [GLIP: Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)

<!-- [ALGORITHM] -->

## Abstract

This paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representation semantic-rich. In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs. The learned representations demonstrate strong zero-shot and few-shot transferability to various object-level recognition tasks. 1) When directly evaluated on COCO and LVIS (without seeing any images in COCO during pre-training), GLIP achieves 49.8 AP and 26.9 AP, respectively, surpassing many supervised baselines. 2) After fine-tuned on COCO, GLIP achieves 60.8 AP on val and 61.5 AP on test-dev, surpassing prior SoTA. 3) When transferred to 13 downstream object detection tasks, a 1-shot GLIP rivals with a fully-supervised Dynamic Head.

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/b87228d7-f000-4a5d-b103-fe535984417a"/>
</div>

## Results and Models

|   Model    | Zero-shot or Funetune | COCO mAP |       Pre-Train Data       |                           Config                           |  Download  |
|:----------:|:---------------------:|:--------:|:--------------------------:|:----------------------------------------------------------:|:----------:| 
| GLIP-T (A) |       Zero-shot       |   42.9   |            O365            | [config](glip_atss_swin-t_a_fpn_dyhead_pretrain_obj365.py) | [model]() |
| GLIP-T (B) |       Zero-shot       |   44.9   |            O365            | [config](glip_atss_swin-t_b_fpn_dyhead_pretrain_obj365.py) | [model]() |
| GLIP-T (C) |       Zero-shot       |   46.7   |         O365,GoldG         |                         [config]()                         | [model]() |
|   GLIP-T   |       Zero-shot       |   46.6   |    O365,GoldG,CC3M,SBU     |                         [config]()                         | [model]() |
|   GLIP-L   |       Zero-shot       |   51.4   | FourODs,GoldG,CC3M+12M,SBU |                         [config](glip_atss_swin-l_fpn_dyhead_pretrain_mixeddata.py)                         | [model]() |

Note: 

1. The weights corresponding to the zero-shot model are adopted from the official weights and converted using the [script](../../tools/model_converters/glip_to_mmdet.py). We have not retrained the model for the time being.
2. We will soon support fine-tuning on COCO.

