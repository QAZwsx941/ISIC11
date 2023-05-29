import torch

# SR : Segmentation Result

# GT : Ground Truth
#
# 当训练模型时，通常会计算一些评估指标来评估模型的性能。在给定的代码中，以下是要计算的评估指标：
#
# - `acc`（准确率）：准确率是分类模型中最常用的评估指标之一，表示模型预测正确的样本数与总样本数之间的比例。
# 在这里，准确率表示预测结果与真实标签完全匹配的像素点的比例。
#
# - `SE`（敏感性/召回率）：敏感性也称为召回率，衡量模型对正样本的识别能力。
# 它表示模型正确预测为正样本的像素点数与真实正样本像素点数之间的比例。敏感性可以帮助评估模型在检测出目标时的性能。
#
# - `SP`（特异性）：特异性衡量模型对负样本的识别能力。
# 它表示模型正确预测为负样本的像素点数与真实负样本像素点数之间的比例。特异性可以帮助评估模型在排除非目标区域时的性能。
#
# - `PC`（精确度）：精确度衡量模型在预测为正样本的像素点中的准确性。
# 它表示模型正确预测为正样本的像素点数与模型预测为正样本的像素点数之间的比例。精确度可以帮助评估模型预测结果的质量。
#
# - `F1`（F1得分）：F1得分是精确度和召回率的加权平均值，综合考虑了模型的准确性和完整性。
# 它是常用的综合评估指标，将精确度和召回率进行平衡，可以帮助评估模型的整体性能。
#
# - `JS`（杰卡德相似度）：杰卡德相似度衡量模型预测结果与真实标签的相似程度。
# 它通过计算预测结果和真实标签的交集与并集之间的比例来衡量相似度。杰卡德相似度的取值范围为0到1，值越接近1表示预测结果与真实标签越相似。
#
# - `DC`（Dice系数）：Dice系数也是衡量预测结果与真实标签的相似程度的指标之一。
# 它通过计算预测结果和真实标签的两倍交集与预测结果和真实标签的总像素数之间的比例来衡量相似度。Dice系数的取值范围为0到1，值越接近1表示
#
# 预测结果与真实标签越相似。
#
# 这些评估指标在每个训练周期中计算，并用于监控模型的性能和训练进展。通过跟踪这些指标的变化，可以了解模型的训练情况和性能表现，并进行必要的调整和改进。

def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # 计算 SR 与 GT 相等的元素个数
    corr = torch.sum(SR == GT)
    # 计算张量的总大小
    tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    # 计算准确率
    acc = float(corr) / float(tensor_size)
    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FN : False Negative
    # 计算 TP（True Positive）：SR 和 GT 同时为 1 的元素个数
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    # 计算 FN（False Negative）：SR 为 0，GT 为 1 的元素个数
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    # 计算敏感度（召回率）
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # 计算 TN（True Negative）：SR 和 GT 同时为 0 的元素个数
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    # 计算 FP（False Positive）：SR 为 1，GT 为 0 的元素个数
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    # 计算特异度
    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)

    # TP : True Positive
    # FP : False Positive
    # 计算 TP（True Positive）：SR 和 GT 同时为 1 的元素个数
    TP = ((SR==1).byte()+(GT==1).byte()) == 2
    # 计算 FP（False Positive）：SR 为 1，GT 为 0 的元素个数
    FP = ((SR==1).byte()+(GT==0).byte()) == 2
    # 计算精确度
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    # 计算敏感度（召回率）
    SE = get_sensitivity(SR,GT,threshold=threshold)
    # 计算精确度
    PC = get_precision(SR,GT,threshold=threshold)
    # 计算 F1 分数
    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # 计算交，并集的元素个数
    
    Inter = torch.sum(SR.byte() + GT.byte() == 2)
    Union = torch.sum(SR.byte() + GT.byte() >= 1)
    # 计算 Jaccard 相似度
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient

    SR = SR > threshold
    GT = GT == torch.max(GT)
    # 计算交集的元素个数
    Inter = torch.sum(SR.byte()+GT.byte() == 2)
    # 计算 Dice 系数
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



