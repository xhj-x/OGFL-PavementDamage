class BboxLoss(nn.Module):
    def __init__(self, reg_max=16, use_direction_iou=True, use_shape_adaptive=True, training_strategy="precision_focused"):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.use_direction_iou = use_direction_iou
        self.use_shape_adaptive = use_shape_adaptive
        self.training_strategy = training_strategy

        # 训练参数
        self.current_epoch = 0
        self.total_epochs = 200

        # 在初始化时打印配置信息
        print("="*60)
        print("📊 Optimized BboxLoss Configuration:")
        print(f"   ✅ DFE Module: Always Enabled")
        print(f"   {'✅' if use_direction_iou else '❌'} Direction-aware IoU Loss (DL): {'Enabled' if use_direction_iou else 'Disabled'}")
        print(f"   {'✅' if use_shape_adaptive else '❌'} Adaptive Shape Classification (ASC): {'Enabled' if use_shape_adaptive else 'Disabled'}")
        print(f"   🎯 Training Strategy: {training_strategy}")
        print(f"   📝 Experiment Type: {self._get_experiment_name()}")
        print("="*60)

        # 记录到日志文件
        with open("optimized_ablation_log.txt", "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Optimized Experiment started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration:\n")
            f.write(f"  - DFE: Enabled\n")
            f.write(f"  - Direction IoU: {use_direction_iou}\n")
            f.write(f"  - Shape Adaptive: {use_shape_adaptive}\n")
            f.write(f"  - Training Strategy: {training_strategy}\n")
            f.write(f"  - Experiment Type: {self._get_experiment_name()}\n")
            f.write(f"{'='*60}\n")

    def _get_experiment_name(self):
        """根据配置返回实验名称"""
        if not self.use_direction_iou and not self.use_shape_adaptive:
            return "Baseline (DFE only)"
        elif self.use_direction_iou and not self.use_shape_adaptive:
            return "DFE + DL"
        elif not self.use_direction_iou and self.use_shape_adaptive:
            return "DFE + ASC"
        else:
            return f"DARC Full (DFE + DL + ASC) - {self.training_strategy}"

    def set_epoch(self, epoch):
        """设置当前训练轮次，用于动态调整参数"""
        self.current_epoch = epoch

    def _get_dynamic_parameters(self):
        """根据训练策略和当前轮次返回动态参数"""
        if not (self.use_direction_iou and self.use_shape_adaptive):
            # 单独使用时保持原参数
            return {
                'alpha_v': 1.5 if self.use_direction_iou else 1.0,
                'shape_weight': 0.10 if self.use_shape_adaptive else 0.0,
                'use_penalty': True
            }

        # ASC+DL组合时的优化策略
        progress = self.current_epoch / self.total_epochs if self.total_epochs > 0 else 0

        if self.training_strategy == "adaptive":
            # 自适应策略：随训练进度动态调整
            if progress < 0.3:  # 前30%：注重召回率
                return {'alpha_v': 1.15, 'shape_weight': 0.03, 'use_penalty': False}
            elif progress < 0.7:  # 中40%：平衡阶段
                return {'alpha_v': 1.25, 'shape_weight': 0.06, 'use_penalty': False}
            else:  # 后30%：注重精确度
                return {'alpha_v': 1.35, 'shape_weight': 0.08, 'use_penalty': True}

        elif self.training_strategy == "staged":
            # 分阶段策略：明确的阶段划分
            if progress < 0.4:
                return {'alpha_v': 1.1, 'shape_weight': 0.02, 'use_penalty': False}
            elif progress < 0.8:
                return {'alpha_v': 1.3, 'shape_weight': 0.05, 'use_penalty': False}
            else:
                return {'alpha_v': 1.5, 'shape_weight': 0.09, 'use_penalty': True}

        elif self.training_strategy == "precision_focused":
            # 新增：注重精确度的策略
            if progress < 0.2:
                return {'alpha_v': 1.2, 'shape_weight': 0.02, 'use_penalty': False}
            else:
                return {'alpha_v': 1.45, 'shape_weight': 0.07, 'use_penalty': True}

        elif self.training_strategy == "hybrid":
            # 混合策略：前期保持精确度，后期微调召回率
            if progress < 0.6:
                return {'alpha_v': 1.4, 'shape_weight': 0.06, 'use_penalty': True}
            else:
                return {'alpha_v': 1.3, 'shape_weight': 0.05, 'use_penalty': True}

        else:  # "balanced" 策略：针对精确度优化的平衡参数
            return {'alpha_v': 1.35, 'shape_weight': 0.04, 'use_penalty': True}

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, pred_cls=None, target_labels=None):

        # 保存fg_mask供后续使用
        self.fg_mask = fg_mask

        # 每隔800次迭代打印一次使用的损失函数
        if not hasattr(self, 'iter_count'):
            self.iter_count = 0
        self.iter_count += 1

        # 获取动态参数 - 放在最前面
        params = self._get_dynamic_parameters()

        if self.iter_count % 800 == 0:
            print(f"\n🔄 Iteration {self.iter_count} (Epoch {self.current_epoch}) - Active losses: {self._get_experiment_name()}")
            print(f"   📊 Dynamic params: alpha_v={params['alpha_v']:.2f}, shape_weight={params['shape_weight']:.3f}, penalty={params['use_penalty']}")

        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        # IoU损失部分 - 使用动态alpha_v参数
        if self.use_direction_iou:
            iou = direction_aware_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask],
                                      xywh=False, alpha_h=1.0, alpha_v=params['alpha_v'], CIoU=True)
            if self.iter_count == 1:  # 第一次迭代时打印
                print(f"   🎯 Using Direction-aware IoU with alpha_h=1.0, alpha_v={params['alpha_v']}")
        else:
            # 使用标准IoU
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            if self.iter_count == 1:
                print("   📦 Using Standard CIoU")

        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # 自适应形状分类损失 - 使用动态权重
        loss_shape = torch.tensor(0.0).to(pred_dist.device)
        if self.use_shape_adaptive and pred_cls is not None and target_labels is not None:
            try:
                # 注意：target_labels已经是前景的标签了，不需要再用fg_mask索引
                loss_shape = self.compute_shape_adaptive_loss(
                    pred_bboxes[fg_mask],  # 前景的预测框
                    pred_cls[fg_mask],     # 前景的预测分数
                    target_labels,         # 已经是前景的标签，不需要再索引
                    use_penalty=params['use_penalty']
                )
                loss_shape = loss_shape * params['shape_weight']

                if self.iter_count == 1:
                    print(f"   🔷 Shape loss computed successfully: {loss_shape.item():.4f} (weight: {params['shape_weight']:.3f})")
            except Exception as e:
                print(f"   ⚠️ Error computing shape loss: {e}")
                print(f"   Debug: pred_bboxes[fg_mask].shape = {pred_bboxes[fg_mask].shape}")
                print(f"   Debug: target_labels.shape = {target_labels.shape}")
                loss_shape = torch.tensor(0.0).to(pred_dist.device)
        elif self.iter_count == 1:
            print("   ⏭️  Shape Adaptive Loss: Disabled")

        # DFL损失
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                     target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        # 每1000次迭代记录损失值
        if self.iter_count % 1000 == 0:
            total_loss = loss_iou + loss_dfl + loss_shape
            print(f"\n📈 Loss values at iteration {self.iter_count}:")
            print(f"   - IoU Loss: {loss_iou.item():.4f}")
            print(f"   - DFL Loss: {loss_dfl.item():.4f}")
            print(f"   - Shape Loss: {loss_shape.item():.4f}")
            print(f"   - Total: {total_loss.item():.4f}")

            # 记录到日志
            with open("optimized_ablation_log.txt", "a") as f:
                f.write(f"Iter {self.iter_count}, Epoch {self.current_epoch}: ")
                f.write(f"IoU={loss_iou.item():.4f}, DFL={loss_dfl.item():.4f}, ")
                f.write(f"Shape={loss_shape.item():.4f}, Total={total_loss.item():.4f}\n")

        return loss_iou, loss_dfl, loss_shape